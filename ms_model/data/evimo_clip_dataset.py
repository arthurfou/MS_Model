"""Dataset EVIMO au format d'entraînement DEVO — pour le run couplé M2 (voir ../../../PLAN.md).

Produit des clips `(images, poses, disps, intrinsics[, gt_dyn_mask], scene_id)` consommables
par `DEVO/train_coupled.py`, à partir des npz EVIMO (events + depth + poses meta + mask).
Géométrie **auto-cohérente** : intrinsèques, poses, depth et voxels viennent tous du même npz.

Format attendu par le forward d'entraînement de DEVO (cf. train.py) :
    images     : (n, C, H, W) voxel grids (C = nb_time_bins)
    poses      : (n, 7) = [tx, ty, tz, qx, qy, qz, qw], caméra->monde (c2w) ; DEVO fait .inv()
    disps      : (n, H, W) disparité (= depth_scale / depth)
    intrinsics : (4,) = [fx, fy, cx, cy]

⚠️ CONVENTION DE POSE — la brique à valider empiriquement. `DEVO/error_explained.md` documente
un bug Sim3/orientation sur EVIMO Box Seq 00 : la convention des poses EVIMO est un piège connu.
L'extraction est isolée dans `evimo_cam_to_pose7()` — si la BA ne converge pas / l'ATE explose,
c'est le premier endroit à corriger (c2w vs w2c, ordre du quaternion, axes caméra OpenCV).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.masking import thicken_mask
from ms_model.data.evimo_dataset import downsample_mask
from ms_model.representations.voxel import make_voxel_sequence


def evimo_cam_to_pose7(frame: dict) -> np.ndarray:
    """meta['frames'][i] -> pose7 [tx,ty,tz, qx,qy,qz,qw] (caméra->monde, c2w).

    Point d'ajustement unique de la convention de pose (voir avertissement en tête de module).
    """
    c = frame["cam"]["pos"]
    t, q = c["t"], c["q"]
    return np.array([t["x"], t["y"], t["z"], q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)


def _load_sequence(npz_path, nb_time_bins, patch_size, depth_scale, provide_gt_mask,
                   mask_thicken_radius, min_depth):
    """Charge une séquence npz -> dict de tenseurs (voxels, poses, disps, intrinsics[, gt])."""
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    meta = data["meta"].item()
    frames = meta["frames"]

    ea = load_events(npz_path)
    frame_ts = np.array([f["ts"] for f in frames], dtype=np.float64)
    voxel_seq = make_voxel_sequence(ea, frame_ts, nb_of_time_bins=nb_time_bins)  # (N, C, H, W)
    N = voxel_seq.shape[0]  # = len(frames) - 1

    poses = np.stack([evimo_cam_to_pose7(frames[i]) for i in range(N)])  # (N, 7)

    depth = data["depth"][:N].astype(np.float32)          # (N, H, W), mm
    depth_m = depth / depth_scale
    disps = np.where(depth_m > (min_depth / depth_scale), 1.0 / np.clip(depth_m, 1e-6, None), 0.0)
    disps = disps.astype(np.float32)                      # (N, H, W)

    K = data["K"]
    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)

    out = {
        "voxels": voxel_seq,                              # (N, C, H, W) tensor
        "poses": torch.from_numpy(poses).float(),         # (N, 7)
        "disps": torch.from_numpy(disps),                 # (N, H, W)
        "intrinsics": torch.from_numpy(intrinsics),       # (4,)
        "N": N,
        "scene_id": str(npz_path.parent.parent.name) + "/" + npz_path.stem,
    }
    if provide_gt_mask:
        fm = load_evimo_mask(npz_path)
        if mask_thicken_radius > 0:
            fm = thicken_mask(fm, radius=mask_thicken_radius)
        gt = torch.stack([downsample_mask(fm.masks[i], patch_size) for i in range(N)])  # (N, Hp, Wp)
        out["gt_dyn"] = gt
    return out


class EvimoClipDataset(Dataset):
    """Clips de `n_frames` frames consécutives depuis des séquences EVIMO npz.

    Args:
        npz_paths: liste de chemins npz EVIMO.
        n_frames: longueur des clips.
        provide_gt_mask: si True, chaque item inclut le masque dynamique GT sous-échantillonné
            (pour la loss de masque supervisée du M2). Sinon, seule la pose-loss façonne le masque.
        stride_clips: pas entre débuts de clips (1 = chevauchement max).
    """

    def __init__(
        self,
        npz_paths: list[Union[str, Path]],
        n_frames: int = 15,
        nb_time_bins: int = 5,
        patch_size: int = 4,
        depth_scale: float = 1000.0,
        provide_gt_mask: bool = False,
        mask_thicken_radius: int = 0,
        stride_clips: int = 1,
        min_depth: float = 1.0,
    ) -> None:
        self.provide_gt_mask = provide_gt_mask
        self.n_frames = n_frames
        self.seqs = []
        self.index = []  # (seq_idx, start)

        for si, path in enumerate(npz_paths):
            print(f"[EvimoClipDataset] ({si + 1}/{len(npz_paths)}) {path}", flush=True)
            seq = _load_sequence(path, nb_time_bins, patch_size, depth_scale,
                                 provide_gt_mask, mask_thicken_radius, min_depth)
            self.seqs.append(seq)
            for start in range(0, seq["N"] - n_frames + 1, stride_clips):
                self.index.append((si, start))

        if not self.index:
            raise ValueError(f"Aucun clip de {n_frames} frames — séquences trop courtes ?")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        si, s = self.index[idx]
        seq = self.seqs[si]
        e = s + self.n_frames
        images = seq["voxels"][s:e].float()
        poses = seq["poses"][s:e]
        disps = seq["disps"][s:e]
        intrinsics = seq["intrinsics"]
        # blob compatible train_coupled : [..., scene_id] (+ gt_dyn avant scene_id si demandé)
        blob = [images, poses, disps, intrinsics]
        if self.provide_gt_mask:
            blob.append(seq["gt_dyn"][s:e])
        blob.append(seq["scene_id"])
        return blob
