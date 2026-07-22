"""Dataset EVIMO au format d'entraînement DEVO — pour le run couplé M2 (voir ../../../PLAN.md).

Produit des clips `(images, poses, disps, intrinsics[, gt_dyn_mask], scene_id)` consommables
par `DEVO/train_coupled.py`, à partir des npz EVIMO (events + depth + poses meta + mask).
Géométrie **auto-cohérente** : intrinsèques, poses, depth et voxels viennent tous du même npz.

Format attendu par le forward d'entraînement de DEVO (cf. train.py) :
    images     : (n, C, H, W) voxel grids (C = nb_time_bins)
    poses      : (n, 7) = [tx, ty, tz, qx, qy, qz, qw], caméra->monde (c2w) ; DEVO fait .inv()
    disps      : (n, H, W) disparité (= depth_scale / depth)
    intrinsics : (4,) = [fx, fy, cx, cy]

Chargement **paresseux** (lazy) :
- Les voxels sont lus par mmap depuis le cache disque `<npz>.voxels_bins{N}.npy` (préproduit par
  `EvimoSegDataset`) → slice de clip = quelques MB de disque, pas de voxelisation à la volée.
- depth / mask GT (compressés dans le npz, ~2 GB décompressés chacun) sont chargés à la demande
  au niveau **séquence**, avec un petit cache LRU par worker (`cache_size` séquences). Peak RAM
  ≈ num_workers × cache_size × ~2 GB (depth) + O(mask compressed).
- Poses et intrinsèques sont préchargés dans __init__ (petits).

⚠️ CONVENTION DE POSE — la brique à valider empiriquement. `DEVO/error_explained.md` documente
un bug Sim3/orientation sur EVIMO Box Seq 00 : la convention des poses EVIMO est un piège connu.
L'extraction est isolée dans `evimo_cam_to_pose7()` — si la BA ne converge pas / l'ATE explose,
c'est le premier endroit à corriger (c2w vs w2c, ordre du quaternion, axes caméra OpenCV).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ms_model.io.masks import FrameMasks
from ms_model.masking import thicken_mask
from ms_model.data.evimo_dataset import downsample_mask


def evimo_cam_to_pose7(frame: dict) -> np.ndarray:
    """meta['frames'][i] -> pose7 [tx,ty,tz, qx,qy,qz,qw] (caméra->monde, c2w).

    Point d'ajustement unique de la convention de pose (voir avertissement en tête de module).
    """
    c = frame["cam"]["pos"]
    t, q = c["t"], c["q"]
    return np.array([t["x"], t["y"], t["z"], q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)


def _voxel_cache_path(npz_path: Path, nb_time_bins: int) -> Path:
    return npz_path.with_suffix(f".voxels_bins{nb_time_bins}.npy")


def _read_meta_and_poses(npz_path: Path):
    """Lit la meta (frames -> poses c2w, K) sans charger events/depth/mask.

    Supporte EVIMO1 (.npz) et EVIMO2 (répertoire dataset_info.npz).
    Pour EVIMO2, les frames sans clé 'cam' (tracking perdu) sont écartées.
    """
    if npz_path.is_dir():
        fi = np.load(npz_path / "dataset_info.npz", allow_pickle=True)
        meta = fi["meta"].item()
        frames = [f for f in meta["frames"] if "cam" in f]
        K = np.asarray(fi["K"])
        return frames, K
    with np.load(npz_path, allow_pickle=True) as data:
        meta = data["meta"].item()
        frames = meta["frames"]
        K = np.asarray(data["K"])
    return frames, K


def _load_depth_and_mask(npz_path: Path, N: int, want_mask: bool):
    """Charge depth (+ mask si demandé) pour N frames — décompresse le npz (I/O lourd).

    Supporte EVIMO1 (.npz) et EVIMO2 (répertoire dataset_depth/mask.npz).
    Depth EVIMO2 : uint16 en mm (identique à EVIMO1).
    """
    if npz_path.is_dir():
        fd = np.load(npz_path / "dataset_depth.npz", allow_pickle=True)
        depth_keys = sorted(fd.keys())[:N]
        depth = np.stack([fd[k] for k in depth_keys]).astype(np.float32)
        mask = None
        if want_mask:
            fm = np.load(npz_path / "dataset_mask.npz", allow_pickle=True)
            mask_keys = sorted(fm.keys())[:N]
            mask = np.stack([fm[k] for k in mask_keys])
        return depth, mask
    with np.load(npz_path, allow_pickle=True) as data:
        depth = data["depth"][:N].astype(np.float32)          # (N, H, W) mm
        mask = data["mask"][:N] if want_mask else None
    return depth, mask


def _prepare_gt_dyn(mask_arr: np.ndarray, ts_frames: np.ndarray,
                    patch_size: int, mask_thicken_radius: int) -> torch.Tensor:
    """Convertit un array (N, H, W) de segmentation en score dynamique (N, Hp, Wp) sous-échantillonné."""
    fm = FrameMasks(masks=mask_arr, ts=ts_frames)
    if mask_thicken_radius > 0:
        fm = thicken_mask(fm, radius=mask_thicken_radius)
    return torch.stack([downsample_mask(fm.masks[i], patch_size) for i in range(len(fm.masks))])


class EvimoClipDataset(Dataset):
    """Clips de `n_frames` frames consécutives depuis des séquences EVIMO npz.

    Args:
        npz_paths: liste de chemins npz EVIMO.
        n_frames: longueur des clips.
        provide_gt_mask: si True, chaque item inclut le masque dynamique GT sous-échantillonné
            (pour la loss de masque supervisée du M2). Sinon, seule la pose-loss façonne le masque.
        stride_clips: pas entre débuts de clips (1 = chevauchement max).
        cache_size: nb de séquences retenues en RAM par worker (LRU, pour depth/mask/disps).
            2 = compromis RAM/thrashing raisonnable avec num_workers=4 et shuffle=True.
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
        cache_size: int = 2,
    ) -> None:
        self.provide_gt_mask = provide_gt_mask
        self.n_frames = n_frames
        self.nb_time_bins = nb_time_bins
        self.patch_size = patch_size
        self.depth_scale = depth_scale
        self.mask_thicken_radius = mask_thicken_radius
        self.min_depth = min_depth
        self.cache_size = max(1, int(cache_size))

        self.paths = [Path(p) for p in npz_paths]
        self.voxel_paths: list[Path] = []
        self.per_seq_N: list[int] = []
        self.poses_all: list[torch.Tensor] = []
        self.intrinsics_all: list[torch.Tensor] = []
        self.frame_ts_all: list[np.ndarray] = []
        self.scene_ids: list[str] = []
        self.index: list[tuple] = []  # (seq_idx, start)

        for si, path in enumerate(self.paths):
            vpath = _voxel_cache_path(path, nb_time_bins)
            if not vpath.exists():
                raise FileNotFoundError(
                    f"[EvimoClipDataset] cache voxel manquant : {vpath}\n"
                    f"    Le cache est produit automatiquement par EvimoSegDataset ; sinon "
                    f"le régénérer via make_voxel_sequence.")
            # header mmap only — no data pulled from disk yet
            N_vox = np.load(vpath, mmap_mode="r").shape[0]

            frames, K = _read_meta_and_poses(path)
            N = min(N_vox, len(frames) - 1)  # sécurité si meta un peu plus grand que voxel
            poses = torch.from_numpy(np.stack([evimo_cam_to_pose7(frames[i]) for i in range(N)])).float()
            intr = torch.from_numpy(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32))
            ts = np.array([f["ts"] for f in frames[:N]], dtype=np.float64)

            self.voxel_paths.append(vpath)
            self.per_seq_N.append(N)
            self.poses_all.append(poses)
            self.intrinsics_all.append(intr)
            self.frame_ts_all.append(ts)
            self.scene_ids.append(f"{path.parent.parent.name}/{path.stem}")
            for start in range(0, N - n_frames + 1, stride_clips):
                self.index.append((si, start))
            print(f"[EvimoClipDataset] ({si + 1}/{len(self.paths)}) meta {path.name} -> N={N} "
                  f"clips={max(0, N - n_frames + 1)}", flush=True)

        if not self.index:
            raise ValueError(f"Aucun clip de {n_frames} frames — séquences trop courtes ?")

        # Cache LRU depth+gt_dyn (par instance / par worker).
        self._cache: "OrderedDict[int, dict]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _get_heavy(self, si: int) -> dict:
        """Depth (disps) + gt_dyn pour la séquence si — chargés à la demande, cache LRU."""
        if si in self._cache:
            self._cache.move_to_end(si)
            return self._cache[si]
        N = self.per_seq_N[si]
        depth, mask = _load_depth_and_mask(self.paths[si], N, want_mask=self.provide_gt_mask)
        depth_m = depth / self.depth_scale
        disps = np.where(depth_m > (self.min_depth / self.depth_scale),
                         1.0 / np.clip(depth_m, 1e-6, None), 0.0).astype(np.float32)
        heavy = {"disps": torch.from_numpy(disps)}  # (N, H, W)
        if self.provide_gt_mask:
            heavy["gt_dyn"] = _prepare_gt_dyn(
                mask, self.frame_ts_all[si], self.patch_size, self.mask_thicken_radius)
        self._cache[si] = heavy
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return heavy

    def __getitem__(self, idx: int):
        si, s = self.index[idx]
        e = s + self.n_frames
        # voxels : mmap slice (quelques MB, jamais tout matérialisé)
        vox = np.array(np.load(self.voxel_paths[si], mmap_mode="r")[s:e])
        images = torch.from_numpy(vox).float()                        # (n, C, H, W)

        heavy = self._get_heavy(si)
        poses = self.poses_all[si][s:e]
        disps = heavy["disps"][s:e]
        # DEVO attend (n_frames, 4) — pas (4,) — car pops.transform fait intrinsics[:, ii].
        intrinsics = self.intrinsics_all[si].unsqueeze(0).expand(self.n_frames, 4).contiguous()
        blob = [images, poses, disps, intrinsics]
        if self.provide_gt_mask:
            blob.append(heavy["gt_dyn"][s:e])
        blob.append(self.scene_ids[si])
        return blob
