import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.masking import thicken_mask
from ms_model.representations.voxel import make_voxel_sequence


def downsample_mask(mask: np.ndarray, patch_size: int) -> torch.Tensor:
    """(H, W) -> (H//patch_size, W//patch_size), binaire (1 = patch contient un pixel dynamique)."""
    t = torch.from_numpy((mask != 0).astype(np.float32))[None, None]
    Hp, Wp = mask.shape[0] // patch_size, mask.shape[1] // patch_size
    out = F.adaptive_max_pool2d(t, (Hp, Wp))
    return out[0, 0]


class EvimoSegDataset(Dataset):
    """Séquences (voxel grids, masques dynamiques par patch) pour la motion segmentation.

    Les voxel grids sont mis en cache sur disque (.voxels_bins<N>.npy) et lus
    via numpy memmap : seules les slices accédées sont chargées en RAM, ce qui
    permet d'entraîner sur de grands datasets sans OOM.
    """

    def __init__(
        self,
        npz_paths: list[Union[str, Path]],
        seq_len: int = 16,
        nb_time_bins: int = 5,
        patch_size: int = 4,
        mask_thicken_radius: int = 0,
    ):
        self.seq_len = seq_len
        # (cache_path, mask_tensor) — voxels restent sur disque
        self.sequences: list[tuple[Path, torch.Tensor]] = []
        self.index: list[tuple[int, int]] = []

        for i, path in enumerate(npz_paths):
            t0 = time.time()
            print(f"[EvimoSegDataset] ({i + 1}/{len(npz_paths)}) chargement {path} ...", flush=True)

            fm = load_evimo_mask(path)
            if mask_thicken_radius > 0:
                fm = thicken_mask(fm, radius=mask_thicken_radius)

            cache_path = Path(path).with_suffix(f".voxels_bins{nb_time_bins}.npy")

            if not cache_path.exists():
                # Ancienne cache .pt présente → migration transparente
                old_cache = Path(path).with_suffix(f".voxels_bins{nb_time_bins}.pt")
                if old_cache.exists():
                    print(f"[EvimoSegDataset]   migration .pt → .npy ...", flush=True)
                    voxel_seq = torch.load(old_cache)
                    np.save(cache_path, voxel_seq.numpy())
                else:
                    ea = load_events(path)
                    voxel_seq = make_voxel_sequence(ea, fm.ts, nb_of_time_bins=nb_time_bins)
                    np.save(cache_path, voxel_seq.numpy())

            # Memmap : l'OS ne charge en RAM que les pages réellement accédées
            voxel_mmap = np.load(cache_path, mmap_mode='r')
            n = voxel_mmap.shape[0]

            mask_seq = torch.stack([downsample_mask(m, patch_size) for m in fm.masks[:-1]])

            seq_idx = len(self.sequences)
            for start in range(0, n - seq_len + 1, seq_len):
                self.index.append((seq_idx, start))
            self.sequences.append((cache_path, mask_seq))

            print(f"[EvimoSegDataset]   -> {n} frames, {time.time() - t0:.1f}s", flush=True)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.index[idx]
        cache_path, mask_seq = self.sequences[seq_idx]
        end = start + self.seq_len
        # Rechargement du memmap nécessaire dans les workers DataLoader (après fork)
        voxel_mmap = np.load(cache_path, mmap_mode='r')
        voxel = torch.from_numpy(np.array(voxel_mmap[start:end], dtype=np.float32))
        return voxel, mask_seq[start:end]
