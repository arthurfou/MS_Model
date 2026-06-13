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

    Chaque npz EVIMO est découpé selon les timestamps GT du masque
    (`load_evimo_mask`), puis en chunks non-recouvrants de longueur `seq_len`.
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
        self.sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.index: list[tuple[int, int]] = []

        for path in npz_paths:
            ea = load_events(path)
            fm = load_evimo_mask(path)
            if mask_thicken_radius > 0:
                fm = thicken_mask(fm, radius=mask_thicken_radius)

            voxel_seq = make_voxel_sequence(ea, fm.ts, nb_of_time_bins=nb_time_bins)
            mask_seq = torch.stack([downsample_mask(m, patch_size) for m in fm.masks[:-1]])

            n = voxel_seq.shape[0]
            seq_idx = len(self.sequences)
            for start in range(0, n - seq_len + 1, seq_len):
                self.index.append((seq_idx, start))
            self.sequences.append((voxel_seq, mask_seq))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.index[idx]
        voxel_seq, mask_seq = self.sequences[seq_idx]
        end = start + self.seq_len
        return voxel_seq[start:end], mask_seq[start:end]
