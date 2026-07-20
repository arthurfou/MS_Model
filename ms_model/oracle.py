"""Fournisseurs de masque « dynamique » à la résolution de la score map DEVO.

Deux implémentations partageant la même interface (callable `ts_us -> (1,1,Hp,Wp)`),
branchables sur le hook additif `run_voxel(..., dyn_mask_provider=...)` du fork DEVO
(voir ../../PLAN.md) :

- `OracleDynMaskProvider` — masque de vérité terrain (GT EVIMO). Sert le jalon 1 / **M0**
  (valider la plomberie d'injection avec un oracle).
- `LearnedDynMaskProvider` — masque prédit par un modèle MS entraîné (convlstm, …), branché
  en **préprocesseur découplé**. Sert le jalon 1 / **M1** (baseline « art antérieur » :
  suppression apprise mais entraînée séparément de la VO).

Convention de score (identique à `ms_model.data.evimo_dataset.downsample_mask`) :
    valeur dans [0, 1] par cellule de score map ; 1 = objet mobile. DEVO annule la score
    map par (1 - score), donc 1 => patch jamais sélectionné.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.io.masks import FrameMasks
from ms_model.masking import thicken_mask
from ms_model.data.evimo_dataset import downsample_mask


class TimestampMaskProvider:
    """Base commune : sert un score (1, 1, Hp, Wp) par timestamp (plus proche voisin).

    Args:
        ts_seconds: (M,) timestamps des frames de score, en secondes.
        scores: (M, Hp, Wp) scores dynamiques dans [0, 1] alignés sur `ts_seconds`.
        device: device des tenseurs retournés.
        max_dt_ms: garde-fou — alerte (une fois) si le plus proche timestamp est plus
            loin que ce seuil (détecte une mauvaise base temporelle).
        ts_offset_s: offset à soustraire de ts_us/1e6 avant lookup. Nécessaire quand
            ts_us vient de DEVO (Unix epoch µs) et ts_seconds est relatif (EVIMO npz).
            Calcul: tss_imgs_us[0]/1e6 − meta_ts[0].
    """

    def __init__(
        self,
        ts_seconds: np.ndarray,
        scores: torch.Tensor,
        device: Union[str, torch.device] = "cuda",
        max_dt_ms: float = 50.0,
        ts_offset_s: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.max_dt_ms = max_dt_ms
        self._ts_offset_s = float(ts_offset_s)

        ts = np.asarray(ts_seconds, dtype=np.float64)
        order = np.argsort(ts)
        self.ts = ts[order]
        self.scores = scores[order].to(self.device).float().clamp_(0.0, 1.0)
        assert self.scores.dim() == 3, f"scores doit être (M, Hp, Wp), reçu {tuple(scores.shape)}"
        assert len(self.ts) == self.scores.shape[0], "ts et scores désalignés"
        self._warned = False

    def _nearest_index(self, t_s: float) -> int:
        i = int(np.searchsorted(self.ts, t_s))
        if i <= 0:
            return 0
        if i >= len(self.ts):
            return len(self.ts) - 1
        return i if (self.ts[i] - t_s) < (t_s - self.ts[i - 1]) else i - 1

    def __call__(self, ts_us: Union[int, float, torch.Tensor]) -> torch.Tensor:
        """Retourne le score dynamique (1, 1, Hp, Wp) pour la frame la plus proche de ts_us."""
        if isinstance(ts_us, torch.Tensor):
            ts_us = ts_us.item()
        t_s = float(ts_us) / 1e6 - self._ts_offset_s
        idx = self._nearest_index(t_s)

        dt_ms = abs(self.ts[idx] - t_s) * 1e3
        if dt_ms > self.max_dt_ms and not self._warned:
            print(
                f"[{type(self).__name__}] ATTENTION: écart timestamp {dt_ms:.1f} ms "
                f"> {self.max_dt_ms} ms (t={t_s:.4f}s, proche={self.ts[idx]:.4f}s). "
                f"Vérifie que ts_us (voxels DEVO) et ts partagent la même base temporelle."
            )
            self._warned = True

        return self.scores[idx][None, None]  # (1, 1, Hp, Wp)


class OracleDynMaskProvider(TimestampMaskProvider):
    """Masque GT dynamique (M0). Binaire : cellule = 1 si le bloc patch×patch contient
    au moins un pixel dynamique (max-pool)."""

    def __init__(
        self,
        frame_masks: FrameMasks,
        patch_size: int = 4,
        device: Union[str, torch.device] = "cuda",
        max_dt_ms: float = 50.0,
        ts_offset_s: float = 0.0,
    ) -> None:
        scores = torch.stack(
            [downsample_mask(m, patch_size) for m in frame_masks.masks]
        )  # (M, Hp, Wp), {0,1}
        super().__init__(frame_masks.ts, scores, device=device, max_dt_ms=max_dt_ms,
                         ts_offset_s=ts_offset_s)

    @classmethod
    def from_evimo_npz(
        cls,
        npz_path: Union[str, Path],
        patch_size: int = 4,
        thicken_radius: int = 0,
        device: Union[str, torch.device] = "cuda",
        max_dt_ms: float = 50.0,
        ts_offset_s: float = 0.0,
    ) -> "OracleDynMaskProvider":
        """Construit le provider depuis un npz EVIMO (clés `mask` + `meta`).

        thicken_radius > 0 dilate les masques (marge autour des objets mobiles) avant
        sous-échantillonnage — un patch en bordure d'objet reste pollué.
        ts_offset_s: offset à soustraire lors du lookup (voir TimestampMaskProvider).
        """
        fm = load_evimo_mask(npz_path)
        if thicken_radius > 0:
            fm = thicken_mask(fm, radius=thicken_radius)
        return cls(fm, patch_size=patch_size, device=device, max_dt_ms=max_dt_ms,
                   ts_offset_s=ts_offset_s)


class LearnedDynMaskProvider(TimestampMaskProvider):
    """Masque prédit par un modèle MS entraîné, branché en préprocesseur découplé (M1).

    Le modèle tourne sur SA propre voxelisation des events de la scène (indépendamment de
    DEVO), produit une probabilité dynamique par frame à la résolution patch (Hp, Wp), puis
    la sert à DEVO par timestamp — exactement comme l'oracle, mais appris.

    threshold: None => score continu (probabilité, injection douce, DEVO-natif) ;
        float => masque binaire (proba > seuil), variante « masque dur » type suppression RPG.
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        weights_path: Union[str, Path],
        config_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        threshold: Optional[float] = None,
        max_dt_ms: float = 50.0,
        ts_offset_s: float = 0.0,
    ) -> None:
        # import local pour ne pas alourdir l'import du module (charge torch/modèles)
        from ms_model.inference import load_model_from_checkpoint
        from ms_model.representations.voxel import make_voxel_sequence

        device = torch.device(device)
        model, data_cfg = load_model_from_checkpoint(weights_path, config_path, device)
        nb_time_bins = data_cfg["nb_time_bins"]
        seq_len = data_cfg["seq_len"]

        ea = load_events(npz_path)
        # ts = bornes de frames (meta EVIMO) — dispo sans GT de segmentation ; à la voxelisation
        # DEVO ce seraient les temps de frame du flux d'events.
        frame_ts = load_evimo_mask(npz_path).ts
        voxel_seq = make_voxel_sequence(ea, frame_ts, nb_of_time_bins=nb_time_bins)  # (N, C, H, W)
        N = voxel_seq.shape[0]

        probs = []
        with torch.no_grad():
            for start in range(0, N, seq_len):
                chunk = voxel_seq[start:start + seq_len].unsqueeze(0).to(device)  # (1, T, C, H, W)
                logits = model(chunk)                       # (1, T, 1, Hp, Wp)
                probs.append(torch.sigmoid(logits)[0, :, 0].cpu())  # (T, Hp, Wp)
        scores = torch.cat(probs, dim=0)                    # (N, Hp, Wp) in [0,1]

        if threshold is not None:
            scores = (scores > threshold).float()

        # la i-ème prédiction couvre [ts_i, ts_{i+1}) -> on l'étiquette par ts_i
        super().__init__(frame_ts[:N], scores, device=device, max_dt_ms=max_dt_ms,
                         ts_offset_s=ts_offset_s)
