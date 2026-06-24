from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ms_model.io.loaders import load_events, load_evimo_mask
from ms_model.models import build_model
from ms_model.representations.voxel import make_voxel_sequence
from ms_model.viz.render import render_event_frame


class Predictor:
    """Charge un modèle entraîné et prédit les masques sur une séquence npz.

    Paramètres
    ----------
    weights_path : chemin vers best.pt ou last.pt
    config_path  : yaml d'entraînement (pour lire l'archi et les params data)
    device       : "cuda", "cpu", ou None (auto-detect)
    """

    def __init__(self, weights_path: str | Path, config_path: str | Path, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        cfg = yaml.safe_load(Path(config_path).read_text())
        data_cfg = cfg["data"]
        model_cfg = dict(cfg["model"])
        model_name = model_cfg.pop("name")

        self.patch_size = data_cfg["patch_size"]
        self.nb_time_bins = data_cfg["nb_time_bins"]
        self.seq_len = data_cfg["seq_len"]

        self.model = build_model(
            model_name,
            in_channels=self.nb_time_bins,
            patch_size=self.patch_size,
            **model_cfg,
        ).to(self.device)

        ckpt = torch.load(weights_path, map_location=self.device)
        # best.pt = state_dict direct ; last.pt = dict avec clé "model"
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

    def predict_sequence(self, npz_path: str | Path) -> dict:
        """Prédit les masques pour toute une séquence npz.

        Retourne un dict avec :
        - "event_frames" : list[np.ndarray (H, W, 3) uint8]
        - "pred_masks"   : list[np.ndarray (H, W) bool]
        - "gt_masks"     : list[np.ndarray (H, W) bool]  (masque GT binaire)
        - "n_frames"     : int
        """
        npz_path = Path(npz_path)
        ea = load_events(npz_path)
        fm = load_evimo_mask(npz_path)

        voxel_seq = make_voxel_sequence(ea, fm.ts, nb_of_time_bins=self.nb_time_bins)
        N, _, H, W = voxel_seq.shape

        all_logits = []
        with torch.no_grad():
            for start in range(0, N, self.seq_len):
                end = min(start + self.seq_len, N)
                chunk = voxel_seq[start:end].unsqueeze(0).to(self.device)  # (1, T, C, H, W)
                logits = self.model(chunk)                                   # (1, T, 1, Hp, Wp)
                all_logits.append(logits[0, :, 0].cpu())                   # (T, Hp, Wp)

        logits_seq = torch.cat(all_logits, dim=0)          # (N, Hp, Wp)
        pred_prob = torch.sigmoid(logits_seq).unsqueeze(1) # (N, 1, Hp, Wp)
        pred_full = F.interpolate(pred_prob, size=(H, W), mode="nearest")  # (N, 1, H, W)
        pred_masks = (pred_full[:, 0] > 0.5).numpy()       # (N, H, W) bool

        event_frames = [render_event_frame(ea, fm.ts[i - 1], fm.ts[i]) for i in range(1, len(fm.ts))]
        gt_masks = [(m != 0) for m in fm.masks[:-1]]

        return {
            "event_frames": event_frames,
            "pred_masks":   [pred_masks[i] for i in range(N)],
            "gt_masks":     gt_masks,
            "n_frames":     N,
        }
