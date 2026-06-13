from dataclasses import dataclass

import numpy as np


@dataclass
class FrameMasks:
    """Séquence de masques de segmentation par frame, alignés sur leurs timestamps.

    Représentation générique : peut venir d'une vérité terrain (EVIMO `mask`)
    ou de la sortie d'un modèle de motion segmentation, peu importe le dataset.

    Attributes:
        masks: (M, H, W) — masques de segmentation. 0 = fond/statique,
            toute autre valeur = id d'objet mobile.
        ts: (M,) float64 — timestamp de chaque masque, en secondes, dans la
            même base temporelle que `EventArray.t`.
    """

    masks: np.ndarray
    ts: np.ndarray

    def __len__(self) -> int:
        return self.masks.shape[0]
