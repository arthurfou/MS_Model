from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EventArray:
    """Représentation pivot des events événementiels.

    Tous les loaders de `ms_model.io.loaders` convertissent leur format
    d'entrée vers cette structure, qui sert ensuite de point de départ
    commun pour les représentations (voxel grid, etc.) et la visualisation.

    Attributes:
        events: (N, 4) float64 — colonnes [t, x, y, p].
            t : timestamp en secondes, croissant.
            x, y : coordonnées pixel.
            p : polarité dans {0, 1}.
        H, W: résolution du capteur (hauteur, largeur).
        frame_ts: (M,) float64 ou None — timestamps de frames en secondes,
            utilisés pour découper le flux en tranches (ex: pour matcher
            les images GT d'EVIMO).
    """

    events: np.ndarray
    H: int
    W: int
    frame_ts: Optional[np.ndarray] = None

    @property
    def t(self) -> np.ndarray:
        return self.events[:, 0]

    @property
    def x(self) -> np.ndarray:
        return self.events[:, 1]

    @property
    def y(self) -> np.ndarray:
        return self.events[:, 2]

    @property
    def p(self) -> np.ndarray:
        return self.events[:, 3]

    def __len__(self) -> int:
        return self.events.shape[0]
