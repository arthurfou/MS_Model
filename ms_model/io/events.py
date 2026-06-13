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

    def info(self) -> dict:
        """Statistiques descriptives d'un EventArray.

        Retourne un dict avec par ex. :
        - n_events
        - resolution (H, W)
        - duration_s (t.max() - t.min())
        - event_rate (n_events / duration_s)
        - polarity_ratio (proportion d'events positifs)
        - n_frames (si frame_ts dispo)
        - mean_fps (si frame_ts dispo)
        """
        duration = self.t.max() - self.t.min()
        return {
            'n_events': len(self.events),
            'resolution': (self.H, self.W),
            'duration': duration,
            'event_rate': len(self.events) / duration,
            'polarity_ratio': np.mean(self.p > 0),
            'n_frames': len(self.frame_ts) if self.frame_ts is not None else None,
            'mean_fps': (len(self.frame_ts) - 1) / duration if self.frame_ts is not None else None,
        }