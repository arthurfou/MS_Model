import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss binaire sur logits.

    Optimise directement le ratio intersection/union — robuste au déséquilibre
    de classes sans avoir besoin de pos_weight.
    Yaml : loss: dice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum() + self.smooth
        den = probs.sum() + targets.sum() + self.smooth
        return 1 - num / den


class FocalLoss(nn.Module):
    """Focal Loss — BCE pondérée par la difficulté de chaque exemple.

    Réduit la contribution des exemples faciles (fond statique bien classifié)
    et concentre l'apprentissage sur les patches difficiles (objets mobiles rares).
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Yaml :
      loss: focal
      loss_focal_gamma: 2.0   # plus grand = plus de focus sur les cas difficiles
      loss_pos_weight: 10.0   # optionnel, comme pour bce
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.gamma = gamma
        w = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


class BCEDiceLoss(nn.Module):
    """Combinaison BCE + Dice Loss.

    BCE apporte la stabilité du gradient pixel par pixel ; Dice force le modèle
    à bien segmenter les régions dynamiques (surtout utile en début d'entraînement).

    Yaml :
      loss: bce_dice
      loss_bce_weight: 0.5    # part de BCE (1 - bce_weight = part Dice)
      loss_pos_weight: 10.0   # optionnel, appliqué à la composante BCE
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        w = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=w)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (1 - self.bce_weight) * self.dice(logits, targets)


def build_loss(
    name: str = "bce",
    pos_weight: float | None = None,
    focal_gamma: float = 2.0,
    bce_weight: float = 0.5,
) -> nn.Module:
    """Construit la loss d'entraînement.

    Paramètres yaml (section training) — tous optionnels sauf loss :
      loss: bce          # bce | dice | focal | bce_dice
      loss_pos_weight: 10.0      # bce, focal, bce_dice — pondère les patches dynamiques
      loss_focal_gamma: 2.0      # focal uniquement
      loss_bce_weight: 0.5       # bce_dice uniquement (part de BCE vs Dice)
    """
    if name == "bce":
        w = torch.tensor([pos_weight]) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=w)
    if name == "dice":
        return DiceLoss()
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    if name == "bce_dice":
        return BCEDiceLoss(bce_weight=bce_weight, pos_weight=pos_weight)
    raise ValueError(f"Loss inconnue: '{name}'. Disponibles: bce, dice, focal, bce_dice")
