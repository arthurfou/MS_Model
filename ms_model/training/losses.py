import torch.nn as nn


def build_loss(name: str = "bce", **kwargs) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    raise ValueError(f"Loss inconnue: '{name}'")
