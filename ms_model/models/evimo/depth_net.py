import torch
import torch.nn as nn
import torch.nn.functional as F

from .ecn import ECNBackbone


class DepthNetwork(nn.Module):
    """Predicts a dense positive depth map from a single 3-channel event slice.

    Input:  (B, 3, H, W)
    Output: (B, 1, H, W) — positive values via softplus activation
    """

    def __init__(
        self,
        in_channels: int = 3,
        ecn_init_channels: int = 32,
        ecn_growth: int = 32,
        ecn_n_stages: int = 4,
    ):
        super().__init__()
        self.ecn = ECNBackbone(
            in_channels=in_channels,
            out_channels=1,
            init_channels=ecn_init_channels,
            growth=ecn_growth,
            n_stages=ecn_n_stages,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → depth: (B, 1, H, W), values > 0"""
        return F.softplus(self.ecn(x))
