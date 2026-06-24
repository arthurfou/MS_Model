import torch
import torch.nn as nn
import torch.nn.functional as F

from .ecn import ECNBackbone


class PoseNetwork(nn.Module):
    """Predicts ego-motion pose, per-object 3-D translations, and pixel-wise mixture masks.

    Input:  (B, n_context_slices * 3, H, W)   — concatenated event slices
    Output:
      poses: (B, 6 + 3*C)
             [vx, vy, vz, wx, wy, wz,  t1x, t1y, t1z, …, tCx, tCy, tCz]
      masks: (B, C+1, H, W) softmax mixture weights (channel 0 = background)

    Architecture:
      - ECNBackbone shared encoder/decoder
      - Spatial decoder head → mask logits (C+1 channels) via softmax
      - Global avg-pool of bottleneck → small MLP → pose vector
    """

    def __init__(
        self,
        n_context_slices: int = 5,
        max_objects: int = 3,
        ecn_init_channels: int = 32,
        ecn_growth: int = 32,
        ecn_n_stages: int = 4,
    ):
        super().__init__()
        self.max_objects = max_objects

        self.ecn = ECNBackbone(
            in_channels=n_context_slices * 3,
            out_channels=max_objects + 1,
            init_channels=ecn_init_channels,
            growth=ecn_growth,
            n_stages=ecn_n_stages,
        )

        btl = self.ecn.bottleneck_channels
        self.pose_head = nn.Sequential(
            nn.Linear(btl, btl // 2),
            nn.ReLU(inplace=True),
            nn.Linear(btl // 2, 6 + 3 * max_objects),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, n_ctx*3, H, W)
        Returns: poses (B, 6+3C), masks (B, C+1, H, W)
        """
        bottleneck, enc_feats = self.ecn.encode(x)

        mask_logits = self.ecn.decode(bottleneck, enc_feats)      # (B, C+1, H, W)
        masks = F.softmax(mask_logits, dim=1)

        pooled = bottleneck.mean(dim=[2, 3])                       # (B, btl)
        poses = self.pose_head(pooled)                             # (B, 6+3C)

        return poses, masks
