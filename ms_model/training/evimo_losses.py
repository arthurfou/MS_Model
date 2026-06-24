"""Loss functions for the EV-IMO SfM pipeline.

Three components (EV-IMO paper Section III-E):
  WarpLoss  — self-supervised event sharpness (coarse warping loss)
  DepthLoss — supervised depth from sparse GT
  MaskLoss  — semi-supervised motion mask BCE + spatial smoothness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WarpLoss(nn.Module):
    """Coarse event warping / sharpness loss.

    For K=1,2 neighbouring slices around the centre, inversely warp them to
    the centre frame using the predicted flow and measure absolute pixel error.

    Loss_coarse = mean over k in {-2,-1,+1,+2} of:
      mean |warp(slice_{c+k}, ±flow) - slice_c|
    """

    def __init__(self, use_fine_loss: bool = False, p_norm: float = 0.5):
        super().__init__()
        self.use_fine_loss = use_fine_loss  # fine-scale 1 ms loss — not yet implemented
        self.p_norm = p_norm

    @staticmethod
    def _inverse_warp(
        src: torch.Tensor, flow: torch.Tensor, sign: float
    ) -> torch.Tensor:
        """Sample src at current_coords + sign * flow.

        src:  (B, 3, H, W)
        flow: (B, 2, H, W) in pixel units
        Returns warped (B, 3, H, W).
        """
        B, _, H, W = src.shape
        device = src.device

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        u_src = xs.unsqueeze(0) + sign * flow[:, 0]   # (B, H, W)
        v_src = ys.unsqueeze(0) + sign * flow[:, 1]

        grid_x = 2.0 * u_src / (W - 1) - 1.0
        grid_y = 2.0 * v_src / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)   # (B, H, W, 2)

        return F.grid_sample(
            src.float(), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )

    def forward(self, slices: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        slices: (B, n_ctx, 3, H, W)
        flow:   (B, 2, H, W)
        """
        B, n_ctx, C, H, W = slices.shape
        centre = n_ctx // 2
        s_c = slices[:, centre].float()

        loss = slices.new_zeros(1).squeeze()
        count = 0

        for k in (-2, -1, 1, 2):
            idx = centre + k
            if idx < 0 or idx >= n_ctx:
                continue
            # sign: flow is from centre → centre+k, so warp k back to centre with sign k/|k|
            sign = float(k) / abs(k)
            warped = self._inverse_warp(slices[:, idx], flow, sign=sign)
            loss = loss + (warped - s_c).abs().mean()
            count += 1

        return loss / max(count, 1)


class DepthLoss(nn.Module):
    """Supervised sparse depth loss (EV-IMO paper Eq. III-E.3).

    Loss_depth = mean(max(d_gt/d_pred, d_pred/d_gt) + |d_pred - d_gt| / d_gt)
               + lambda_smooth * ||Δ² d_pred||_1

    Only computed at non-NaN GT pixels.
    """

    def __init__(self, lambda_smooth: float = 0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(
        self,
        depth_pred: torch.Tensor,   # (B, 1, H, W)
        depth_gt: torch.Tensor,     # (B, H, W), NaN = background
    ) -> torch.Tensor:
        pred = depth_pred.squeeze(1)   # (B, H, W)
        valid = ~torch.isnan(depth_gt)

        if valid.any():
            p = pred[valid].clamp(min=1.0)
            t = depth_gt[valid].clamp(min=1.0)
            ratio = torch.max(t / p, p / t)
            abs_rel = (p - t).abs() / t
            main_loss = (ratio + abs_rel).mean()
        else:
            main_loss = pred.mean() * 0.0   # differentiable zero

        # L1 of discrete Laplacian of depth_pred (second-order smoothness)
        d = pred
        lap = (
            d[:, 2:, 1:-1] + d[:, :-2, 1:-1] - 2 * d[:, 1:-1, 1:-1]
            + d[:, 1:-1, 2:] + d[:, 1:-1, :-2] - 2 * d[:, 1:-1, 1:-1]
        )
        smooth_loss = lap.abs().mean()

        return main_loss + self.lambda_smooth * smooth_loss


class MaskLoss(nn.Module):
    """Motion mask semi-supervised loss (EV-IMO paper Section III-E.2).

    Two terms:
      BCE  — background pixels (mask_gt == 0) should have mixture weight 0 ≈ 1
      Smooth — L1 spatial gradient of all mixture weights
    """

    def __init__(self, lambda_smooth: float = 0.1, lambda_bce: float = 1.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_bce = lambda_bce

    def forward(
        self,
        masks_pred: torch.Tensor,   # (B, C+1, H, W) softmax probabilities
        mask_gt: torch.Tensor,      # (B, H, W) int64, 0 = background
    ) -> torch.Tensor:
        bg_pixel = mask_gt == 0                              # (B, H, W)
        bg_proba = masks_pred[:, 0]                          # (B, H, W)

        if bg_pixel.any():
            bce = -torch.log(bg_proba[bg_pixel].clamp(min=1e-7)).mean()
        else:
            bce = masks_pred.sum() * 0.0

        dy = (masks_pred[:, :, 1:, :] - masks_pred[:, :, :-1, :]).abs().mean()
        dx = (masks_pred[:, :, :, 1:] - masks_pred[:, :, :, :-1]).abs().mean()

        return self.lambda_bce * bce + self.lambda_smooth * (dy + dx)


class EvimoLoss(nn.Module):
    """Combined EV-IMO loss.

    Loss_total = Loss_warp + w_depth * Loss_depth + w_mask * Loss_mask
    """

    def __init__(
        self,
        w_depth: float = 1.0,
        w_mask: float = 0.5,
        warp_loss_kwargs: dict | None = None,
        depth_loss_kwargs: dict | None = None,
        mask_loss_kwargs: dict | None = None,
    ):
        super().__init__()
        self.warp_loss = WarpLoss(**(warp_loss_kwargs or {}))
        self.depth_loss = DepthLoss(**(depth_loss_kwargs or {}))
        self.mask_loss = MaskLoss(**(mask_loss_kwargs or {}))
        self.w_depth = w_depth
        self.w_mask = w_mask

    def forward(
        self, output: dict, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns (total_loss, components_dict) where components keys are
        'warp', 'depth', 'mask' — useful for WandB logging.
        """
        l_warp = self.warp_loss(batch["slices"], output["flow"])
        l_depth = self.depth_loss(output["depth"], batch["depth_gt"])
        l_mask = self.mask_loss(output["masks"], batch["mask_gt"])

        total = l_warp + self.w_depth * l_depth + self.w_mask * l_mask
        return total, {
            "warp": l_warp.detach(),
            "depth": l_depth.detach(),
            "mask": l_mask.detach(),
        }
