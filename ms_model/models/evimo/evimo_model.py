"""Full EV-IMO pipeline: depth + pose/mask + optical flow from event slices.

Reference: Mitrokhin et al., "EV-IMO: Motion Segmentation Dataset and
Learning Pipeline for Event Cameras", IROS 2019 (arxiv 1903.07520).
"""
import torch
import torch.nn as nn

from ms_model.models.base import register_model
from .depth_net import DepthNetwork
from .pose_net import PoseNetwork


_ECN_VARIANTS: dict[str, dict] = {
    "baseline": {"init_channels": 32, "growth": 32, "n_stages": 4},
    "shallow":  {"init_channels": 8,  "growth": 8,  "n_stages": 2},
}


@register_model("evimo")
class EvimoModel(nn.Module):
    """EV-IMO SfM pipeline.

    build_model("evimo", **cfg) kwargs:
      max_objects          int = 3
      H                    int = 260
      W                    int = 346
      n_context_slices     int = 5
      ecn_variant          str = "baseline"  # or "shallow"

    forward(batch) expects a dict with keys:
      "slices"    (B, n_ctx, 3, H, W)
      "K"         (B, 3, 3)

    Returns a dict with keys:
      "depth"  (B, 1, H, W)      — positive depth in mm
      "poses"  (B, 6+3*C)        — [ego_6dof | obj1_3d | … | objC_3d]
      "masks"  (B, C+1, H, W)    — softmax mixture weights
      "flow"   (B, 2, H, W)      — pixel-space optical flow
    """

    def __init__(
        self,
        max_objects: int = 3,
        H: int = 260,
        W: int = 346,
        n_context_slices: int = 5,
        ecn_variant: str = "baseline",
        **ignored_kwargs,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.H, self.W = H, W
        self.n_ctx = n_context_slices
        self.half = n_context_slices // 2

        ecn = _ECN_VARIANTS[ecn_variant]

        self.depth_net = DepthNetwork(
            in_channels=3,
            ecn_init_channels=ecn["init_channels"],
            ecn_growth=ecn["growth"],
            ecn_n_stages=ecn["n_stages"],
        )
        self.pose_net = PoseNetwork(
            n_context_slices=n_context_slices,
            max_objects=max_objects,
            ecn_init_channels=ecn["init_channels"],
            ecn_growth=ecn["growth"],
            ecn_n_stages=ecn["n_stages"],
        )

    # ------------------------------------------------------------------

    def compute_flow(
        self,
        poses: torch.Tensor,   # (B, 6+3C)
        masks: torch.Tensor,   # (B, C+1, H, W)
        K: torch.Tensor,       # (B, 3, 3)
    ) -> torch.Tensor:
        """Compute pixel-space optical flow from pose and mixture masks.

        Uses the camera projection model (EV-IMO paper Eq. 1):
          (u, v) = ½[[-1,0,x],[0,-1,y]] * [vx,vy,vz]ᵀ
                 + [[xy,-(1+x²),y],[(1+y²),-xy,-x]] * [wx,wy,wz]ᵀ
        where (x, y) are normalised image coordinates.

        Per-pixel pose (Eq. 2): p_pixel = p_ego + Σ_j m_ij * [t_j; 0_3]
        """
        B = poses.shape[0]
        device = poses.device
        H, W = self.H, self.W

        p_ego = poses[:, :6]                                         # (B, 6)
        obj_t = poses[:, 6:].reshape(B, self.max_objects, 3)         # (B, C, 3)

        # Pad object translations to 6-DOF (rotation = 0)
        obj_t6 = torch.cat(
            [obj_t, torch.zeros(B, self.max_objects, 3, device=device)], dim=-1
        )                                                             # (B, C, 6)

        # Mixture: weighted sum of object 6-DOF poses
        weighted_obj = torch.einsum("bchw,bcx->bxhw", masks[:, 1:].float(), obj_t6)
        p_pixel = p_ego.view(B, 6, 1, 1) + weighted_obj              # (B, 6, H, W)

        # Normalised image coordinates
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        xn = (xs.unsqueeze(0) - cx) / fx    # (B, H, W)
        yn = (ys.unsqueeze(0) - cy) / fy

        vx = p_pixel[:, 0]; vy = p_pixel[:, 1]; vz = p_pixel[:, 2]
        wx = p_pixel[:, 3]; wy = p_pixel[:, 4]; wz = p_pixel[:, 5]

        # Paper Eq. 1 (normalised flow)
        u_n = 0.5 * (-vx + xn * vz) + (xn * yn * wx - (1 + xn**2) * wy + yn * wz)
        v_n = 0.5 * (-vy + yn * vz) + ((1 + yn**2) * wx - xn * yn * wy - xn * wz)

        # Convert to pixel flow
        return torch.stack([fx * u_n, fy * v_n], dim=1)              # (B, 2, H, W)

    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        slices = batch["slices"]    # (B, n_ctx, 3, H, W)
        K = batch["K"]             # (B, 3, 3)
        B = slices.shape[0]

        depth = self.depth_net(slices[:, self.half])                  # (B, 1, H, W)

        pose_input = slices.reshape(B, self.n_ctx * 3, self.H, self.W)
        poses, masks = self.pose_net(pose_input)                      # (B, 6+3C), (B, C+1, H, W)

        flow = self.compute_flow(poses, masks, K)                     # (B, 2, H, W)

        return {"depth": depth, "poses": poses, "masks": masks, "flow": flow}
