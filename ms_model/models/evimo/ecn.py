"""Evenly Cascading Network (ECN) encoder-decoder backbone.

Based on the architecture described in EV-IMO (arxiv 1903.07520, ref [41]).
U-Net-style encoder-decoder with additive channel growth and GroupNorm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _n_groups(channels: int) -> int:
    """Largest divisor of `channels` that is ≤ 8, for GroupNorm."""
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class _EncoderStage(nn.Module):
    """Stride-2 spatial downsampling + feature refinement."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_n_groups(c_out), c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(_n_groups(c_out), c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DecoderStage(nn.Module):
    """×2 bilinear upsample + skip concat + refinement."""

    def __init__(self, c_skip: int, c_prev: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv2d(c_skip + c_prev, c_out, 1, bias=False)
        self.refine = nn.Sequential(
            nn.GroupNorm(_n_groups(c_out), c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(_n_groups(c_out), c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(prev, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.proj(torch.cat([skip, up], dim=1)))


class ECNBackbone(nn.Module):
    """Evenly Cascading Network (ECN) encoder-decoder.

    Channel schedule (additive growth):
      enc_ch = [init_channels, init+growth, init+2*growth, ..., init+n_stages*growth]

    Preset variants (from EV-IMO paper):
      baseline: init_channels=32, growth=32, n_stages=4  (~2M params)
      shallow : init_channels=8,  growth=8,  n_stages=2  (~40k params)

    Args:
        in_channels:   input feature channels
        out_channels:  output channels of the final 1×1 head
        init_channels: channels after input projection
        growth:        additive channel growth per encoder stage
        n_stages:      number of encoder (= decoder) stages
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_channels: int = 32,
        growth: int = 32,
        n_stages: int = 4,
    ):
        super().__init__()
        self.n_stages = n_stages

        enc_ch = [init_channels + i * growth for i in range(n_stages + 1)]

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, enc_ch[0], 3, padding=1, bias=False),
            nn.GroupNorm(_n_groups(enc_ch[0]), enc_ch[0]),
            nn.ReLU(inplace=True),
        )

        self.encoders = nn.ModuleList(
            [_EncoderStage(enc_ch[i], enc_ch[i + 1]) for i in range(n_stages)]
        )

        # dec stage i: skip from enc_ch[n_stages-1-i], prev from dec output above
        dec_ch = list(reversed(enc_ch))  # [deepest, …, shallowest]
        self.decoders = nn.ModuleList([
            _DecoderStage(dec_ch[i + 1], dec_ch[i], dec_ch[i + 1])
            for i in range(n_stages)
        ])

        self.output_head = nn.Conv2d(enc_ch[0], out_channels, 1)
        self.bottleneck_channels = enc_ch[-1]

    # ------------------------------------------------------------------
    # Two-phase API for networks that need access to the bottleneck.
    # ------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode input and return (bottleneck, encoder_features).

        encoder_features[0]  = output of input_proj  (full resolution)
        encoder_features[-1] = bottleneck             (deepest level)
        """
        x = self.input_proj(x)
        feats = [x]
        for enc in self.encoders:
            x = enc(x)
            feats.append(x)
        return x, feats

    def decode(
        self, bottleneck: torch.Tensor, enc_feats: list[torch.Tensor]
    ) -> torch.Tensor:
        """Decode bottleneck back to full resolution and apply output head."""
        x = bottleneck
        for i, dec in enumerate(self.decoders):
            skip = enc_feats[-(i + 2)]
            x = dec(skip, x)
        return self.output_head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, enc_feats = self.encode(x)
        return self.decode(bottleneck, enc_feats)
