import math

import torch
import torch.nn as nn

from .base import register_model


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch_size: int, height: int, width: int, device):
        shape = (batch_size, self.hidden_dim, height, width)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


@register_model("convlstm")
class ConvLSTMSeg(nn.Module):
    """Segmentation dynamique/statique par patch à partir d'une séquence de voxel grids.

    Entrée : (B, T, in_channels, H, W)
    Sortie : (B, T, 1, H/patch_size, W/patch_size) — logits (avant sigmoid).
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 32, patch_size: int = 4):
        super().__init__()
        n_down = int(math.log2(patch_size))
        if 2 ** n_down != patch_size:
            raise ValueError(f"patch_size doit être une puissance de 2, reçu {patch_size}")

        layers = []
        c = in_channels
        for _ in range(n_down):
            layers += [nn.Conv2d(c, hidden_dim, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            c = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.patch_size = patch_size

        self.cell = ConvLSTMCell(hidden_dim, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        h, c = self.cell.init_state(B, Hp, Wp, x.device)
        outputs = []
        for t in range(T):
            feat = self.encoder(x[:, t])
            if feat.shape[-2:] != (Hp, Wp):
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (Hp, Wp))
            h, c = self.cell(feat, (h, c))
            outputs.append(self.head(h))

        return torch.stack(outputs, dim=1)
