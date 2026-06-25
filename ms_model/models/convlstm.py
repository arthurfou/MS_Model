import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                feat = F.adaptive_avg_pool2d(feat, (Hp, Wp))
            h, c = self.cell(feat, (h, c))
            outputs.append(self.head(h))

        return torch.stack(outputs, dim=1)


def _encoder_stage(c_in: int, c_out: int, num_groups: int) -> nn.Sequential:
    """Un étage de downsampling x2 : conv stride-2 + conv stride-1 + GroupNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in,  c_out, 3, stride=2, padding=1, bias=False),
        nn.GroupNorm(num_groups, c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
        nn.GroupNorm(num_groups, c_out),
        nn.ReLU(inplace=True),
    )


class _DecoderBlock(nn.Module):
    """Upsample ×2 + concat skip + conv de raffinement."""

    def __init__(self, c_skip: int, c_up: int, c_out: int, num_groups: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(c_skip + c_up, c_out, 1, bias=False),
            nn.GroupNorm(num_groups, c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x_up, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x_skip, x_up], dim=1))


@register_model("convlstm_v2")
class ConvLSTMSeg_v2(nn.Module):
    """ConvLSTM v2 — encodeur enrichi + tête de classification à 2 couches.

    Améliorations par rapport à v1 :
    - 2 convolutions par étage de downsampling (vs 1) pour plus de capacité d'extraction
    - GroupNorm après chaque conv (stable sur petits batchs, remplace BatchNorm)
    - Tête de classification à 2 couches 1×1 (vs 1)

    Entrée : (B, T, in_channels, H, W)
    Sortie : (B, T, 1, H/patch_size, W/patch_size) — logits (avant sigmoid).

    Paramètres yaml :
      model:
        name: convlstm_v2
        hidden_dim: 64
        num_groups: 8   # diviseur de hidden_dim pour GroupNorm (optionnel, défaut 8)
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 64,
                 patch_size: int = 4, num_groups: int = 8):
        super().__init__()
        n_down = int(math.log2(patch_size))
        if 2 ** n_down != patch_size:
            raise ValueError(f"patch_size doit être une puissance de 2, reçu {patch_size}")
        if hidden_dim % num_groups != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) doit être divisible par num_groups ({num_groups})")

        stages = []
        c = in_channels
        for _ in range(n_down):
            stages.append(_encoder_stage(c, hidden_dim, num_groups))
            c = hidden_dim
        self.encoder = nn.Sequential(*stages)
        self.patch_size = patch_size

        self.cell = ConvLSTMCell(hidden_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        h, c = self.cell.init_state(B, Hp, Wp, x.device)
        outputs = []
        for t in range(T):
            feat = self.encoder(x[:, t])
            if feat.shape[-2:] != (Hp, Wp):
                feat = F.adaptive_avg_pool2d(feat, (Hp, Wp))
            h, c = self.cell(feat, (h, c))
            outputs.append(self.head(h))

        return torch.stack(outputs, dim=1)


@register_model("convlstm_v3")
class ConvLSTMSeg_v3(nn.Module):
    """ConvLSTM v3 — encodeur v2 + passe bidirectionnelle.

    Exécute la séquence dans les deux sens (forward + backward) puis
    concatène les états cachés avant la tête de classification. Cela
    permet au réseau d'exploiter le contexte futur pour chaque frame.

    Entrée : (B, T, in_channels, H, W)
    Sortie : (B, T, 1, H/patch_size, W/patch_size) — logits (avant sigmoid).

    Paramètres yaml :
      model:
        name: convlstm_v3
        hidden_dim: 64
        num_groups: 8
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 64,
                 patch_size: int = 4, num_groups: int = 8):
        super().__init__()
        n_down = int(math.log2(patch_size))
        if 2 ** n_down != patch_size:
            raise ValueError(f"patch_size doit être une puissance de 2, reçu {patch_size}")
        if hidden_dim % num_groups != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) doit être divisible par num_groups ({num_groups})")

        stages = []
        c = in_channels
        for _ in range(n_down):
            stages.append(_encoder_stage(c, hidden_dim, num_groups))
            c = hidden_dim
        self.encoder = nn.Sequential(*stages)
        self.patch_size = patch_size

        self.fwd_cell = ConvLSTMCell(hidden_dim, hidden_dim)
        self.bwd_cell = ConvLSTMCell(hidden_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        # Encode all frames up front (shared encoder)
        feats = []
        for t in range(T):
            f = self.encoder(x[:, t])
            if f.shape[-2:] != (Hp, Wp):
                f = F.adaptive_avg_pool2d(f, (Hp, Wp))
            feats.append(f)

        # Forward pass
        h_f, c_f = self.fwd_cell.init_state(B, Hp, Wp, x.device)
        fwd_h = []
        for t in range(T):
            h_f, c_f = self.fwd_cell(feats[t], (h_f, c_f))
            fwd_h.append(h_f)

        # Backward pass
        h_b, c_b = self.bwd_cell.init_state(B, Hp, Wp, x.device)
        bwd_h = [None] * T
        for t in range(T - 1, -1, -1):
            h_b, c_b = self.bwd_cell(feats[t], (h_b, c_b))
            bwd_h[t] = h_b

        # Predict at each step from combined forward + backward state
        outputs = [self.head(torch.cat([fwd_h[t], bwd_h[t]], dim=1)) for t in range(T)]
        return torch.stack(outputs, dim=1)


@register_model("convlstm_v4")
class ConvLSTMSeg_v4(nn.Module):
    """ConvLSTM v4 — encodeur 3 stages + décodeur U-Net + LSTM au goulot.

    Idée : descendre jusqu'à H/8 pour un grand champ récepteur, faire tourner
    le LSTM à ce niveau, puis remonter à H/4 avec un skip de l'encodeur (U-Net).
    La prédiction reste à H/4 (patch_size=4), compatible avec les configs existantes.

    Schéma :
      enc1 : (in_ch, H,   W)   → (D//2, H/2, W/2)
      enc2 : (D//2,  H/2, W/2) → (D,   H/4, W/4)  ← skip
      enc3 : (D,     H/4, W/4) → (D,   H/8, W/8)  ← goulot LSTM
      LSTM  : (D, H/8, W/8) → hidden state
      dec   : cat(hidden, skip_enc2) → (D//2, H/4, W/4)
      head  : → (1, H/4, W/4)

    Entrée : (B, T, in_channels, H, W)
    Sortie : (B, T, 1, H/4, W/4) — logits.

    Paramètres yaml :
      model:
        name: convlstm_v4
        hidden_dim: 96
        num_groups: 8
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 96,
                 patch_size: int = 4, num_groups: int = 8):
        super().__init__()
        if hidden_dim % num_groups != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) doit être divisible par num_groups ({num_groups})")
        if (hidden_dim // 2) % num_groups != 0:
            raise ValueError(f"hidden_dim//2 ({hidden_dim // 2}) doit être divisible par num_groups ({num_groups})")

        D = hidden_dim
        self.patch_size = patch_size

        self.enc1 = _encoder_stage(in_channels, D // 2, num_groups)   # → H/2
        self.enc2 = _encoder_stage(D // 2, D, num_groups)              # → H/4  (skip)
        self.enc3 = _encoder_stage(D, D, num_groups)                   # → H/8  (goulot)

        self.cell = ConvLSTMCell(D, D)

        self.dec = _DecoderBlock(c_skip=D, c_up=D, c_out=D // 2, num_groups=num_groups)

        self.head = nn.Sequential(
            nn.Conv2d(D // 2, D // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(D // 4, 1, kernel_size=1),
        )

    @staticmethod
    def _s2_out(n: int) -> int:
        """Taille de sortie d'un conv stride-2 (padding=1, kernel=3)."""
        return (n + 2 * 1 - 3) // 2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        # Résolution au goulot (3 stages stride-2)
        Hb = self._s2_out(self._s2_out(self._s2_out(H)))
        Wb = self._s2_out(self._s2_out(self._s2_out(W)))

        h, c = self.cell.init_state(B, Hb, Wb, x.device)
        outputs = []

        for t in range(T):
            xt = x[:, t]
            s2 = self.enc2(self.enc1(xt))    # (B, D,   H/4, W/4) — skip
            bot = self.enc3(s2)              # (B, D,   H/8, W/8) — goulot

            h, c = self.cell(bot, (h, c))

            dec = self.dec(h, s2)            # (B, D//2, H/4, W/4)
            out = self.head(dec)             # (B, 1,    H/4, W/4)

            # Aligne sur la résolution exacte attendue par le dataset
            if out.shape[-2:] != (Hp, Wp):
                out = F.adaptive_avg_pool2d(out, (Hp, Wp))

            outputs.append(out)

        return torch.stack(outputs, dim=1)


@register_model("convlstm_v4b")
class ConvLSTMSeg_v4b(nn.Module):
    """ConvLSTM v4 bidirectionnel — encodeur 3 stages + LSTM bidir + décodeur U-Net.

    Combine les avantages de v3 (contexte futur) et v4 (grand champ récepteur + skips).
    Deux cellules LSTM (forward + backward) à H/8 ; leurs états sont concaténés
    avant le décodeur U-Net qui remonte à H/4 via le skip d'enc2.

    Entrée : (B, T, in_channels, H, W)
    Sortie : (B, T, 1, H/4, W/4) — logits.

    Paramètres yaml :
      model:
        name: convlstm_v4b
        hidden_dim: 96
        num_groups: 8
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 96,
                 patch_size: int = 4, num_groups: int = 8):
        super().__init__()
        if hidden_dim % num_groups != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) doit être divisible par num_groups ({num_groups})")
        if (hidden_dim // 2) % num_groups != 0:
            raise ValueError(f"hidden_dim//2 ({hidden_dim // 2}) doit être divisible par num_groups ({num_groups})")

        D = hidden_dim
        self.patch_size = patch_size

        self.enc1 = _encoder_stage(in_channels, D // 2, num_groups)
        self.enc2 = _encoder_stage(D // 2, D, num_groups)    # → H/4 (skip)
        self.enc3 = _encoder_stage(D, D, num_groups)          # → H/8 (goulot)

        self.fwd_cell = ConvLSTMCell(D, D)
        self.bwd_cell = ConvLSTMCell(D, D)

        # Décodeur reçoit 2D depuis les deux cellules + D du skip
        self.dec = _DecoderBlock(c_skip=D, c_up=2 * D, c_out=D // 2, num_groups=num_groups)

        self.head = nn.Sequential(
            nn.Conv2d(D // 2, D // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(D // 4, 1, kernel_size=1),
        )

    @staticmethod
    def _s2_out(n: int) -> int:
        return (n + 2 * 1 - 3) // 2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        Hb = self._s2_out(self._s2_out(self._s2_out(H)))
        Wb = self._s2_out(self._s2_out(self._s2_out(W)))

        # Encoder toutes les frames (skips + goulots)
        s2_list, bot_list = [], []
        for t in range(T):
            s2 = self.enc2(self.enc1(x[:, t]))
            s2_list.append(s2)
            bot_list.append(self.enc3(s2))

        # Passe forward
        h_f, c_f = self.fwd_cell.init_state(B, Hb, Wb, x.device)
        fwd_h = []
        for t in range(T):
            h_f, c_f = self.fwd_cell(bot_list[t], (h_f, c_f))
            fwd_h.append(h_f)

        # Passe backward
        h_b, c_b = self.bwd_cell.init_state(B, Hb, Wb, x.device)
        bwd_h = [None] * T
        for t in range(T - 1, -1, -1):
            h_b, c_b = self.bwd_cell(bot_list[t], (h_b, c_b))
            bwd_h[t] = h_b

        # Décodage à chaque pas de temps
        outputs = []
        for t in range(T):
            h_cat = torch.cat([fwd_h[t], bwd_h[t]], dim=1)   # (B, 2D, H/8, W/8)
            dec = self.dec(h_cat, s2_list[t])                 # (B, D//2, H/4, W/4)
            out = self.head(dec)
            if out.shape[-2:] != (Hp, Wp):
                out = F.adaptive_avg_pool2d(out, (Hp, Wp))
            outputs.append(out)

        return torch.stack(outputs, dim=1)
