"""Tests unitaires pour le pipeline EV-IMO."""
from pathlib import Path

import numpy as np
import pytest
import torch

EVIMO_NPZ = Path(__file__).resolve().parents[2] / "datasets/EV-IMO/eval/box/npz/seq_00.npz"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_array():
    from ms_model.io.loaders import load_events
    return load_events(EVIMO_NPZ)


@pytest.fixture(scope="module")
def slice_seq(event_array):
    from ms_model.representations.event_slice import make_slice_sequence
    ea = event_array
    return make_slice_sequence(ea, float(ea.t.min()), float(ea.t.min()) + 0.5, slice_dt=0.025)


# ---------------------------------------------------------------------------
# event_slice
# ---------------------------------------------------------------------------

class TestEventSlice:
    def test_to_event_slice_shape(self, event_array):
        from ms_model.representations.event_slice import to_event_slice
        ea = event_array
        s = to_event_slice(ea, float(ea.t.min()), float(ea.t.min()) + 0.025)
        assert s.shape == (3, 260, 346)
        assert s.dtype == torch.float32

    def test_to_event_slice_empty_window(self, event_array):
        from ms_model.representations.event_slice import to_event_slice
        ea = event_array
        s = to_event_slice(ea, 999.0, 999.025)  # hors plage → zéros
        assert s.shape == (3, 260, 346)
        assert s.sum() == 0.0

    def test_channels_non_negative(self, event_array):
        from ms_model.representations.event_slice import to_event_slice
        ea = event_array
        s = to_event_slice(ea, float(ea.t.min()), float(ea.t.min()) + 0.025)
        assert (s >= 0).all()

    def test_timestamp_channel_in_0_1(self, event_array):
        from ms_model.representations.event_slice import to_event_slice
        ea = event_array
        s = to_event_slice(ea, float(ea.t.min()), float(ea.t.min()) + 0.025)
        ts = s[2]  # timestamp channel
        assert ts.min() >= 0.0
        assert ts.max() <= 1.0

    def test_make_slice_sequence_shape(self, event_array):
        from ms_model.representations.event_slice import make_slice_sequence
        ea = event_array
        seq = make_slice_sequence(ea, float(ea.t.min()), float(ea.t.min()) + 0.1, slice_dt=0.025)
        assert seq.shape == (4, 3, 260, 346)

    def test_make_slice_sequence_finite(self, slice_seq):
        assert torch.isfinite(slice_seq).all()


# ---------------------------------------------------------------------------
# EvimoSliceDataset
# ---------------------------------------------------------------------------

class TestEvimoSliceDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        from ms_model.data.evimo_slice_dataset import EvimoSliceDataset
        return EvimoSliceDataset([EVIMO_NPZ], slice_dt=0.025, n_slices_context=5)

    def test_len_positive(self, dataset):
        assert len(dataset) > 0

    def test_item_keys(self, dataset):
        item = dataset[0]
        assert set(item.keys()) == {"slices", "depth_gt", "mask_gt", "ego_vel", "K"}

    def test_slices_shape(self, dataset):
        item = dataset[0]
        assert item["slices"].shape == (5, 3, 260, 346)

    def test_depth_gt_shape(self, dataset):
        item = dataset[0]
        assert item["depth_gt"].shape == (260, 346)

    def test_mask_gt_values(self, dataset):
        # mask_gt doit être entier ≥ 0
        item = dataset[0]
        assert item["mask_gt"].dtype == torch.int64
        assert (item["mask_gt"] >= 0).all()

    def test_ego_vel_shape(self, dataset):
        item = dataset[0]
        assert item["ego_vel"].shape == (6,)

    def test_K_shape(self, dataset):
        item = dataset[0]
        assert item["K"].shape == (3, 3)

    def test_slices_finite(self, dataset):
        item = dataset[0]
        assert torch.isfinite(item["slices"]).all()


# ---------------------------------------------------------------------------
# ECNBackbone
# ---------------------------------------------------------------------------

class TestECNBackbone:
    @pytest.mark.parametrize("variant,min_p,max_p", [
        ("baseline",   900_000, 4_000_000),
        ("shallow",  10_000,    100_000),
    ])
    def test_param_count(self, variant, min_p, max_p):
        from ms_model.models.evimo.ecn import ECNBackbone
        from ms_model.models.evimo.evimo_model import _ECN_VARIANTS
        cfg = _ECN_VARIANTS[variant]
        m = ECNBackbone(in_channels=3, out_channels=1, **cfg)
        n = sum(p.numel() for p in m.parameters())
        assert min_p < n < max_p, f"{variant}: {n} params hors plage [{min_p}, {max_p}]"

    def test_forward_shape(self):
        from ms_model.models.evimo.ecn import ECNBackbone
        m = ECNBackbone(in_channels=3, out_channels=1, init_channels=8, growth=8, n_stages=2)
        out = m(torch.randn(2, 3, 260, 346))
        assert out.shape == (2, 1, 260, 346)

    def test_encode_decode_roundtrip(self):
        from ms_model.models.evimo.ecn import ECNBackbone
        m = ECNBackbone(in_channels=3, out_channels=1, init_channels=8, growth=8, n_stages=2)
        x = torch.randn(2, 3, 260, 346)
        btl, feats = m.encode(x)
        out = m.decode(btl, feats)
        assert out.shape == (2, 1, 260, 346)


# ---------------------------------------------------------------------------
# EvimoModel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fake_batch():
    B = 2
    return {
        "slices": torch.randn(B, 5, 3, 260, 346),
        "K": torch.tensor([[[460., 0, 173], [0, 460, 130], [0, 0, 1.]]]).expand(B, -1, -1),
        "depth_gt": torch.rand(B, 260, 346) * 700 + 500,
        "mask_gt": torch.zeros(B, 260, 346, dtype=torch.long),
    }


class TestEvimoModel:
    @pytest.fixture(scope="class")
    def model(self):
        from ms_model.models import build_model
        return build_model("evimo", ecn_variant="shallow")

    def test_registered(self):
        from ms_model.models.base import MODEL_REGISTRY
        assert "evimo" in MODEL_REGISTRY

    def test_output_keys(self, model, fake_batch):
        out = model(fake_batch)
        assert set(out.keys()) == {"depth", "poses", "masks", "flow"}

    def test_depth_shape(self, model, fake_batch):
        out = model(fake_batch)
        assert out["depth"].shape == (2, 1, 260, 346)

    def test_depth_positive(self, model, fake_batch):
        out = model(fake_batch)
        assert (out["depth"] > 0).all()

    def test_poses_shape(self, model, fake_batch):
        out = model(fake_batch)
        assert out["poses"].shape == (2, 15)  # 6 + 3*3

    def test_masks_shape(self, model, fake_batch):
        out = model(fake_batch)
        assert out["masks"].shape == (2, 4, 260, 346)  # C+1 = 4

    def test_masks_sum_to_one(self, model, fake_batch):
        out = model(fake_batch)
        sums = out["masks"].sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_flow_shape(self, model, fake_batch):
        out = model(fake_batch)
        assert out["flow"].shape == (2, 2, 260, 346)

    def test_backward(self, model, fake_batch):
        from ms_model.training.evimo_losses import EvimoLoss
        out = model(fake_batch)
        loss, _ = EvimoLoss()(out, fake_batch)
        loss.backward()  # ne doit pas lever d'exception


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestEvimoLosses:
    def test_warp_loss_zero_flow(self):
        from ms_model.training.evimo_losses import WarpLoss
        slices = torch.rand(2, 5, 3, 64, 64)
        flow = torch.zeros(2, 2, 64, 64)
        loss = WarpLoss()(slices, flow)
        # Avec flow=0, slices voisines ≠ slice centrale → loss > 0
        assert loss.item() > 0

    def test_depth_loss_valid_pixels(self):
        from ms_model.training.evimo_losses import DepthLoss
        pred = torch.full((2, 1, 10, 10), 700.0)
        gt = torch.full((2, 10, 10), 700.0)
        loss = DepthLoss(lambda_smooth=0.0)(pred, gt)
        # pred == gt → ratio=1, abs_rel=0 → loss ≈ 1.0
        assert abs(loss.item() - 1.0) < 0.01

    def test_depth_loss_all_nan(self):
        from ms_model.training.evimo_losses import DepthLoss
        pred = torch.rand(2, 1, 10, 10)
        gt = torch.full((2, 10, 10), float("nan"))
        loss = DepthLoss()(pred, gt)
        assert torch.isfinite(loss)

    def test_mask_loss_trivial_penalized(self):
        from ms_model.training.evimo_losses import MaskLoss
        # all-background prediction quand il y a des objets → loss élevée
        masks = torch.zeros(1, 4, 10, 10)
        masks[:, 0] = 1.0
        mask_gt = torch.zeros(1, 10, 10, dtype=torch.long)
        mask_gt[0, 3:7, 3:7] = 1
        loss_trivial = MaskLoss(lambda_smooth=0.0)(masks, mask_gt)

        # prediction correcte → loss faible
        masks_ok = masks.clone()
        masks_ok[0, 0, 3:7, 3:7] = 0.05
        masks_ok[0, 1, 3:7, 3:7] = 0.95
        loss_ok = MaskLoss(lambda_smooth=0.0)(masks_ok, mask_gt)

        assert loss_trivial.item() > loss_ok.item()

    def test_evimo_loss_backward(self, fake_batch):
        from ms_model.models import build_model
        from ms_model.training.evimo_losses import EvimoLoss
        model = build_model("evimo", ecn_variant="shallow")
        out = model(fake_batch)
        total, comps = EvimoLoss()(out, fake_batch)
        assert torch.isfinite(total)
        assert all(torch.isfinite(v) for v in comps.values())
        total.backward()
