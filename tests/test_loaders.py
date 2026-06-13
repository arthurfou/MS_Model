from pathlib import Path

import numpy as np
import pytest

from ms_model.io.loaders import load_events, load_evimo_npz, load_npy_canonical

IPAL_ROOT = Path(__file__).resolve().parents[2]
EVIMO_NPZ = IPAL_ROOT / "EVOwithMS/datasets/EV-IMO/eval/tabletop/npz/seq_00.npz"
CANONICAL_NPY = IPAL_ROOT / "EVOwithMS/datasets/EV-IMO_filtered/eval/tabletop/raw/seq_00/evs.npy"
CANONICAL_TSS = IPAL_ROOT / "EVOwithMS/datasets/EV-IMO_filtered/eval/tabletop/raw/seq_00/tss_imgs_us.txt"


def test_load_evimo_npz():
    ea = load_evimo_npz(EVIMO_NPZ)

    assert ea.H == 260
    assert ea.W == 346
    assert ea.events.shape[1] == 4
    assert len(ea) == ea.events.shape[0]

    # t en secondes, croissant, démarre à 0
    assert ea.t.min() == 0.0
    assert ea.t.max() > ea.t.min()
    assert np.all(np.diff(ea.t) >= 0)

    # polarité dans {0, 1}
    assert set(np.unique(ea.p)) <= {0.0, 1.0}

    # une frame timestamp par entrée de "index"
    assert ea.frame_ts is not None
    assert ea.frame_ts.ndim == 1
    assert ea.frame_ts[0] == ea.t.min()


def test_load_npy_canonical_converts_us_to_s():
    raw = np.load(CANONICAL_NPY)
    ea = load_npy_canonical(CANONICAL_NPY, frame_ts_path=CANONICAL_TSS)

    # même nombre d'events, mêmes x/y/p, seul t change d'unité
    assert ea.events.shape == raw.shape
    np.testing.assert_allclose(ea.t, raw[:, 0] / 1e6)
    np.testing.assert_allclose(ea.x, raw[:, 1])
    np.testing.assert_allclose(ea.y, raw[:, 2])
    np.testing.assert_allclose(ea.p, raw[:, 3])

    # frame_ts converti en secondes aussi
    raw_tss = np.loadtxt(CANONICAL_TSS, dtype=np.float64)
    np.testing.assert_allclose(ea.frame_ts, raw_tss / 1e6)


def test_load_events_dispatches_on_extension():
    ea_npz = load_events(EVIMO_NPZ)
    assert ea_npz.H == 260 and ea_npz.W == 346

    ea_npy = load_events(CANONICAL_NPY)
    assert ea_npy.frame_ts is None  # pas de frame_ts_path passé ici


def test_load_events_unsupported_format(tmp_path):
    bogus = tmp_path / "events.bag"
    bogus.write_bytes(b"not really a bag")

    with pytest.raises(ValueError):
        load_events(bogus)
