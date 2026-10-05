import numpy as np
import mlacs as m


def test_theoretical_acf_and_variance():
    assert np.allclose(m.ar_acf((0.5,), 3), [0.5, 0.25, 0.125])
    r = m.ar_acf((0.3, 0.4), 2)
    assert np.allclose(r, [0.5, 0.55])
    assert np.isclose(m.ar_variance((0.5,)), 4 / 3)


def test_variance_is_preserved():
    rng = np.random.default_rng(0)
    x = m.simulate_piecewise_ar([m.Segment((0.5,), 3000), m.Segment((0.3, 0.4), 3000)], 300, rng=rng)
    assert abs(x[:, :3000].var() - 1) < 0.03 and abs(x[:, 3000:].var() - 1) < 0.03


def test_block_acf_shape():
    x = np.random.default_rng(1).standard_normal((4, 1000))
    assert m.block_acf(x, 50, 3).shape == (4, 20, 3)


def test_calibration_and_detection():
    rng = np.random.default_rng(2)
    null = m.simulate_piecewise_ar([m.Segment((0.5,), 3000)], 600, rng=rng)
    d = m.MLACS(lags=3)
    h = m.calibrate_threshold(d, null[:300], 2000, 0.05)
    S, _, _ = d.statistic_path(null[300:], 2000)
    far = (S.max(axis=1) > h).mean()
    assert far < 0.12
    x = m.simulate_piecewise_ar([m.Segment((0.5,), 3010), m.Segment((0.3, 0.4), 1990)], 100, rng=rng)
    r = d.run(x, 2000, h)
    assert (r["alarm_time"] > 3010).mean() > 0.8


def test_offline_split():
    rng = np.random.default_rng(3)
    x = m.simulate_piecewise_ar([m.Segment((0.5,), 3010), m.Segment((0.3, 0.4), 1990)], 1, rng=rng)[0]
    assert abs(m.offline_single_split(x, "ar", 2) - 3010) < 200
