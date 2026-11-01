"""Calibration of the cutoff band: scaling a band, and recovering the factors from predictions whose band is known to be too narrow."""
import numpy as np
import pandas as pd

from live.quali import calibrate_band as CB
from live.quali import predict_elimination as PE


def test_scale_band_widens_each_side_by_its_factor_and_interpolates_in_the_time_left():
    table = [(0.0, 1.0, 1.0), (600.0, 2.0, 3.0)]
    assert PE.scale_band(90.0, 89.8, 90.4, 0.0, table) == (89.8, 90.4)                    # factor 1: unchanged
    lo, hi = PE.scale_band(90.0, 89.8, 90.4, 600.0, table)
    assert np.isclose(lo, 90.0 - 2 * 0.2) and np.isclose(hi, 90.0 + 3 * 0.4)
    lo, hi = PE.scale_band(90.0, 89.8, 90.4, 300.0, table)                                # halfway between the two rows
    assert np.isclose(lo, 90.0 - 1.5 * 0.2) and np.isclose(hi, 90.0 + 2.0 * 0.4)
    assert np.isclose(PE.scale_band(90.0, 89.8, 90.4, 5000.0, table)[1], 90.0 + 3 * 0.4)  # beyond the table: the last row
    assert PE.scale_band(90.0, 89.8, 90.4, 100.0, None) == (89.8, 90.4)                    # no calibration
    assert np.isnan(PE.scale_band(np.nan, np.nan, np.nan, 100.0, table)[0])


def too_narrow(n=6000, seed=0, true_sd=0.5, claimed_half=0.25):
    """Predictions whose real error is N(0, true_sd) but whose band claims +-claimed_half (a 90 % band would need 1.645 * true_sd)."""
    rng = np.random.default_rng(seed)
    truth = 90.0 + rng.normal(0, true_sd, n)
    return pd.DataFrame(dict(session=[f"s{i % 10}" for i in range(n)], part=1 + np.arange(n) % 2, clock=rng.uniform(1, 900, n), truth=truth,
                             pred=90.0, lo=90.0 - claimed_half, hi=90.0 + claimed_half))


def test_the_fit_recovers_the_factor_that_gives_ninety_percent_coverage():
    df = too_narrow()
    table = CB.fit(df)
    need = 1.645 * 0.5 / 0.25                                                             # 3.29
    assert all(abs(down - need) < 0.5 and abs(up - need) < 0.5 for _, down, up in table)
    out = CB.apply(df, table)
    assert abs(out["covered_cal"].mean() - 0.90) < 0.02
    assert ((df["lo"] <= df["truth"]) & (df["truth"] <= df["hi"])).mean() < 0.5               # the band as claimed covers well under half


def test_the_fit_is_honest_out_of_sample_and_separates_the_two_sides():
    train, test = too_narrow(seed=1), too_narrow(seed=2)
    train["truth"] = train["truth"] + 0.15                                                 # the real cutoff sits above the prediction: bias
    table = CB.fit(train)
    assert all(up > down for _, down, up in table)                                         # more room is needed above than below
    out = CB.apply(test.assign(truth=test["truth"] + 0.15), table)
    assert out["covered_cal"].mean() > 0.85


def test_a_band_that_is_already_wide_enough_is_left_alone():
    df = too_narrow(claimed_half=0.82)                                                     # 1.645 * 0.5
    assert all(abs(down - 1.0) < 0.2 and abs(up - 1.0) < 0.2 for _, down, up in CB.fit(df))
