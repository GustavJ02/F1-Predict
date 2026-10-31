"""pre.race.predict_sc: reading deployments, weather and phase from a recording, the fit, and the simulated distributions."""
import numpy as np
import pandas as pd

import pre.race.predict_sc as M


def events(status, wet=(), total=50):
    """Recording of a 50 lap race: track_status rows [(lap, status), ...] and rain on the laps in `wet`."""
    rows = [dict(event="track_status", track_status=s, race_lap=lap, total_laps=total) for lap, s in status]
    rows += [dict(event="snapshot", race_lap=lap, total_laps=total, rainfall=1) for lap in wet]
    rows.append(dict(event="lap", race_lap=1, total_laps=total, compound="SOFT"))
    return pd.DataFrame(rows).reindex(columns=["event", "track_status", "race_lap", "total_laps", "rainfall", "compound"])


def test_race_record_deployments_lengths_and_laps_at_risk():
    r = M.race_record(events([(1, "Yellow"), (1, "AllClear"), (2, "VSCDeployed"), (2, "VSCDeployed"), (4, "VSCEnding"), (4, "AllClear"),
                              (20, "Yellow"), (20, "SCDeployed"), (24, "AllClear"), (30, "SCDeployed"), (31, "Red"), (32, "AllClear")],
                             wet=(10, 11, 12)), day_of_year=60 + (335 - 60) // 2)
    assert r["vsc"] == [(2, 2)]                           # a repeated row is the same deployment
    assert r["sc"] == [(20, 4), (30, 2)]
    # laps 3 (after the VSC), 21-23 (after the SC) and 31 (after the second SC) are blocked; the rest of laps 3-50 is at risk
    assert r["dry"] == 48 - 5 - 3 and r["wet"] == 3
    assert r["early_dry"] == 2 and r["early_wet"] == 0
    assert abs(r["phase"] - 0.5) < 0.01 and abs(r["wet_share"] - 3 / 50) < 1e-9


def test_wet_tyres_make_a_lap_wet():
    ev = events([(1, "AllClear")])
    ev = pd.concat([ev, pd.DataFrame([dict(event="lap", race_lap=7, total_laps=50, compound="INTERMEDIATE")] * 3
                                     + [dict(event="lap", race_lap=8, total_laps=50, compound="SOFT")] * 3)], ignore_index=True)
    wet = M.wet_laps(ev, 50)
    assert wet[7] and not wet[8] and wet.sum() == 1


def test_a_deployment_still_out_at_the_flag_is_counted():
    r = M.race_record(events([(1, "AllClear"), (49, "VSCDeployed")]))
    assert r["vsc"] == [(49, 1)]


def test_phase_of_the_season():
    assert M.phase_of(M.SEASON_START) == 0 and M.phase_of(M.SEASON_END) == 1 and M.phase_of(0) == 0 and M.phase_of(400) == 1


def test_nb_marginal_is_a_distribution_and_tends_to_poisson():
    n = np.arange(0, 60)
    assert abs(np.exp(M.nb_logpmf(n, 2.0, 100.0, 50.0)).sum() - 1) < 1e-9
    from scipy.stats import poisson
    assert abs(np.exp(M.nb_logpmf(3, 1e6, 1e6 / 0.02, 50.0)) - poisson.pmf(3, 1.0)) < 1e-4


def synthetic(n_races=300, rate=0.02, wet_mult=1.0, phase_slope=0.0, season_mult=None, seed=0):
    """Races whose late deployments are Poisson with `rate` per lap at risk, times wet / phase / season effects."""
    rng = np.random.default_rng(seed)
    records = {}
    for i in range(n_races):
        year = 2018 + i % 4
        phase, wet = rng.random(), int(rng.random() < 0.3) * 20
        dry = 57 - wet
        lam = rate * np.exp(phase_slope * (phase - 0.5)) * (season_mult or {}).get(year, 1.0) * (dry + wet_mult * wet)
        n = rng.poisson(lam)
        records[(str(year), f"C{i}")] = dict(laps=60, sc=[(10 + j, 3) for j in range(n)], vsc=[], dry=dry, wet=wet, early_dry=2,
                                             early_wet=0, wet_share=wet / 60, phase=phase)
    return records


def test_fit_recovers_the_rate_and_finds_no_circuit_differences():
    p = M.fit(synthetic())["kinds"]["sc"]
    assert abs(p["mu"] - 0.02) < 0.004 and p["a_circuit"] > 20


def test_fit_recovers_a_wet_effect():
    p = M.fit(synthetic(n_races=600, wet_mult=3.0))["kinds"]["sc"]
    assert 2.0 < np.exp(p["b_wet"]) < 4.5


def test_fit_recovers_a_phase_effect_and_season_differences():
    p = M.fit(synthetic(n_races=600, phase_slope=1.5, season_mult={2018: 0.5, 2019: 1.0, 2020: 1.5, 2021: 2.0}))["kinds"]["sc"]
    assert 1.5 < np.exp(p["b_phase"]) < 9                   # end vs start of the season is about x4.5
    assert p["a_season"] < 100                              # the seasons differ
    mean = {y: (lambda s, r: s / r)(*M.posterior({"kinds": {"sc": p}}, "sc", year=y)) for y in (2018, 2021)}
    assert mean[2021] > 1.5 * mean[2018]


def params(sc=0.0, vsc=0.0, b_wet=0.0, b_phase=0.0, early=0.0, a=1e4, wet_shares=(0.4,), p_wet_race=0.0):
    kind = lambda mu: dict(mu=mu, b_wet=b_wet, b_phase=b_phase, early=early, a_season=a, a_circuit=a, se_wet=0.1, se_phase=0.1,
                           seasons={}, circuits={})
    return dict(kinds=dict(sc=kind(sc), vsc=kind(vsc)), lengths=dict(sc=[3], vsc=[1]), wet_shares=list(wet_shares), p_wet_race=p_wet_race)


def test_simulation_counts():
    none = M.simulate(params(1e-9, 1e-9), "X", 2026, 50, sims=2000)
    assert none["sc"].sum() < 5 and none["vsc"].sum() < 5          # the rate floor is 1e-6 per lap
    c = M.simulate(params(sc=0.02), "X", 2026, 50, sims=50000)
    assert 0.6 < c["sc"].mean() < 1.0 and c["vsc"].sum() < 20      # nothing starts while one is out (3 laps)
    assert (M.simulate(params(sc=0.5), "X", 2026, 50, sims=2000)["sc"] <= 50 / 3 + 1).all()


def test_rain_raises_sc_counts_and_phase_raises_vsc_counts():
    base = params(sc=0.01, vsc=0.01, b_wet=1.2, b_phase=1.0)
    dry = M.simulate(base, "X", 2026, 60, rain_prob=0.0, sims=40000)
    wet = M.simulate(base, "X", 2026, 60, rain_prob=1.0, sims=40000)
    assert wet["sc"].mean() > 1.1 * dry["sc"].mean()
    late = M.simulate(base, "X", 2026, 60, phase=1.0, rain_prob=0.0, sims=40000)
    early = M.simulate(base, "X", 2026, 60, phase=0.0, rain_prob=0.0, sims=40000)
    assert late["vsc"].mean() > 1.5 * early["vsc"].mean()
    known = M.simulate(base, "X", 2026, 60, wet=np.ones(61, bool), sims=40000)      # every lap wet, as a known array
    assert known["sc"].mean() > wet["sc"].mean()


def test_distribution_table():
    c = M.simulate(params(0.02, 0.03), "X", 2026, 50, sims=20000)
    d = M.distribution(c)
    assert np.allclose(d.iloc[:M.SHOW + 1].sum(), 1.0)
    assert list(d.columns) == ["Full safety car (SC)", "VSC only", "SC or VSC"]
    assert abs(d.loc["mean", "SC or VSC"] - (d.loc["mean", "Full safety car (SC)"] + d.loc["mean", "VSC only"])) < 1e-9
    assert "mean" in M.table(d)
