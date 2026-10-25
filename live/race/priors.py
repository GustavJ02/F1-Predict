"""Race priors estimated from the races we have recorded: what the projection (live.race.projection) assumes
before this race shows its own behaviour.

    lap model     lap_time = driver pace + compound offset + wear * tyre age - fuel * lap         (pooled regression)
    stints        how long stints last per compound as a Kaplan-Meier survival curve (stints that end with the race or a
                  retirement are censored: they were at least that long), per circuit and pooled; which compound follows which
    pit loss      seconds lost by a stop (in lap + out lap vs normal pace), per circuit and pooled
    passing       P(a car within 1.5 s gets past the car ahead on a lap) = logistic(a + b * pace advantage + circuit effect)
    safety car    SC / VSC deployments per lap (per circuit, shrunk towards the pooled rate), typical length in laps,
                  and how many times likelier one is in the first two laps (`early_boost`)
    start         the share of cars starting the race on each compound (pooled)

Estimated by `python -m live.race.priors`, which writes live/race/priors.json (committed, so the live runner doesn't
need the data directory). Re-run it after a few more races. `collect` + `fit(..., exclude=...)` allow a
leave-one-race-out fit for backtests.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from f1live.events import has_events, load
from f1live.paths import DATA
from pre.history import race_key
from live.race.battles import pair_table
from live.race.pace import clean_laps, lap_rows

PRIORS_FILE = Path(__file__).with_name("priors.json")
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
CIRCUIT_ALIASES = {("2025", "Spanish"): "Barcelona"}      # the 2025 'Spanish' GP was Barcelona; 2026 Spanish is Madrid
PASS_CLIP = (-1.0, 2.0)
MIN_TEMP = 5.0           # track temperatures at or below this are sensor gaps
MIN_CIRCUIT_STINTS = 10          # stints of a compound needed before a circuit gets its own survival curve
SHRINK_STOPS, SHRINK_PASS, SHRINK_SC = 3, 40, 25          # pseudo-observations pulling a circuit towards the pooled value


def circuit_of(year, race):
    return CIRCUIT_ALIASES.get((str(year), race), race_key(race))


# ---------------------------------------------------------------------------------------------- collection

def collect_race(h, with_passes=True):
    """Small per-race tables from one race's events: clean laps, stints, pit losses, SC events, pass opportunities."""
    laps = lap_rows(h)
    total = int(h["total_laps"].max())
    c = clean_laps(laps)
    c = c[c["compound"].isin(COMPOUNDS) & c["tyre_laps"].notna()].copy()
    c["med"] = c.groupby(["driver", "stint"])["lap_time_s"].transform("median")
    c = c[c["lap_time_s"] < c["med"] + 1.5]
    c["temp"] = c["track_temp"].where(c["track_temp"] > MIN_TEMP)            # 0 / missing readings are sensor gaps
    clean = c.rename(columns={"tyre_laps": "age", "lap_number": "lap"})[["driver", "stint", "compound", "age", "lap", "lap_time_s", "temp"]]

    stints, loss = [], []
    for drv, g in laps.groupby("driver"):
        g = g.reset_index(drop=True)
        dry = g[g["compound"].isin(COMPOUNDS)]
        for st, gg in dry.groupby("stint"):
            stints.append(dict(driver=drv, stint=st, compound=gg["compound"].iloc[-1], length=int(gg["tyre_laps"].max()),
                               pit=bool((dry["stint"] > st).any()), total=total))
        for i, r in g.iterrows():
            if r["lap_kind"] == "in" and i + 1 < len(g) and g.loc[i + 1, "lap_kind"] == "out" \
                    and r["track_status"] == "AllClear" and g.loc[i + 1, "track_status"] == "AllClear":
                nb = g[(g["lap_kind"] == "racing") & (g["track_status"] == "AllClear")
                       & g["lap_number"].between(r["lap_number"] - 4, r["lap_number"] + 5)]["lap_time_s"]
                if len(nb) >= 4:
                    loss.append(dict(loss=float(np.clip(r["lap_time_s"] + g.loc[i + 1, "lap_time_s"] - 2 * nb.median(), 5, 60))))

    ts = h[h["event"] == "track_status"][["ts", "track_status", "race_lap"]]
    deployed = ts["track_status"].isin(["SCDeployed", "VSCDeployed"])
    sc = dict(laps=total, sc=int((ts["track_status"] == "SCDeployed").sum()), vsc=int((ts["track_status"] == "VSCDeployed").sum()),
              early=int((deployed & (ts["race_lap"] <= 2)).sum()), sc_laps=[], )
    start = None
    for _, r in ts.iterrows():
        if r["track_status"] in ("SCDeployed", "VSCDeployed") and start is None:
            start = (r["race_lap"], r["track_status"])
        elif r["track_status"] == "AllClear" and start is not None:
            sc["sc_laps"].append((start[1], float(max(r["race_lap"] - start[0], 1))))
            start = None

    passes = collect_passes(h, laps) if with_passes else pd.DataFrame(columns=["delta", "passed"])
    return dict(clean=clean, stints=pd.DataFrame(stints), loss=pd.DataFrame(loss), sc=sc, passes=passes)


def collect_passes(h, laps):
    """One row per (lap, adjacent pair within 1.5 s, green, no stops, same lap): pace advantage and whether it passed."""
    by_drv = {d: g.sort_values("ts").set_index("lap_number") for d, g in laps.groupby("driver")}
    rows = []
    for lap0 in range(6, int(h["total_laps"].max()) - 2):
        up = h[h["race_lap"] <= lap0]
        cur = up[up["race_lap"] == lap0]
        if cur.empty or cur["track_status"].iloc[-1] != "AllClear":
            continue
        _, pairs = pair_table(up)
        if pairs.empty:
            continue
        last_n = up[up["event"] == "lap"].groupby("driver")["lap_number"].max()
        for _, q in pairs[pairs["gap_s"] <= 1.5].iterrows():
            a, b = by_drv.get(q["front"]), by_drv.get(q["rear"])
            na, nb = last_n.get(q["front"]), last_n.get(q["rear"])
            if a is None or b is None or na is None or nb is None:
                continue
            if any(k not in idx.index for idx, k in ((a, na), (a, na + 1), (b, nb), (b, nb + 1))):
                continue
            a0, a1, b0, b1 = a.loc[na], a.loc[na + 1], b.loc[nb], b.loc[nb + 1]
            if "AllClear" != b1["track_status"] or "AllClear" != a1["track_status"]:
                continue
            if b1["n_pit_stops"] != b0["n_pit_stops"] or a1["n_pit_stops"] != a0["n_pit_stops"]:
                continue
            if b0["laps_down"] != a0["laps_down"] or b1["laps_down"] != a1["laps_down"]:
                continue
            rows.append(dict(delta=float(q["delta_s"]), passed=bool(b1["position"] < a1["position"])))
    return pd.DataFrame(rows, columns=["delta", "passed"])


def collect(data_dir=DATA, with_passes=True, verbose=False):
    """{(year, race): per-race tables} for every race with an events.csv."""
    out = {}
    for p in sorted(q for q in Path(data_dir).glob("*/*/race/events.csv") if has_events(q)):
        year, race = p.parent.parent.parent.name, p.parent.parent.name
        if verbose:
            print("collecting", year, race, flush=True)
        out[(year, race)] = collect_race(load(p), with_passes)
    return out


# ---------------------------------------------------------------------------------------------- fitting

def fit_lap_model(items):
    d = pd.concat([i["clean"].assign(race=f"{k[0]}/{k[1]}") for k, i in items.items()])
    g = d.groupby(["race", "driver"])
    y = d["lap_time_s"] - g["lap_time_s"].transform("mean")
    X = pd.DataFrame({"lap": d["lap"] - g["lap"].transform("mean")})
    temp_ref = float(d["temp"].mean())
    dt = (d["temp"] - temp_ref).fillna(0.0)                  # a lap without a temperature reading is taken at the reference
    for comp in COMPOUNDS:
        ind = (d["compound"] == comp).astype(float)
        X["off_" + comp] = ind - ind.groupby([d["race"], d["driver"]]).transform("mean")
        a = d["age"] * ind
        X["age_" + comp] = a - a.groupby([d["race"], d["driver"]]).transform("mean")
        at = a * dt
        X["ageT_" + comp] = at - at.groupby([d["race"], d["driver"]]).transform("mean")
    beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    b = dict(zip(X.columns, beta))
    off = {c: b["off_" + c] for c in COMPOUNDS}
    ref = off["MEDIUM"]                                      # driver pace is the level on mediums
    resid = y.values - X.values @ beta
    return dict(fuel=float(-b["lap"]), offset={c: float(off[c] - ref) for c in COMPOUNDS},
                wear={c: float(b["age_" + c]) for c in COMPOUNDS}, lap_noise=float(np.std(resid)),
                wear_temp={c: float(b["ageT_" + c]) for c in COMPOUNDS}, temp_ref=temp_ref)


def kaplan_meier(stints):
    """Survival curve of tyre age at the stop: {'age': [a1, a2, ...], 'surv': [S(a1), ...]}, S = P(stint lasts beyond a).

    stints: DataFrame with length (tyre age at the end of the stint) and pit (True = ended by a stop, False = censored)."""
    s = stints[["length", "pit"]].dropna()
    ages, surv, S = [], [], 1.0
    for a in sorted(s.loc[s["pit"], "length"].unique()):
        at_risk = int((s["length"] >= a).sum())
        events = int(((s["length"] == a) & s["pit"]).sum())
        S *= 1 - events / at_risk
        ages.append(int(a))
        surv.append(float(S))
    return dict(age=ages, surv=surv)


def shrink(circuit_vals, pooled, k):
    """{circuit: (n, value)} -> shrunk values; value = (n*v + k*pooled) / (n + k)."""
    return {c: float((n * v + k * pooled) / (n + k)) for c, (n, v) in circuit_vals.items()}


def fit_logistic(delta, passed, circuit, ridge=2.0, iters=40):
    """logit(p) = a + b*delta + effect[circuit], ridge-penalised circuit effects. -> a, b, {circuit: effect}"""
    cs = sorted(set(circuit))
    idx = np.array([cs.index(c) for c in circuit])
    n, k = len(delta), len(cs)
    X = np.zeros((n, 2 + k))
    X[:, 0], X[:, 1] = 1.0, np.clip(delta, *PASS_CLIP)
    X[np.arange(n), 2 + idx] = 1.0
    y = np.asarray(passed, float)
    pen = np.r_[0, 0, np.full(k, ridge)]
    w = np.zeros(2 + k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) + pen * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen) + 1e-6 * np.eye(2 + k)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1]), {c: float(w[2 + i]) for i, c in enumerate(cs)}


MIN_YEAR_RACES = 6


def fit(collected, exclude=(), lap_year=None):
    """Priors (a JSON-able dict) from per-race tables; races in `exclude` ((year, race) keys) are left out.

    The lap model (compound offsets and wear differ from season to season) is fitted on the races of `lap_year`
    only, if there are at least MIN_YEAR_RACES of them; everything else uses all races."""
    items = {k: v for k, v in collected.items() if k not in set(exclude)}
    circ = {k: circuit_of(*k) for k in items}
    same_year = {k: v for k, v in items.items() if lap_year is not None and k[0] == str(lap_year)}
    lap_model = fit_lap_model(same_year if len(same_year) >= MIN_YEAR_RACES else items)

    stints = pd.concat([i["stints"].assign(circuit=circ[k], race=f"{k[0]}/{k[1]}") for k, i in items.items()],
                       ignore_index=True)
    stint = dict(pooled={c: kaplan_meier(stints[stints["compound"] == c]) for c in COMPOUNDS}, circuits={})
    for c in sorted(set(circ.values())):
        sub = stints[stints["circuit"] == c]
        km = {k: kaplan_meier(sub[sub["compound"] == k]) for k in COMPOUNDS if (sub["compound"] == k).sum() >= MIN_CIRCUIT_STINTS}
        km = {k: v for k, v in km.items() if v["age"]}
        if km:
            stint["circuits"][c] = km
    seqs = stints.sort_values(["race", "driver", "stint"]).groupby(["race", "driver"])["compound"].apply(list)
    trans = {a: {b: 0 for b in COMPOUNDS} for a in COMPOUNDS}
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            trans[a][b] += 1
    pit_all = pd.concat([i["loss"].assign(circuit=circ[k]) for k, i in items.items()])
    pooled_loss = float(pit_all["loss"].median())
    pit = dict(pooled=pooled_loss, circuits=shrink({c: (len(g), float(g["loss"].median()))
                                                    for c, g in pit_all.groupby("circuit")}, pooled_loss, SHRINK_STOPS))

    pas = pd.concat([i["passes"].assign(circuit=circ[k]) for k, i in items.items()])
    a, b, eff = fit_logistic(pas["delta"].to_numpy(float), pas["passed"].to_numpy(), pas["circuit"].tolist())
    passing = dict(a=a, b=b, circuits=eff)

    laps = sum(i["sc"]["laps"] for i in items.values())
    sc_n = sum(i["sc"]["sc"] for i in items.values())
    vsc_n = sum(i["sc"]["vsc"] for i in items.values())
    lens = [l for i in items.values() for kind, l in i["sc"]["sc_laps"] if kind == "SCDeployed"]
    vlens = [l for i in items.values() for kind, l in i["sc"]["sc_laps"] if kind == "VSCDeployed"]
    per_c = {}
    for k, i in items.items():
        d = per_c.setdefault(circ[k], dict(laps=0, sc=0, vsc=0))
        d["laps"] += i["sc"]["laps"]; d["sc"] += i["sc"]["sc"]; d["vsc"] += i["sc"]["vsc"]
    early = sum(i["sc"].get("early", 0) for i in items.values())
    early_boost = float(np.clip((early / (2 * len(items))) / ((sc_n + vsc_n) / laps), 1.0, 8.0))
    safety = dict(sc_rate=sc_n / laps, vsc_rate=vsc_n / laps, early_boost=early_boost, sc_laps=float(np.mean(lens)) if lens else 4.0,
                  vsc_laps=float(np.mean(vlens)) if vlens else 2.0,
                  circuits={c: dict(sc_rate=(d["sc"] + SHRINK_SC * sc_n / laps) / (d["laps"] + SHRINK_SC),
                                    vsc_rate=(d["vsc"] + SHRINK_SC * vsc_n / laps) / (d["laps"] + SHRINK_SC))
                            for c, d in per_c.items()})
    first = stints[stints["stint"] == stints.groupby(["race", "driver"])["stint"].transform("min")]
    shares = first["compound"].value_counts(normalize=True)
    start = {c: float(shares.get(c, 0.0)) for c in COMPOUNDS}
    return dict(races=sorted(f"{y}/{r}" for y, r in items), lap_model=lap_model, stint=stint, transitions=trans,
                pit_loss=pit, passing=passing, safety=safety, start=start)


def save(priors, path=PRIORS_FILE):
    Path(path).write_text(json.dumps(priors, indent=1, sort_keys=True))


def load_priors(path=PRIORS_FILE):
    return json.loads(Path(path).read_text())


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA, type=Path)
    ap.add_argument("--out", default=PRIORS_FILE, type=Path)
    a = ap.parse_args()
    collected = collect(a.data_dir, verbose=True)
    priors = fit(collected)      # pooled over seasons: a per-season lap model (fit(..., lap_year=...)) backtested no better
    save(priors, a.out)
    lm, ps, sf = priors["lap_model"], priors["passing"], priors["safety"]
    print(f"\nwrote {a.out} from {len(priors['races'])} races")
    print(f"lap model: fuel {lm['fuel']:.3f} s/lap, offsets {lm['offset']}, wear {lm['wear']}, noise {lm['lap_noise']:.2f}")
    print(f"pit loss {priors['pit_loss']['pooled']:.1f} s; passing logit = {ps['a']:.2f} + {ps['b']:.2f} * pace advantage; "
          f"SC {sf['sc_rate']:.4f}/lap ({sf['sc_laps']:.1f} laps), VSC {sf['vsc_rate']:.4f}/lap")
    eff = pd.Series(ps["circuits"]).sort_values()
    print("hardest to pass:", ", ".join(f"{c} {v:+.1f}" for c, v in eff.head(4).items()),
          "| easiest:", ", ".join(f"{c} {v:+.1f}" for c, v in eff.tail(4).items()))


if __name__ == "__main__":
    main()
