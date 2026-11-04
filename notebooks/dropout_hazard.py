"""
RQ2: dropout hazard along the running segment, interval-censored.

Why this script exists
----------------------
`total_distance_km` from the official platform is the cumulative distance at the
LAST TIMING CHECKPOINT an athlete crossed, not a continuous distance. Within one
event, run-segment DNFs take only a handful of distinct values (dropout_hazard.txt
section 1). "Kilometres run before dropout" is therefore a lower bound whose granularity
varies by event, and a histogram of it compared with a uniform distribution is confounded
by where the mats are. This script estimates instead a per-km dropout hazard that respects
the interval censoring.

Course positions
----------------
Each event's own cycling completion and finish (modal finisher distances; _common.
event_course_table, nominal fallback) define its run: km_run = total_distance_km - cycling
completion at two decimals, and the run length = finish - cycling completion
(_common.run_position; the same definition in dropout_point.py section 5). The recorded distances
of events whose finishers' distances do not form a usable course (basis nominal_incoherent or
nominal_scale) are not used (_common.attach_course): their DNFs count as DNFs without a distance.

Run DNFs (main analysis, T2 rule)
---------------------------------
  * in-race DNF (DNF without a recorded finish time) beyond the cycling completion and before
    the finish distance: run DNF at its km_run;
  * in-race DNF exactly at the cycling-completion distance WITH a T2 split > 0 of at most the T2
    duration limit of its event type (the 99th percentile of the official-source finishers' T2
    splits, _common.t2_limits): it crossed the timing mat at the transition exit, so it enters the
    run at km 0. With a longer T2 split, or without one, it withdrew in the transition and is not a
    run DNF.
Sensitivity "strict": no DNF at the cycling-completion distance is a run DNF. Sensitivity
"t2_no_limit": the T2 rule without the duration limit (any T2 split > 0). Sensitivity
"including_stopped": the main analysis with the stopped or wholesale-coded editions kept
(_common.STOPPED_EDITIONS, excluded from every other analysis; T2 limits of the analysed
population). Records at or beyond the finish distance are not run DNFs (they crossed the last mat of
the course; their interval would have zero width) and are reported.

Estimator
---------
For each event (official source, one race type):
  * mats      = distinct km_run values observed among its run DNFs (km 0 included)
  * interval i = [m_i, m_{i+1}) ; the last interval ends at the event's run length
  * d_i       = run DNFs whose last recorded checkpoint is m_i (the interval start)
  * n_i       = athletes at risk at m_i = athletes with a recorded finish time (finishers
                + finish-time DNFs of that event) + run DNFs with last mat >= m_i
  * h_i       = d_i / (n_i * w_i)                      [DNF per athlete-km]
Pooled over events: sum(d) / sum(n * w) within a band of interval starts, and a
per-km curve obtained by spreading each interval uniformly over the 1-km grid.
Cluster bootstrap over events (B = N_BOOT; a fresh generator seeded with SEED in
every bootstrap call, so each estimate is reproducible on its own) gives 95% CIs.

Conventions / caveats (declared in Methods)
------------------------------------------
  * Mats at the start of the run = mats below START_MAT_KM past the cycling completion. Under the
    T2 rule km 0 (the transition exit) is a mat of every event in which at least one athlete with a
    T2 split within the limit withdrew there; mats within 1 km keep their role. "First checkpoint"
    = first mat >= START_MAT_KM. Sensitivity: 0.5 and 1.5 km. Under the strict rule km 0 is
    never a mat.
  * Events without a checkpoint at or beyond the analysis' threshold of the start of the run (every
    mat at the start of the run, so their "first interval" would be the whole run) are excluded from both
    first-interval contrasts (between: they leave the first-interval group and have no remainder;
    within: they leave the event set) and from the pooled hazard by band of interval start (Table S3
    panel A), in the main analysis and in every sensitivity analysis that produces bands; they stay in
    the per-km curve. Their number is persisted per event type and analysis
    (dropout_hazard_contrasts.csv and dropout_hazard_bands.csv, n_events_excluded_no_checkpoint), and
    they are listed for the main analysis (dropout_hazard.txt section 1).
  * Mats are inferred from DNF positions. Two opposite biases follow: a mat where
    no DNF was recorded is invisible (interval widened, hazard there biased DOWN), and
    a mat is visible only if at least one DNF was recorded there (zero-truncation:
    exposure at the start of the run is counted only when a DNF occurred there,
    hazard there biased UP); both biases are declared rather than modelled.
  * Exposure uses the full interval width for every athlete at risk, including
    those whose race ended inside it.
  * The main contrast (a) is reported two ways: BETWEEN (first interval from
    the events in which the start of the run was observed vs the rest of the run from all
    events) and WITHIN (both quantities from the events in which the start of the run was observed).
  * Curve contrasts exclude, for each event, its terminal km bin: the 1-km bin that
    contains its finish (intervals ending at the finish mat). The curve CSV gives the
    pooled hazard over the other cells, with the terminal cells reported apart.
  * Events with fewer than MIN_RUN_DNF run DNFs under the analysis' own rule are excluded from
    mat inference (sensitivity: 20 and 50).

Outputs (notebooks/results/):
  dropout_hazard.txt            narrative + tables
  dropout_hazard_bands.csv      pooled hazard by band x race type (main + sensitivities; events without a
                                checkpoint at or beyond the threshold of the start of the run excluded, as in the
                                first-interval contrasts)
  dropout_hazard_curve.csv      per-km hazard with bootstrap CI x race type (main, strict, t2_no_limit,
                                including_stopped)
  dropout_hazard_contrasts.csv  main contrasts (ratios with CI; main analysis + sensitivities): events in
                                the contrast, with the start of the run observed, in the cohort, and
                                excluded for lacking a checkpoint at or beyond the threshold of the start of the run
  dropout_hazard_events.csv     per-event summary (n DNF, n mats, first checkpoint, run length);
                                main, strict, t2_no_limit and including_stopped analyses (MIN_RUN_DNF,
                                START_MAT_KM), column `analysis`
  dropout_hazard_cohort_seasons.csv  cohort events by season, with the first and last season of the cohort
                                (main analysis and the years_le_2019 / years_ge_2021 sensitivities)
The five CSVs carry the reproducibility header as '#' lines (_common.write_csv, full precision
with float_format %.10g); read them with pd.read_csv(..., comment='#').
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_dataset, event_course_table, attach_course, run_position, t2_limits, reproducibility_header, write_csv,
    RESULTS_DIR, SEED, COURSE_COLUMNS, NOMINAL_RUN_KM, RUN_CLASS_LABELS, RUN_RULES, STOPPED_EDITIONS,
    T2_LIMIT_QUANTILE, UNUSABLE_DISTANCE_BASES,
)

# --- Constants ---
MIN_RUN_DNF = 30          # events with fewer run DNFs are not used for mat inference
MIN_RUN_DNF_SENS = (20, 50)
START_MAT_KM = 1.0        # mats below this are mats at the start of the run (km 0 included)
START_MAT_SENS = (0.5, 1.5)
UNDERPOWERED_START_MATS = 10   # contrasts with fewer events in which the start of the run was observed are flagged
N_BOOT = 1000
# Analyses with their own cohort, curve and within-event contrast: rule of _common.RUN_RULES, stopped editions kept
ANALYSES = {
    "main": ("t2_rule", False),
    "strict": ("strict", False),
    "t2_no_limit": ("t2_no_limit", False),
    "including_stopped": ("t2_rule", True),
}
ANALYSIS_TAGS = {
    "main": " — T2 rule",
    "strict": " — strict rule: no DNF at the cycling-completion distance enters the run (sensitivity)",
    "t2_no_limit": " — T2 rule without the duration limit (sensitivity)",
    "including_stopped": " — T2 rule, stopped or wholesale-coded editions kept (sensitivity)",
}
BANDS = {
    "im": [(0, 1), (1, 3), (3, 5), (5, 10), (10, 21.1), (21.1, np.inf)],
    "him": [(0, 1), (1, 3), (3, 5), (5, 10), (10, np.inf)],
}
COLS = list(dict.fromkeys([
    "finish_status", "race_type", "race_year", "source", "overall_sec", "total_distance_km",
    "event_id", "event_name", "t2_sec",
] + COURSE_COLUMNS))
OUT_TXT = RESULTS_DIR / "dropout_hazard.txt"
OUT_BANDS = "dropout_hazard_bands.csv"           # CSV names under RESULTS_DIR (written by write_csv)
OUT_CURVE = "dropout_hazard_curve.csv"
OUT_CONTRASTS = "dropout_hazard_contrasts.csv"
OUT_EVENTS = "dropout_hazard_events.csv"
OUT_SEASONS = "dropout_hazard_cohort_seasons.csv"
SEASON_ANALYSES = ("main", "years_le_2019", "years_ge_2021")   # cohorts whose seasons are persisted
CSV_FLOAT = "%.10g"


def band_label(lo: float, hi: float) -> str:
    return f"[{lo:g}, {'end' if np.isinf(hi) else f'{hi:g}'})"


def prepare(df: pd.DataFrame, course: pd.DataFrame, rt: str,
            t2_limit: dict[str, float]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict, pd.DataFrame]:
    """Return (run DNFs under each rule of _common.RUN_RULES, completed counts per event, log, in-race DNFs
    with a distance value). The course positions are attached first, so the distances of events with an
    unusable course are missing (_common.attach_course) before the in-race DNFs with a distance are taken."""
    x = attach_course(df[(df["race_type"] == rt) & (df["source"] != "coachcox")].copy(), course)
    has_time = (x["overall_sec"].fillna(0) > 0)   # finish time recorded
    inrace = (x["finish_status"] == "DNF") & ~has_time
    dnf = x[inrace & x["total_distance_km"].notna()].copy()
    dnf = run_position(dnf, course, t2_limit)
    log = {"dnf_with_distance": len(dnf), "dnf_distance_not_used": int((inrace & x["distance_not_used"]).sum()),
           **{k: int((dnf["run_class"] == k).sum()) for k in RUN_CLASS_LABELS}}
    runs = {}
    for rule, classes in RUN_RULES.items():
        r = dnf[dnf["run_class"].isin(classes)].copy()
        r["mat"] = r["km_run"]
        runs[rule] = r
        log[f"run_dnf_{rule}"] = len(r)
    completed = x[(x["finish_status"] == "FIN") | ((x["finish_status"] == "DNF") & has_time)]
    comp = completed.groupby("event_id", observed=True).agg(
        n_fin=("finish_status", lambda s: int((s == "FIN").sum())),
        n_ft=("finish_status", lambda s: int((s == "DNF").sum())),   # finish-time DNFs
    )
    return runs, comp, log, dnf


def event_intervals(run: pd.DataFrame, comp: pd.DataFrame, rt: str,
                    min_dnf: int, include_finish_time: bool = True,
                    years: tuple[int, int] | None = None,
                    start_mat_km: float = START_MAT_KM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the interval table (one row per event x mat) and the per-event summary."""
    rows, ev_rows = [], []
    if years is not None:
        run = run[(run["race_year"] >= years[0]) & (run["race_year"] <= years[1])]
    for ev, g in run.groupby("event_id", observed=True):
        if len(g) < min_dnf or ev not in comp.index:
            continue
        n_completed = int(comp.loc[ev, "n_fin"]) + (int(comp.loc[ev, "n_ft"]) if include_finish_time else 0)
        run_len = float(g["run_len"].iloc[0])
        counts = g["mat"].value_counts().sort_index()
        mats = counts.index.to_numpy(dtype=float)
        d = counts.to_numpy()
        at_risk = n_completed + d[::-1].cumsum()[::-1]      # completed + DNFs at or beyond mat i
        nxt = np.append(mats[1:], run_len)
        width = nxt - mats
        # A mat at the finish would get zero exposure: fail loudly instead of producing an inf/NaN hazard.
        assert (width > 0).all(), f"{rt} event {ev}: run DNF recorded at the finish distance (zero-width interval)"
        for m, nx, w, di, ni in zip(mats, nxt, width, d, at_risk):
            rows.append({"event_id": ev, "mat": m, "next": nx, "width": w, "d": int(di), "at_risk": int(ni),
                         "is_start": bool(m < start_mat_km), "run_len": run_len})
        start_mats = mats[mats < start_mat_km]
        cps = mats[mats >= start_mat_km]
        ev_rows.append({
            "event_id": ev, "event_name": str(g["event_name"].iloc[0]), "race_year": int(g["race_year"].iloc[0]),
            "n_run_dnf": len(g), "n_dnf_km0": int((g["mat"] == 0).sum()), "n_mats": len(mats),
            "n_completed": n_completed, "run_len_km": run_len,
            "has_start_mat": bool(len(start_mats)), "has_mat_at_km0": bool((mats == 0).any()),
            "first_checkpoint_km": float(cps.min()) if len(cps) else np.nan,
            "in_first_interval_contrasts": bool(len(cps)),   # a checkpoint at or beyond start_mat_km
            "share_dnf_before_first_checkpoint": float((g["mat"] < start_mat_km).mean()),
        })
    return pd.DataFrame(rows), pd.DataFrame(ev_rows)


def with_checkpoint(iv: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The intervals of the events with a checkpoint at or beyond the analysis' threshold of the start of the run
    (at least one interval that does not start at the start of the run), and the number of events left out: those
    whose every mat lies at the start of the run, excluded from the first-interval contrasts (bootstrap_ratio) and
    the bands."""
    has_cp = (~iv["is_start"]).groupby(iv["event_id"]).transform("any").to_numpy(dtype=bool)
    return iv[has_cp], int(iv["event_id"].nunique() - iv.loc[has_cp, "event_id"].nunique())


def pooled_by_band(iv: pd.DataFrame, rt: str) -> pd.DataFrame:
    """Pooled hazard by band of interval start over the events with a checkpoint at or beyond the threshold of the
    start of the run (the events of the first-interval contrasts; with_checkpoint)."""
    iv, n_excluded = with_checkpoint(iv)
    out = []
    for lo, hi in BANDS[rt]:
        b = iv[(iv["mat"] >= lo) & (iv["mat"] < hi)]
        if len(b) == 0:
            continue
        exposure = float((b["at_risk"] * b["width"]).sum())
        out.append({
            "band_start_km": band_label(lo, hi), "n_intervals": len(b), "n_events": b["event_id"].nunique(),
            "dnf": int(b["d"].sum()), "athlete_km": exposure,
            "hazard_per_1e4_athlete_km": b["d"].sum() / exposure * 1e4,
            "median_width_km": float(b["width"].median()), "max_width_km": float(b["width"].max()),
            "n_events_excluded_no_checkpoint": n_excluded,
        })
    return pd.DataFrame(out)


def season_rows(ev: pd.DataFrame, rt: str, analysis: str) -> tuple[list[dict], str]:
    """Cohort events by season (all cohort events, and those in the first-interval contrasts), with the first and
    last season of the cohort, and a text line."""
    g = ev.groupby("race_year").agg(n_events_cohort=("event_id", "size"),
                                    n_events_in_first_interval_contrasts=("in_first_interval_contrasts", "sum"))
    first, last = int(g.index.min()), int(g.index.max())
    rows = [{"race_type": rt, "analysis": analysis, "race_year": int(y), "n_events_cohort": int(r.n_events_cohort),
             "n_events_in_first_interval_contrasts": int(r.n_events_in_first_interval_contrasts),
             "first_season": first, "last_season": last} for y, r in g.iterrows()]
    line = (f"cohort seasons ({analysis}): first {first}, last {last}; cohort events by season: "
            + ", ".join(f"{int(y)}: {int(n)}" for y, n in g["n_events_cohort"].items()))
    return rows, line


def per_event_km_arrays(iv: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per event: numerator and exposure spread uniformly over 1-km bins, and the terminal-cell mask.

    Returns (num[e, k], exp[e, k], term[e, k]) with k = 0..K-1, K = ceil(longest run length); each
    event's intervals end at its own run length. Events are spread at density d / width, so the
    numerator in km k = d / width * overlap; exposure = at_risk * overlap. term marks, for each event,
    the 1-km bin that contains its finish (its terminal bin).
    """
    events = iv["event_id"].unique()
    idx = {e: i for i, e in enumerate(events)}
    run_len = iv.groupby("event_id")["run_len"].first().reindex(events).to_numpy(dtype=float)
    n_bins = int(np.ceil(run_len.max() - 1e-9))
    num = np.zeros((len(events), n_bins))
    exp_ = np.zeros((len(events), n_bins))
    edges = np.arange(n_bins + 1, dtype=float)
    for r in iv.itertuples(index=False):
        lo, hi = r.mat, r.next
        overlap = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None)
        e = idx[r.event_id]
        num[e] += r.d / r.width * overlap
        exp_[e] += r.at_risk * overlap
    term = np.zeros_like(num, dtype=bool)
    term[np.arange(len(events)), np.ceil(run_len - 1e-9).astype(int) - 1] = True
    return num, exp_, term


def bootstrap_band_ratio(num: np.ndarray, exp_: np.ndarray, a: int, b: int, c: int, d: int | None) -> tuple[float, float, int]:
    """Cluster-bootstrap CI for hazard(km a..b) / hazard(km c..d) from per-event km arrays (terminal cells zeroed)."""
    rng = np.random.default_rng(SEED)
    n_ev = num.shape[0]
    ratios = np.empty(N_BOOT)
    for r in range(N_BOOT):
        s = rng.integers(0, n_ev, n_ev)
        ns, xs = num[s], exp_[s]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[r] = (ns[:, a:b].sum() / xs[:, a:b].sum()) / (ns[:, c:d].sum() / xs[:, c:d].sum())
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return lo, hi, int(np.isnan(ratios).sum())


def bootstrap_curve(num: np.ndarray, exp_: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pooled per-km hazard with cluster-bootstrap percentile CI (events resampled; terminal cells zeroed).

    Returns (point, ci_low, ci_high, n_reps_with_any_nan); NaN counted only in bins with exposure in
    the full sample (a bin is NaN in a replicate only if the resample has zero exposure there)."""
    rng = np.random.default_rng(SEED)
    with np.errstate(divide="ignore", invalid="ignore"):
        point = num.sum(0) / exp_.sum(0)
    n_ev = num.shape[0]
    boots = np.empty((N_BOOT, num.shape[1]))
    for b in range(N_BOOT):
        s = rng.integers(0, n_ev, n_ev)
        with np.errstate(divide="ignore", invalid="ignore"):
            boots[b] = num[s].sum(0) / exp_[s].sum(0)
    has_exp = exp_.sum(0) > 0
    n_nan = int(np.isnan(boots[:, has_exp]).any(axis=1).sum())
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi, n_nan


def bootstrap_ratio(iv: pd.DataFrame, within: bool) -> dict:
    """Hazard in the intervals that start at the start of the run vs the rest of the run; cluster bootstrap CI.

    Events without a checkpoint at or beyond the threshold of the start of the run (every interval starts at
    the start of the run: no remainder, and their first interval would be the whole run) are excluded first.
    within=False: first-interval hazard from the remaining events in which the start of the run was observed,
    rest-of-run hazard from ALL remaining events. within=True: both from the remaining events in which the
    start of the run was observed.
    """
    rng = np.random.default_rng(SEED)
    fm = iv["is_start"].to_numpy()
    d = iv["d"].to_numpy(dtype=float)
    ex = (iv["at_risk"] * iv["width"]).to_numpy(dtype=float)
    df_e = pd.DataFrame({"e": iv["event_id"].to_numpy(), "d1": d * fm, "x1": ex * fm,
                         "d0": d * ~fm, "x0": ex * ~fm}).groupby("e").sum()
    n_cohort = len(df_e)
    df_e = df_e[df_e["x0"] > 0]   # every interval has exposure > 0, so x0 > 0 <=> a checkpoint at or beyond the threshold
    n_excluded = n_cohort - len(df_e)
    if within:
        df_e = df_e[df_e["x1"] > 0]
    arr = df_e.to_numpy()
    tot = arr.sum(0)
    h1, h0 = tot[0] / tot[1], tot[2] / tot[3]
    ratios = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, len(arr), len(arr))
        a = arr[s].sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[b] = (a[0] / a[1]) / (a[2] / a[3])
    n_nan = int(np.isnan(ratios).sum())
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    n_start = int((df_e["x1"] > 0).sum())
    return {"n_events": len(arr), "n_events_with_start_mat": n_start,
            "h_first_per_1e4": h1 * 1e4, "h_rest_per_1e4": h0 * 1e4,
            "ratio": h1 / h0, "ci_low": lo, "ci_high": hi, "n_nan_reps": n_nan,
            "underpowered": n_start < UNDERPOWERED_START_MATS,
            "n_events_cohort": n_cohort, "n_events_excluded_no_checkpoint": n_excluded}


def fmt_contrast(c: dict) -> str:
    return (f"{c['h_first_per_1e4']:.2f} vs {c['h_rest_per_1e4']:.2f} per 10,000 athlete-km; "
            f"ratio {c['ratio']:.2f} (95% CI {c['ci_low']:.2f}-{c['ci_high']:.2f}; "
            f"events {c['n_events']}, with the start of the run observed {c['n_events_with_start_mat']}; cohort "
            f"{c['n_events_cohort']}, excluded without a checkpoint at or beyond the threshold of the start of the run "
            f"{c['n_events_excluded_no_checkpoint']}; "
            f"NaN reps {c['n_nan_reps']}{'; UNDERPOWERED' if c['underpowered'] else ''})")


def curve_block(iv: pd.DataFrame, rt: str, analysis: str) -> tuple[pd.DataFrame, dict, list[str]]:
    """Per-km curve (terminal cells excluded), its contrast km 0-3 vs km 3-end, and text lines."""
    num, exp_, term = per_event_km_arrays(iv)
    num_nt, exp_nt = np.where(term, 0.0, num), np.where(term, 0.0, exp_)
    point, clo, chi, n_nan_curve = bootstrap_curve(num_nt, exp_nt)
    k = np.arange(len(point))
    exp_t, num_t = (exp_ * term).sum(0), (num * term).sum(0)
    tot = exp_nt.sum(0) + exp_t
    with np.errstate(divide="ignore", invalid="ignore"):
        tshare = np.where(tot > 0, exp_t / tot, np.nan)
    curve = pd.DataFrame({"race_type": rt, "analysis": analysis, "km_bin_start": k, "km_bin_end": k + 1,
                          "terminal_bin": np.nan_to_num(tshare) > 0.5,
                          "hazard_per_1e4": point * 1e4, "ci_low": clo * 1e4, "ci_high": chi * 1e4,
                          "dnf_spread": num_nt.sum(0), "athlete_km": exp_nt.sum(0),
                          "n_events": (exp_nt > 0).sum(0),
                          "dnf_spread_terminal": num_t, "athlete_km_terminal": exp_t, "terminal_share": tshare})
    h = lambda a, b: num_nt[:, a:b].sum() / exp_nt[:, a:b].sum() * 1e4
    h03, h3e = h(0, 3), h(3, None)
    clo_r, chi_r, nan_r = bootstrap_band_ratio(num_nt, exp_nt, 0, 3, 3, None)
    contrast = {"race_type": rt, "analysis": analysis, "contrast": "curve_km0-3_vs_km3-end_excl_terminal",
                "n_events": num.shape[0], "n_events_with_start_mat": np.nan,
                "h_first_per_1e4": h03, "h_rest_per_1e4": h3e,
                "ratio": h03 / h3e, "ci_low": clo_r, "ci_high": chi_r, "n_nan_reps": nan_r, "underpowered": False,
                "n_events_cohort": num.shape[0], "n_events_excluded_no_checkpoint": 0}   # the curve keeps every cohort event
    term_h = num_t.sum() / exp_t.sum() * 1e4
    lines = [f"  (b) per-km curve: km 0-3 = {h03:.2f}; km 3-end = {h3e:.2f} per 10,000 athlete-km; "
             f"ratio {h03 / h3e:.2f} (95% CI {clo_r:.2f}-{chi_r:.2f}; each event's terminal bin excluded; "
             f"NaN reps: curve {n_nan_curve}, ratio {nan_r})"]
    if rt == "im":
        lines.append(f"      km 3-21 = {h(3, 21):.2f}; km 21-end = {h(21, None):.2f}; terminal bins (pooled) = {term_h:.2f}")
    else:
        lines.append(f"      terminal bins (pooled) = {term_h:.2f}")
    ti = np.ceil(iv.groupby("event_id")["run_len"].first() - 1e-9).astype(int) - 1
    lines.append("      terminal bin of each event (bin start km: events): " + ", ".join(f"{a}: {b}" for a, b in ti.value_counts().sort_index().items()))
    return curve, contrast, lines


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset...")
    df_all = load_dataset(COLS, keep_stopped=True)      # stopped editions kept: sensitivity "including_stopped"
    is_stopped = df_all["event_id"].astype("string").isin(list(STOPPED_EDITIONS)).fillna(False).to_numpy()
    df = df_all[~is_stopped].copy()                     # the analysed population (as in every other script)
    course = event_course_table(df)
    course_all = event_course_table(df_all)
    t2_lim = t2_limits(df)       # T2 duration limits of the analysed population (also applied with the stopped editions kept)
    for frame in (df, df_all):
        frame["race_type"] = frame["race_type"].astype(str)
        frame["source"] = frame["source"].astype(str)
        frame["finish_status"] = frame["finish_status"].astype(str)

    out = [reproducibility_header("RQ2 — Interval-censored dropout hazard along the run")]
    out.append(f"Constants: MIN_RUN_DNF={MIN_RUN_DNF} (sens {MIN_RUN_DNF_SENS}), START_MAT_KM={START_MAT_KM} "
               f"(sens {START_MAT_SENS}), N_BOOT={N_BOOT}; positions compared at two decimals; course positions per event "
               f"(modal finisher distances; nominal run length IM {NOMINAL_RUN_KM['im']:.2f}, HIM {NOMINAL_RUN_KM['him']:.2f} km "
               f"where the finishers carry no distance)")
    out.append(f"T2 duration limit (_common.t2_limits; {T2_LIMIT_QUANTILE * 100:.0f}th percentile of the T2 splits of the "
               f"official-source finishers with a T2 split > 0, analysed population): IM {t2_lim['im']:,.2f} s = "
               f"{t2_lim['im'] / 60:.2f} min; HIM {t2_lim['him']:,.2f} s = {t2_lim['him'] / 60:.2f} min")
    out.append("Official source only (total_distance_km is official-only). Events = in-race DNFs (DNFs without a")
    out.append("recorded finish time). Finish-time DNFs (DNFs with a recorded finish time, overall_sec > 0) are")
    out.append("non-events at risk over the whole run, like finishers. DQ excluded. Exposure starts at the first")
    out.append("observed run mat. km_run = total_distance_km - the event's cycling completion, at two decimals.")
    out.append("The recorded distances of the events whose course basis is " + " or ".join(UNUSABLE_DISTANCE_BASES))
    out.append("are not used (_common.attach_course): their DNFs count as DNFs without a distance.")
    out.append("Main analysis (T2 rule): run DNFs = DNFs beyond the cycling completion and before the finish")
    out.append("distance (at their km_run) + DNFs at the cycling-completion distance with a T2 split of at most the T2")
    out.append("duration limit of their event type (at km 0, the transition exit). Sensitivities: strict rule, the former")
    out.append("only; T2 rule without the duration limit, any T2 split > 0. Identical definition in dropout_point.py")
    out.append("section 5. Mats at the start of the run: mats < START_MAT_KM, km 0 included (observed whenever at least")
    out.append("one DNF with a T2 split within the limit withdrew there). Events without a checkpoint at or beyond the")
    out.append("threshold of the start of the run are excluded from both first-interval contrasts and from the pooled")
    out.append("hazard by band, and kept in the per-km curve. Every bootstrap uses a fresh generator seeded with SEED.")
    out.append("Curve contrasts exclude each event's terminal km bin (the bin that contains its finish).")
    out.append(f"Sensitivity 'including_stopped': the main analysis with the {len(STOPPED_EDITIONS)} stopped or wholesale-coded editions kept")
    out.append("(_common.STOPPED_EDITIONS; excluded from every other analysis).")

    bands_all, curve_all, contrasts_all, events_all, seasons_all = [], [], [], [], []
    for rt in ["im", "him"]:
        label = "IRONMAN" if rt == "im" else "IRONMAN 70.3"
        runs, comp, log, dnf = prepare(df, course, rt, t2_lim)
        runs_incl, comp_incl, log_incl, _ = prepare(df_all, course_all, rt, t2_lim)
        out.append("\n" + "=" * 70)
        out.append(f"{label}")
        out.append("=" * 70)
        out.append(f"In-race DNFs with distance data: {log['dnf_with_distance']:,}")
        out.append(f"  (in-race DNFs whose recorded distance is not used, course rule: {log['dnf_distance_not_used']:,})")
        out.append(f"  before the cycling completion (no mat crossed, swim or cycling segment; not modelled): {log['before']:,}")
        out.append(f"  at the cycling-completion distance: {log['at_t2'] + log['at_t2_long'] + log['at_no_t2']:,}")
        out.append(f"    with a T2 split within the duration limit (T2-completers, enter the run at km 0): {log['at_t2']:,}")
        out.append(f"    with a T2 split longer than the limit (withdrew in the transition; not modelled in the main analysis): "
                   f"{log['at_t2_long']:,}")
        out.append(f"    without a T2 split (withdrew in the transition; not modelled): {log['at_no_t2']:,}")
        out.append(f"  beyond the cycling completion and before the finish distance: {log['run']:,}")
        out.append(f"  at the finish distance (crossed the finish mat without a finish time; excluded): {log['at_finish']:,}")
        out.append(f"  beyond the run length (data anomalies, excluded): {log['beyond']:,}")
        out.append(f"  run-segment DNFs, main analysis (T2 rule): {log['run_dnf_t2_rule']:,}; strict rule: {log['run_dnf_strict']:,}; "
                   f"T2 rule without the duration limit: {log['run_dnf_t2_no_limit']:,}")
        t2ev = dnf[dnf["run_class"] == "at_t2"].groupby("event_id", observed=True).agg(
            name=("event_name", "first"), n=("run_class", "size")).sort_values("n", ascending=False)
        out.append(f"  T2-completers at km 0 by event: {len(t2ev):,} events; largest: "
                   + "; ".join(f"{r.name} {r.n:,}" for r in t2ev.head(5).itertuples(index=False)))

        runs_by_analysis = {a: ((runs_incl if incl else runs)[rule], comp_incl if incl else comp)
                            for a, (rule, incl) in ANALYSES.items()}
        run_dnf_total = {a: (log_incl if incl else log)[f"run_dnf_{rule}"] for a, (rule, incl) in ANALYSES.items()}
        results = {}
        for analysis, (rset, cset) in runs_by_analysis.items():
            iv, ev = event_intervals(rset, cset, rt, MIN_RUN_DNF)
            ev["race_type"] = rt
            ev["analysis"] = analysis
            events_all.append(ev)
            results[analysis] = (iv, ev)
        iv, ev = results["main"]

        # --- Section 1: checkpoint structure per event ---
        cov = iv["d"].sum() / log["run_dnf_t2_rule"] * 100
        out.append(f"\nSECTION 1 — Checkpoint structure (events with >= {MIN_RUN_DNF} run DNFs; main analysis)")
        out.append(f"  events in cohort: {len(ev):,}  | run DNFs covered: {iv['d'].sum():,} ({cov:.4f}% of run DNFs; "
                   f"{iv['d'].sum():,} / {log['run_dnf_t2_rule']:,})")
        rows, line = season_rows(ev, rt, "main")
        seasons_all += rows
        out.append(f"  {line}")
        out.append(f"  T2-completers at km 0 in the cohort: {int(ev['n_dnf_km0'].sum()):,} of {log['at_t2']:,}; "
                   f"events with a mat at km 0: {int(ev['has_mat_at_km0'].sum()):,}")
        out.append(f"  distinct run checkpoints per event: median {ev['n_mats'].median():.0f} "
                   f"(P10 {ev['n_mats'].quantile(.1):.0f}, P90 {ev['n_mats'].quantile(.9):.0f})")
        out.append(f"  DNFs per interval: median {iv['d'].median():.0f} (P25 {iv['d'].quantile(.25):.0f}, P75 {iv['d'].quantile(.75):.0f})")
        out.append(f"  events with a mat at the start of the run (< {START_MAT_KM} km): {ev['has_start_mat'].mean()*100:.4f}% "
                   f"({int(ev['has_start_mat'].sum()):,} / {len(ev):,}); with a mat at the start of the run other than km 0: "
                   f"{int(((iv['is_start']) & (iv['mat'] > 0)).groupby(iv['event_id']).any().sum()):,}")
        fc = ev["first_checkpoint_km"].dropna()
        out.append(f"  first run checkpoint (km past the event's cycling completion): median {fc.median():.3f} "
                   f"(P10 {fc.quantile(.1):.3f}, P25 {fc.quantile(.25):.3f}, P75 {fc.quantile(.75):.3f}, "
                   f"P90 {fc.quantile(.9):.3f}, min {fc.min():.3f}, max {fc.max():.3f}), n = {len(fc)} "
                   f"(events without any mat >= {START_MAT_KM} km: {ev['first_checkpoint_km'].isna().sum()})")
        nocp = ev[~ev["in_first_interval_contrasts"]].sort_values(["race_year", "event_name"])
        out.append(f"  events without a checkpoint at or beyond {START_MAT_KM} km (excluded from both first-interval contrasts, "
                   f"kept in the per-km curve): {len(nocp)}")
        for r in nocp.itertuples(index=False):
            out.append(f"    {r.event_name} [{r.event_id}]: run DNFs {r.n_run_dnf:,} (at km 0 {r.n_dnf_km0:,}), mats {r.n_mats}, "
                       f"run length {r.run_len_km:.2f} km")
        sh = ev["share_dnf_before_first_checkpoint"]
        out.append(f"  share of an event's run DNFs recorded at a mat at the start of the run (before the first checkpoint): "
                   f"median {sh.median()*100:.1f}%, pooled {iv.loc[iv['is_start'], 'd'].sum() / iv['d'].sum() * 100:.1f}%")
        out.append(f"  run length of the cohort events (km): median {ev['run_len_km'].median():.4f} "
                   f"(min {ev['run_len_km'].min():.4f}, max {ev['run_len_km'].max():.4f})")
        for analysis, lab in [("strict", "strict rule"), ("t2_no_limit", "T2 rule without the duration limit"),
                              ("including_stopped", "including stopped editions")]:
            ivs, evs = results[analysis]
            fcs = evs["first_checkpoint_km"].dropna()
            extra = (f"run DNFs {run_dnf_total[analysis]:,} (T2-completers at km 0 {log_incl['at_t2']:,}); "
                     if analysis == "including_stopped" else "")
            out.append(f"  {lab}: {extra}cohort {len(evs):,} events, run DNFs covered {ivs['d'].sum():,} "
                       f"({ivs['d'].sum() / run_dnf_total[analysis] * 100:.4f}% of {run_dnf_total[analysis]:,}); "
                       f"DNFs at km 0 in the cohort {int(evs['n_dnf_km0'].sum()):,}; events with the start of the run observed "
                       f"{int(evs['has_start_mat'].sum()):,}; without a checkpoint at or beyond {START_MAT_KM} km "
                       f"{int((~evs['in_first_interval_contrasts']).sum())}; median first checkpoint {fcs.median():.3f} km")

        # --- Section 2: pooled hazard by band ---
        bands = pooled_by_band(iv, rt)
        n_exc = int(bands["n_events_excluded_no_checkpoint"].iloc[0])
        out.append(f"\nSECTION 2 — Pooled hazard by band of interval start (DNF per 10,000 athlete-km; main analysis; events "
                   f"without a checkpoint at or beyond {START_MAT_KM} km excluded, as in the first-interval contrasts: {n_exc})")
        bands.insert(0, "race_type", rt)
        bands.insert(1, "analysis", "main")
        bands_all.append(bands)
        out.append(bands.drop(columns=["race_type", "analysis", "n_events_excluded_no_checkpoint"]).round(2)
                   .to_string(index=False))

        # --- Section 3: main contrasts ---
        out.append(f"\nSECTION 3 — Main contrasts (cluster bootstrap over events, {N_BOOT} reps, seed {SEED})")
        for analysis in ANALYSES:
            iv_a, _ = results[analysis]
            out.append(f"  [{analysis}{ANALYSIS_TAGS[analysis]}]")
            for within, tag in [(False, "between"), (True, "within")]:
                c = bootstrap_ratio(iv_a, within)
                contrasts_all.append({"race_type": rt, "analysis": analysis, "contrast": f"first_interval_vs_rest_{tag}", **c})
                out.append(f"  (a-{tag}) start of the run -> first checkpoint vs rest of run: {fmt_contrast(c)}")
            curve, contrast, lines = curve_block(iv_a, rt, analysis)
            contrasts_all.append(contrast)
            curve_all.append(curve)
            out += lines
            if analysis == "main":
                out.append("  per-km curve (hazard per 10,000 athlete-km over each event's non-terminal bins, 95% CI;")
                out.append("  n_events = events with exposure in the bin; terminal_share = share of the bin's exposure in terminal bins):")
                out.append("  " + curve[["km_bin_start", "hazard_per_1e4", "ci_low", "ci_high", "n_events", "terminal_share"]].round(2)
                           .to_string(index=False).replace("\n", "\n  "))

        # --- Section 4: sensitivities ---
        out.append(f"\nSECTION 4 — Sensitivity analyses (first interval vs rest, BETWEEN version; dropout_hazard_bands.csv "
                   f"holds their bands, without the events lacking a checkpoint at or beyond {START_MAT_KM} km)")
        sens = {
            f"min_dnf_{MIN_RUN_DNF_SENS[0]}": dict(min_dnf=MIN_RUN_DNF_SENS[0]),
            f"min_dnf_{MIN_RUN_DNF_SENS[1]}": dict(min_dnf=MIN_RUN_DNF_SENS[1]),
            "exclude_2026": dict(min_dnf=MIN_RUN_DNF, years=(2002, 2025)),
            "years_le_2019": dict(min_dnf=MIN_RUN_DNF, years=(2002, 2019)),
            "years_ge_2021": dict(min_dnf=MIN_RUN_DNF, years=(2021, 2026)),
            "at_risk_without_finish_time": dict(min_dnf=MIN_RUN_DNF, include_finish_time=False),
            f"start_mat_{START_MAT_SENS[0]}": dict(min_dnf=MIN_RUN_DNF, start_mat_km=START_MAT_SENS[0]),
            f"start_mat_{START_MAT_SENS[1]}": dict(min_dnf=MIN_RUN_DNF, start_mat_km=START_MAT_SENS[1]),
            "strict": dict(min_dnf=MIN_RUN_DNF),
            "t2_no_limit": dict(min_dnf=MIN_RUN_DNF),
            "including_stopped": dict(min_dnf=MIN_RUN_DNF),
        }
        for name, kw in sens.items():
            rset, cset = runs_by_analysis.get(name, runs_by_analysis["main"])
            iv_s, ev_s = event_intervals(rset, cset, rt, **kw)
            if len(iv_s) == 0:
                out.append(f"  {name}: no events")
                continue
            if not name.startswith("start_mat_"):   # the sensitivities of the start of the run move the contrasts only: no bands
                b = pooled_by_band(iv_s, rt)
                b.insert(0, "race_type", rt); b.insert(1, "analysis", name)
                bands_all.append(b)
            if name in ANALYSES:   # contrasts already in section 3
                c = next(x for x in contrasts_all if x["race_type"] == rt and x["analysis"] == name
                         and x["contrast"] == "first_interval_vs_rest_between")
            else:
                c = bootstrap_ratio(iv_s, within=False)
                contrasts_all.append({"race_type": rt, "analysis": name, "contrast": "first_interval_vs_rest_between", **c})
            out.append(f"  {name}: {fmt_contrast(c)}")
            if name in SEASON_ANALYSES:
                rows, line = season_rows(ev_s, rt, name)
                seasons_all += rows
                out.append(f"    {line}")

    # --- Save ---
    write_csv(pd.concat(bands_all), OUT_BANDS, "RQ2 — pooled dropout hazard by band of interval start (main + sensitivities; "
              "events without a checkpoint at or beyond the threshold of the start of the run excluded, as in the first-interval "
              "contrasts: n_events_excluded_no_checkpoint)", float_format=CSV_FLOAT)
    write_csv(pd.DataFrame(seasons_all), OUT_SEASONS, "RQ2 — seasons of the hazard cohort events (main analysis and the "
              "years_le_2019 / years_ge_2021 sensitivities; first and last season of each cohort)", float_format=CSV_FLOAT)
    write_csv(pd.concat(curve_all), OUT_CURVE, "RQ2 — per-km dropout hazard with cluster-bootstrap CI (main, strict, t2_no_limit, "
              "including_stopped; terminal bins apart)", float_format=CSV_FLOAT)
    write_csv(pd.DataFrame(contrasts_all), OUT_CONTRASTS, "RQ2 — main contrasts of the dropout hazard (main + sensitivities)",
              float_format=CSV_FLOAT)
    write_csv(pd.concat(events_all).sort_values(["analysis", "race_type", "race_year", "event_name"]), OUT_EVENTS,
              "RQ2 — per-event summary of the hazard cohort (main, strict, t2_no_limit and including_stopped analyses)",
              float_format=CSV_FLOAT)
    text = "\n".join(out)
    OUT_TXT.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print(f"\nSaved: {OUT_TXT}")


if __name__ == "__main__":
    main()
