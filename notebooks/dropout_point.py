"""
RQ2: Dropout point analysis — in which segment does the race of a DNF end?
Dual method: (1) total_distance_km (in-race DNFs), (2) split inference, and their cross-tabulation.
Also characterises the two DNF groups named operationally in the manuscript:
  - finish-time DNFs: DNFs with a recorded overall finish time (overall_sec > 0)
  - in-race DNFs: DNFs without a recorded finish time (overall_sec missing or 0)
Distances are compared with each event's own course positions (swim exit, cycling completion,
finish: modal finisher distances; _common.event_course_table, nominal fallback) at two decimals.
The recorded distances of events whose finishers' distances do not form a usable course (basis
nominal_incoherent or nominal_scale) are not used: _common.attach_course sets them to missing, so
every distance-based statistic below treats those records as records without a distance (section 3
lists the events and keeps the coverage of the field before the rule).
A DNF at the cycling-completion distance enters the run at km 0 under the T2 rule only when its T2
split is at most the T2 duration limit of its event type (_common.t2_limits; section 5).
Produces: results/dropout_point.txt, dropout_point_segments.csv (section 3 shares: distance classification and,
as variant 't2_rule', the segments of the hazard's run-DNF definition), event_course.csv, dropout_point_t2.csv
(T2 duration limit and the DNFs at the cycling-completion distance by T2 split), dropout_point_by_season.csv
(by season and event type: share of official-source in-race DNFs carrying a distance value as recorded, section 3;
share of the run-segment DNFs of the main rule recorded at km 0, section 5; numerators and denominators)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_dataset, infer_gender, infer_last_completed_segment, infer_dropout_from_distance,
    event_course_table, attach_course, run_position, t2_limits, km_past, reproducibility_header, write_csv,
    RESULTS_DIR, COURSE_COLUMNS, NOMINAL_COURSE, NO_MAT_LABEL, COURSE_BASIS_LABELS, RUN_CLASS_LABELS, RUN_RULES,
    UNUSABLE_DISTANCE_BASES, T2_LIMIT_QUANTILE,
)

TYPICAL_CUTOFF_H = {"im": 17.0, "him": 8.5}   # typical cut-off (hours); Figure 4 reference lines
RANK_PLACEHOLDER = 99999                      # rank_overall value of unranked records
TOP_EDITION_SHARE = 0.10                      # concentration: top 10% of editions by count (floor)
IM_AXIS_MAX_H = 20.0                          # Figure 4a x-axis limit (IRONMAN)
SHORT_TIME_H = {"im": 4.5, "him": 2.0}         # Figure 4 left tail: overall times below these are counted
FIG4_BIN_MIN = {"im": 15, "him": 10}          # Figure 4 bin widths (min), edges on whole minutes from 0 h (generate_figures.FIG4_AXIS)
SEGMENTS = [NO_MAT_LABEL, "swim", "bike", "run"]
AT_FINISH_LABEL, BEYOND_LABEL = "at the finish distance", "beyond the finish distance"
T2_RULE_SEGMENTS = SEGMENTS + [AT_FINISH_LABEL, BEYOND_LABEL]   # variant 't2_rule' of dropout_point_segments.csv
WITHIN_KM = 3.0                               # finish-time DNFs: |recorded distance - finish| <= 3 km
CSV_FLOAT = "%.10g"


def t2_rule_segment(d: pd.DataFrame) -> pd.Series:
    """Segment under the T2 rule, from run_class exactly as the hazard's run DNFs (RUN_RULES['t2_rule']: 'at_t2'
    at km 0 + 'run'): 'run' for those; the finish-distance classes apart ('at_finish', 'beyond'); otherwise the
    distance classification (no timing mat crossed / swim / bike), so a DNF at the cycling-completion distance that
    does not enter the run keeps the segment the distance classification gives it."""
    rc = d["run_class"].astype(str)
    return pd.Series(np.select([rc.isin(RUN_RULES["t2_rule"]).to_numpy(), (rc == "at_finish").to_numpy(),
                                (rc == "beyond").to_numpy()], ["run", AT_FINISH_LABEL, BEYOND_LABEL],
                               d["dropout_distance"].astype(str).to_numpy()), index=d.index)


def season_table(frame: pd.DataFrame, num: str, den: str, pct: str) -> str:
    """Text table by season: numerator / denominator = share (%) for IM, HIM and both event types."""
    lines = [f"  {'season':6s} {'IM':>24s} {'HIM':>24s} {'both':>24s}"]
    for y, g in frame.groupby("race_year"):
        cells = []
        for rt in ("im", "him", "all"):
            r = g[g["race_type"] == rt]
            cells.append("—" if r.empty or int(r[den].iloc[0]) == 0 else
                         f"{int(r[num].iloc[0]):,} / {int(r[den].iloc[0]):,} = {r[pct].iloc[0]:.2f}%")
        lines.append(f"  {int(y):<6d} " + " ".join(f"{c:>24s}" for c in cells))
    return "\n".join(lines)


def by_season(d: pd.DataFrame, num: pd.Series, den: pd.Series, names: tuple[str, str, str]) -> pd.DataFrame:
    """Counts by season x event type, plus both event types ('all'): names = (numerator, denominator, share %)."""
    x = pd.DataFrame({"race_year": d["race_year"].astype(int).to_numpy(), "race_type": d["race_type"].astype(str).to_numpy(),
                      "num": num.to_numpy(dtype=int), "den": den.to_numpy(dtype=int)})
    t = pd.concat([x.groupby(["race_year", "race_type"])[["num", "den"]].sum().reset_index(),
                   x.groupby("race_year")[["num", "den"]].sum().reset_index().assign(race_type="all")])
    t[names[2]] = np.where(t["den"] > 0, 100 * t["num"] / t["den"].where(t["den"] > 0), np.nan)
    return t.rename(columns={"num": names[0], "den": names[1]})


def ratio(num: float, den: float) -> str:
    """'num / den = share%' with the share at four decimals."""
    return f"{int(num):,} / {int(den):,} = {num / den * 100:.4f}%"


def course_summary(course: pd.DataFrame, dnf_events: set) -> list[str]:
    """How the course positions were obtained and how many events deviate from the nominal ones."""
    out = ["\nCourse positions of the official-source events (modal finisher distances; _common.event_course_table):"]
    for rt in ["im", "him"]:
        c = course[course["race_type"] == rt]
        out.append(f"  {rt.upper()}: {len(c):,} events with finishers; basis: "
                   + ", ".join(f"{b} {int((c['basis'] == b).sum()):,}" for b in COURSE_BASIS_LABELS))
    for b, lab in COURSE_BASIS_LABELS.items():
        out.append(f"    {b}: {lab}")
    for scope, sel in [("all events with finishers", None), ("events with in-race DNFs carrying a distance", dnf_events)]:
        out.append(f"  Deviation from the nominal positions at two decimals ({scope}):")
        for rt in ["im", "him"]:
            c = course[course["race_type"] == rt]
            if sel is not None:
                c = c[c.index.isin(sel)]
            own = c[c["basis"].isin(["own", "own_nominal_run"])]
            nom = NOMINAL_COURSE[rt]
            dev = {k: int((own[k] - nom[k]).round(2).ne(0).sum()) for k in ("swim_end", "bike_end", "finish")}
            out.append(f"    {rt.upper()}: events {len(c):,}, with own positions {len(own):,}; deviating from nominal in any position "
                       f"{int(c['deviates_from_nominal'].sum()):,} (swim exit {dev['swim_end']:,}, cycling completion "
                       f"{dev['bike_end']:,}, finish {dev['finish']:,})")
    out.append("  Most common own positions (km; number of events with own positions):")
    for rt in ["im", "him"]:
        own = course[(course["race_type"] == rt) & course["basis"].isin(["own", "own_nominal_run"])]
        for k, lab in (("swim_end", "swim exit"), ("bike_end", "cycling completion"), ("finish", "finish"), ("run_len", "run length")):
            vc = own[k].round(4).value_counts().head(5)
            out.append(f"    {rt.upper()} {lab}: " + ", ".join(f"{f'{v:.4f}'.rstrip('0').rstrip('.')} ({n:,})" for v, n in vc.items())
                       + f"; nominal {(NOMINAL_COURSE[rt]['finish'] - NOMINAL_COURSE[rt]['bike_end']) if k == 'run_len' else NOMINAL_COURSE[rt][k]:.2f}")
    return out


def main():
    cols = list(dict.fromkeys([
        "finish_status", "race_type", "source", "age_group", "event_id", "event_name", "race_year",
        "swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec", "overall_sec",
        "total_distance_km", "rank_overall",
    ] + COURSE_COLUMNS))
    print("Loading dataset...")
    allrec = load_dataset(cols)
    course = event_course_table(allrec)
    t2_lim = t2_limits(allrec)   # T2 duration limit per event type (official-source finishers' T2 splits)
    fin_off = allrec[((allrec["source"] == "official") & (allrec["finish_status"] == "FIN")
                      & (allrec["t2_sec"].fillna(0) > 0)).to_numpy()]
    fin_t2 = {rt: fin_off.loc[fin_off["race_type"] == rt, "t2_sec"].astype("float64") for rt in ["im", "him"]}
    event_names = allrec[["event_id", "event_name"]].dropna().drop_duplicates("event_id")
    event_names = pd.Series(event_names["event_name"].astype(str).to_numpy(), index=event_names["event_id"].astype(str).to_numpy())
    df = allrec[allrec["finish_status"] == "DNF"].copy()
    del allrec, fin_off
    df = infer_gender(df)
    df["has_recorded_distance"] = df["total_distance_km"].notna()   # before the course rule (raw coverage, section 3)
    df = attach_course(df, course)   # course positions; distances of events with an unusable course set to missing

    out = []
    out.append(reproducibility_header("RQ2 — Dropout Point Analysis"))

    # --- Finish-time vs in-race DNFs ---
    out.append("=" * 60)
    out.append("SECTION 1: FINISH-TIME vs IN-RACE DNFs")
    out.append("=" * 60)
    out.append("\nFinish-time DNF: DNF with a recorded overall finish time (overall_sec > 0).")
    out.append("In-race DNF: DNF without a recorded finish time (overall_sec missing or 0).")

    has_time = df["overall_sec"].notna() & (df["overall_sec"] > 0)
    ft = df[has_time].copy()
    inr = df[~has_time]
    out.append(f"\nFinish-time DNFs: {ratio(len(ft), len(df))}")
    out.append(f"In-race DNFs: {ratio(len(inr), len(df))}")

    for rt in ["im", "him"]:
        sub_ft = ft[ft["race_type"] == rt]
        sub_in = inr[inr["race_type"] == rt]
        sub_all = df[df["race_type"] == rt]
        out.append(f"\n{rt.upper()}: finish-time {ratio(len(sub_ft), len(sub_all))}; "
                   f"in-race {ratio(len(sub_in), len(sub_all))}")
        if len(sub_ft) > 0:
            hrs = sub_ft["overall_sec"] / 3600
            out.append(f"  Finish-time DNF overall time (hours): P25={hrs.quantile(.25):.3f}, P50={hrs.quantile(.50):.3f}, "
                       f"P75={hrs.quantile(.75):.3f}, P90={hrs.quantile(.90):.3f}")

    # Sex breakdown
    ft_g = ft.groupby(["race_type", "gender"], observed=True).size()
    out.append("\nFinish-time DNFs by sex × event type:")
    out.append(ft_g.to_string())

    # --- Finish-time DNFs: overall time vs the typical cut-off, ranking, concentration ---
    # The official source flags these athletes as DNF and leaves them unranked while recording a
    # complete overall finish time; the statistics below describe the group without assuming why.
    ft["hours"] = ft["overall_sec"].astype("float64") / 3600
    ft["cutoff_h"] = ft["race_type"].astype(str).map(TYPICAL_CUTOFF_H)
    below = ft["hours"] < ft["cutoff_h"]
    src_ft = ft["source"].astype(str)
    out.append("\nFinish-time DNFs — overall time below the typical cut-off "
               f"(IRONMAN {TYPICAL_CUTOFF_H['im']:.1f} h, IRONMAN 70.3 {TYPICAL_CUTOFF_H['him']:.1f} h):")
    for rt in ["im", "him"]:
        m = ft["race_type"] == rt
        by_src = "; ".join(f"{s} {ratio((below & m & (src_ft == s)).sum(), (m & (src_ft == s)).sum())}"
                           for s in ["official", "coachcox"] if (m & (src_ft == s)).any())
        out.append(f"  {rt.upper()}: below {TYPICAL_CUTOFF_H[rt]:.1f} h: {ratio((below & m).sum(), m.sum())} ({by_src})")
    off = src_ft == "official"
    out.append(f"  both event types, below the typical cut-off of their event type: {ratio(below.sum(), len(ft))} "
               f"(official {ratio((below & off).sum(), off.sum())})")
    over = int(((ft["race_type"] == "im") & (ft["hours"] > IM_AXIS_MAX_H)).sum())
    out.append(f"  IM: overall time > {IM_AXIS_MAX_H:.1f} h (outside the Figure 4a axis): {over:,}")
    for rt, label in [("im", "IM"), ("him", "HIM")]:
        h = ft.loc[ft["race_type"] == rt, "hours"]
        short = h[h < SHORT_TIME_H[rt]]
        out.append(f"  {label}: overall time < {SHORT_TIME_H[rt]:.1f} h (Figure 4 left tail): {len(short):,}"
                   + (f" (shortest {short.min():.3f} h)" if len(short) else "")
                   + f"; shortest overall time among {label} finish-time DNFs: {h.min():.3f} h")
    # Tallest bar of Figure 4 (same bins: constant width from 0 h, left-closed; times beyond the
    # last edge fall outside every bin) and the bar that starts at the typical cut-off.
    for rt, label in [("im", "IM"), ("him", "HIM")]:
        h = ft.loc[ft["race_type"] == rt, "hours"].to_numpy()
        width_h = FIG4_BIN_MIN[rt] / 60
        start = np.floor(np.round(h * 60, 9) / FIG4_BIN_MIN[rt]) * width_h
        counts = pd.Series(start).round(6).value_counts().sort_index()
        mode_start = float(counts.idxmax())
        at_cut = float(counts.get(round(TYPICAL_CUTOFF_H[rt], 6), 0))
        out.append(f"  {label}: tallest Figure 4 bar {mode_start:.3f}-{mode_start + width_h:.3f} h "
                   f"({FIG4_BIN_MIN[rt]}-min bins): {ratio(int(counts.max()), len(h))}; "
                   f"bar starting at the typical cut-off ({TYPICAL_CUTOFF_H[rt]:.1f}-{TYPICAL_CUTOFF_H[rt] + width_h:.3f} h): "
                   f"{ratio(int(at_cut), len(h))}; at or above the typical cut-off: "
                   f"{ratio(int((h >= TYPICAL_CUTOFF_H[rt]).sum()), len(h))}")

    rank = ft["rank_overall"]
    unranked = rank.isna() | (rank == RANK_PLACEHOLDER)
    out.append(f"\nFinish-time DNFs — rank in the overall classification (unranked = rank_overall missing "
               f"or placeholder {RANK_PLACEHOLDER}):")
    out.append(f"  unranked: {ratio(unranked.sum(), len(ft))} (placeholder: {int((rank == RANK_PLACEHOLDER).sum()):,}; "
               f"missing: {int(rank.isna().sum()):,})")
    for s in ["official", "coachcox"]:
        m = src_ft == s
        out.append(f"    {s}: {ratio((unranked & m).sum(), m.sum())}")

    per_edition = ft["event_id"].value_counts()          # editions with >= 1 finish-time DNF, by count
    n_any = df["event_id"].nunique()                     # editions with >= 1 DNF of any kind
    k_top = int(np.floor(TOP_EDITION_SHARE * len(per_edition)))
    out.append("\nFinish-time DNFs — concentration by edition (event_id):")
    out.append(f"  editions with >= 1 finish-time DNF: {len(per_edition):,} of {n_any:,} editions with >= 1 DNF of any kind")
    out.append(f"  finish-time DNFs per edition (editions with >= 1): median {per_edition.median():.1f} "
               f"(P25 {per_edition.quantile(.25):.1f}, P75 {per_edition.quantile(.75):.1f}, max {per_edition.max():,})")
    out.append(f"  top 10% of those editions by count ({k_top:,} editions): "
               f"{ratio(per_edition.iloc[:k_top].sum(), per_edition.sum())} of finish-time DNFs")

    # --- Internal consistency of finish-time DNFs (Methods §DNF Classification) ---
    # Two distance criteria are reported because total_distance_km is the position of the last
    # timing mat crossed (not a continuous distance): a record with a finish time may carry the
    # last intermediate mat as distance. The reference is the finish position of each event's
    # course (its finishers' modal total distance; nominal where the event has no usable course).
    split_cols = ["swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec"]
    out.append("\nInternal consistency of finish-time DNFs:")
    ft_sum = ft[split_cols].fillna(0).astype("int64").sum(axis=1)
    within5 = (ft_sum - ft["overall_sec"].astype("int64")).abs() <= 5
    all5 = (ft[split_cols].fillna(0) > 0).all(axis=1)
    out.append("  sum of the five split times (missing = 0) within 5 s of overall_sec, all finish-time DNFs: "
               f"{ratio(within5.sum(), len(ft))}")
    out.append(f"  all five split times recorded (> 0): {ratio(all5.sum(), len(ft))}")
    out.append(f"  within 5 s among those with all five splits: {ratio(within5[all5].sum(), all5.sum())}")
    for src in ["official", "coachcox"]:
        m = src_ft == src
        out.append(f"    {src}: within 5 s {within5[m].mean()*100:.2f}% (n = {int(m.sum()):,}); all five splits {all5[m].mean()*100:.2f}%")
    ft_dist = ft[ft["total_distance_km"].notna()]
    to_finish = km_past(ft_dist["total_distance_km"], ft_dist["finish"])
    at_full = to_finish >= 0
    within3 = to_finish.abs() <= WITHIN_KM      # two-sided: no further than 3 km from the finish, either side
    out.append(f"  with distance data: {ratio(len(ft_dist), len(ft))} of finish-time DNFs (recorded distances not used under "
               f"the course rule, section 3: {int(ft['distance_not_used'].sum()):,})")
    out.append("  total_distance_km against the event's modal finisher distance (finish position of its course, "
               "at two decimals; nominal where the event has no usable course):")
    out.append(f"    at or above the event's modal finisher distance: {ratio(at_full.sum(), len(ft_dist))}")
    out.append(f"    within {WITHIN_KM:g} km of the event's modal finisher distance, two-sided (|distance - finish| <= "
               f"{WITHIN_KM:g} km): {ratio(within3.sum(), len(ft_dist))}")
    out.append(f"    more than {WITHIN_KM:g} km beyond the event's modal finisher distance: {ratio((to_finish > WITHIN_KM).sum(), len(ft_dist))}")
    for rt in ["im", "him"]:
        m = ft_dist["race_type"].astype(str) == rt
        if m.any():
            out.append(f"    {rt.upper()}: at or above {at_full[m].mean()*100:.2f}%, within {WITHIN_KM:g} km (two-sided) "
                       f"{within3[m].mean()*100:.2f}% (n = {int(m.sum()):,})")

    # --- Zero-valued splits among in-race DNFs (Methods §DNF Classification) ---
    # The zero-as-missing rule is about the official platform, which fills absent
    # splits with 0; the aggregator has its own conventions, so both are shown.
    out.append("\nSplit times > 0 among in-race DNFs (share of DNFs with a recorded time; numerators in the n columns,")
    out.append("denominators on the n line below):")
    out.append(f"  {'segment':8s} {'all':>9s} {'official':>9s} {'coachcox':>9s} {'n all':>9s} {'n official':>10s} {'n coachcox':>10s}")
    src = inr["source"].astype(str)
    for col, lbl in [("swim_sec", "swim"), ("t1_sec", "T1"), ("bike_sec", "bike"), ("t2_sec", "T2"), ("run_sec", "run")]:
        pos = inr[col].fillna(0) > 0
        po, pc = pos[src == "official"], pos[src == "coachcox"]
        out.append(f"  {lbl:8s} {pos.mean()*100:8.3f}% {po.mean()*100:8.3f}% {pc.mean()*100:8.3f}% "
                   f"{int(pos.sum()):>9,} {int(po.sum()):>10,} {int(pc.sum()):>10,}")
    any_split = (inr[split_cols].fillna(0) > 0).any(axis=1)
    out.append(f"  n: {len(inr):,} all, {int((src == 'official').sum()):,} official, {int((src == 'coachcox').sum()):,} coachcox")
    out.append(f"  no split time at all: {ratio((~any_split).sum(), len(inr))} of in-race DNFs "
               f"(official {ratio((~any_split & (src == 'official')).sum(), (src == 'official').sum())})")

    # --- Method 2: Split inference (all DNFs) ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 2: LAST COMPLETED SEGMENT — METHOD 2 (SPLIT INFERENCE)")
    out.append("=" * 60)
    out.append("\nLogic: last segment with split > 0 = last segment with a recorded time.")
    out.append("IMPORTANT: This identifies the last COMPLETED segment, not where the")
    out.append("race ended. The race may have ended during this segment or during the")
    out.append("transition to the next segment.")
    out.append("'finish_time': DNFs with a recorded overall finish time (overall_sec > 0); no segment is inferred.")
    out.append("Applied to ALL DNFs (both sources).")

    df = infer_last_completed_segment(df)
    for rt in ["im", "him"]:
        sub = df[df["race_type"] == rt]
        ct = sub["last_completed_segment"].value_counts()
        pct = (ct / len(sub) * 100).round(3)
        out.append(f"\n{rt.upper()} (n={len(sub):,}):")
        out.append(pd.DataFrame({"N": ct, "%": pct}).to_string())

    # By sex (in-race DNFs with a segment label, i.e. excluding finish_time and unknown)
    for rt in ["im", "him"]:
        sub = df[(df["race_type"] == rt) & (~df["last_completed_segment"].isin(["finish_time", "unknown"]))]
        seg = sub["last_completed_segment"].astype(str)
        sub_g = seg.groupby(sub["gender"], observed=True).value_counts(normalize=True).unstack(fill_value=0) * 100
        out.append(f"\n{rt.upper()} — Last completed segment by sex (%, excl finish-time DNFs & unknown):")
        out.append(sub_g.round(1).to_string())

    # --- Method 1: Distance-based (official source only, in-race DNFs) ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3: DROPOUT POINT — METHOD 1 (DISTANCE-BASED)")
    out.append("=" * 60)
    # Coverage of the distance field, defined on ALL DNFs (both sources, finish-time DNFs
    # included): the share of DNFs that carry a distance value (Methods §Dropout Point Localization),
    # as recorded and after the course rule.
    with_dist_raw = df["has_recorded_distance"]
    with_dist = df["total_distance_km"].notna()   # after the course rule (_common.attach_course)
    in_race = ~(df["overall_sec"].notna() & (df["overall_sec"] > 0))
    out.append("\nCoverage of the distance field — share of ALL DNFs (both sources, finish-time DNFs included)")
    out.append("carrying a distance value (the field exists only in the official source):")
    out.append(f"  all DNFs with a distance value: {ratio(with_dist_raw.sum(), len(df))}")
    for rt in ["im", "him"]:
        m = df["race_type"] == rt
        out.append(f"  {rt.upper()} DNFs with a distance value: {ratio((with_dist_raw & m).sum(), m.sum())}")
    bad = course[course["basis"].isin(UNUSABLE_DISTANCE_BASES)].sort_values(["race_type", "basis"])
    not_used = df["distance_not_used"]
    eids = df["event_id"].astype(str)
    out.append("\nDistances not used (course rule, _common.attach_course): the recorded distances of the events whose finishers'")
    out.append("distances do not form a usable course (basis " + " or ".join(UNUSABLE_DISTANCE_BASES) + ") are treated as")
    out.append("missing in every distance-based analysis (finish-time DNF distance checks in section 1, sections 3-5, the")
    out.append("hazard, the professional distance analysis and Figure 3). Events (modal finisher swim / swim + cycling / run /")
    out.append("total distance, km):")
    for e, r in bad.iterrows():
        m = (eids == e).to_numpy()
        out.append(f"  {event_names.get(e, e)} ({r['race_type']}, {r['basis']}) [{e}]: {r['swim']:g} / {r['swim_bike']:g} / "
                   f"{r['run']:g} / {r['total']:g}; finishers {int(r['n_finishers']):,}; DNFs with a distance "
                   f"{int((not_used & m).sum()):,} (in-race {int((not_used & m & in_race).sum()):,})")
    out.append(f"  DNF records whose distance is not used: {int(not_used.sum()):,} in {len(bad)} events (in-race "
               f"{int((not_used & in_race).sum()):,}, finish-time {int((not_used & ~in_race).sum()):,}; IM "
               f"{int((not_used & (df['race_type'] == 'im')).sum()):,}, HIM {int((not_used & (df['race_type'] == 'him')).sum()):,})")
    out.append(f"  all DNFs with a distance value used (after the course rule): {ratio(with_dist.sum(), len(df))}")
    for rt in ["im", "him"]:
        m = df["race_type"] == rt
        out.append(f"  {rt.upper()} DNFs with a distance value used (after the course rule): {ratio((with_dist & m).sum(), m.sum())}")
    nd = course[course["basis"] == "nominal_no_distance"]
    m_nd = eids.isin(set(nd.index)).to_numpy() & with_dist
    out.append(f"  events whose finishers carry no distance (basis nominal_no_distance; their DNF distances, if any, are placed on "
               f"the nominal positions): {len(nd):,} events holding {int(m_nd.sum()):,} DNFs with a distance (in-race "
               f"{int((m_nd & in_race).sum()):,})")
    m_nc = ((df["source"] == "official") & (df["course_basis"] == "nominal_no_course")).to_numpy() & with_dist
    out.append(f"  official-source events without finishers (no course row; nominal positions): {int(m_nc.sum()):,} DNFs with a "
               f"distance in {eids[m_nc].nunique():,} events")

    dnf_official = df[(df["source"] == "official") & in_race].copy()   # official in-race DNFs
    dnf_official = infer_dropout_from_distance(dnf_official, course)
    dnf_official = run_position(dnf_official, course, t2_lim)

    # Coverage of the distance field by season: official-source in-race DNFs carrying a distance value as recorded
    # (before the course rule); dropout_point_by_season.csv
    cov_season = by_season(dnf_official, dnf_official["has_recorded_distance"].astype(int), pd.Series(1, index=dnf_official.index),
                           ("n_inrace_with_recorded_distance", "n_inrace_official", "pct_with_recorded_distance"))
    out.append("\nCoverage of the distance field by season — official-source in-race DNFs carrying a distance value, as")
    out.append("recorded (before the course rule): with a distance / in-race DNFs = share:")
    out.append(season_table(cov_season, "n_inrace_with_recorded_distance", "n_inrace_official", "pct_with_recorded_distance"))
    dnf_events = set(dnf_official.loc[dnf_official["total_distance_km"].notna(), "event_id"].astype(str))
    out += course_summary(course, dnf_events)
    ec = course.copy()
    ec["distance_used"] = ~ec["basis"].isin(UNUSABLE_DISTANCE_BASES)
    ev_off = dnf_official["event_id"].astype(str)
    ec["n_inrace_dnf_with_distance"] = ev_off[dnf_official["has_recorded_distance"].to_numpy()].value_counts().reindex(ec.index).fillna(0).astype(int)
    ec["n_inrace_dnf_distance_used"] = ev_off[dnf_official["total_distance_km"].notna().to_numpy()].value_counts().reindex(ec.index).fillna(0).astype(int)
    write_csv(ec.reset_index(), "event_course.csv",
              "RQ2 — course positions of the official-source events (modal finisher distances; nominal fallback)",
              float_format=CSV_FLOAT)

    out.append("\nClassification: in-race DNFs only (official source). Finish-time DNFs are excluded: they are a")
    out.append("separate category (section 1), as in sections 4-5 and in the hazard (dropout_hazard.py).")
    out.append("Uses total_distance_km field (official source only): the position of the last timing mat crossed.")
    out.append("3 segments + no mat (no T1/T2 — transitions do not accumulate distance).")
    out.append("Boundaries: each event's own swim exit, cycling completion and finish (modal finisher distances;")
    out.append("nominal IM 3.86 / 184.11 / 226.31 km, HIM 1.93 / 92.05 / 113.15 km where the finishers carry no distance),")
    out.append("every boundary compared at two decimals (_common.km_past):")
    out.append(f"  '{NO_MAT_LABEL}': distance 0 (the field's default value; compatible with withdrawal during the swim);")
    out.append("  swim: at or before the swim exit; bike: beyond it, at or before the cycling completion; run: beyond the")
    out.append("  cycling completion, the finish distance included. A DNF at the cycling-completion distance is 'bike'.")
    out.append("  Records whose distance is not used (course rule above) are 'unknown'.")
    out.append("T2 rule (variant 't2_rule' of dropout_point_segments.csv): the segments of the hazard's run-DNF definition")
    out.append("(_common.run_position, RUN_RULES['t2_rule']; section 5): 'run' = DNFs at the cycling-completion distance with")
    out.append("a T2 split of at most the T2 duration limit of their event type (enter the run at km 0) + DNFs beyond the")
    out.append("cycling completion and before the finish distance; DNFs at or beyond the finish distance are separate")
    out.append("categories (not run DNFs); every other DNF keeps its distance class. The 'run' share is therefore the run")
    out.append("DNFs of the hazard over the in-race DNFs with a distance value used.")

    seg_rows = []
    for rt in ["im", "him"]:
        sub = dnf_official[dnf_official["race_type"] == rt]
        ct = sub["dropout_distance"].astype(str).value_counts().reindex(SEGMENTS + ["unknown"]).fillna(0).astype(int)
        pct = (ct / len(sub) * 100).round(3)
        out.append(f"\n{rt.upper()} (n={len(sub):,}):")
        out.append(pd.DataFrame({"N": ct, "%": pct}).rename_axis("dropout_distance").to_string())
        with_d = sub[sub["total_distance_km"].notna()]
        n_wd = len(with_d)
        at_t2 = int((with_d["run_class"] == "at_t2").sum())
        out.append(f"  shares of the {n_wd:,} in-race DNFs with a distance value: "
                   + "; ".join(f"{s} {ratio(ct[s], n_wd)}" for s in SEGMENTS))
        out.append(f"  at the cycling-completion distance (in 'bike'): "
                   f"{int(with_d['run_class'].isin(['at_t2', 'at_t2_long', 'at_no_t2']).sum()):,}, of which with a T2 split "
                   f"within the duration limit (enter the run at km 0 under the T2 rule): {at_t2:,}; with a T2 split longer than "
                   f"the limit: {int((with_d['run_class'] == 'at_t2_long').sum()):,}; without a T2 split: "
                   f"{int((with_d['run_class'] == 'at_no_t2').sum()):,}")
        ct_t2 = t2_rule_segment(with_d).value_counts().reindex(T2_RULE_SEGMENTS).fillna(0).astype(int)
        assert int(ct_t2.sum()) == n_wd, "t2_rule segments do not cover the in-race DNFs with a distance value"
        out.append("  T2 rule ('t2_rule'): " + "; ".join(f"{s} {ratio(ct_t2[s], n_wd)}" for s in T2_RULE_SEGMENTS))
        for variant, counts, segs in [("distance", ct, SEGMENTS), ("t2_rule", ct_t2, T2_RULE_SEGMENTS)]:
            for s in segs:
                n = int(counts[s])
                seg_rows.append({"race_type": rt, "variant": variant, "segment": s, "n": n,
                                 "n_with_distance": n_wd, "pct_of_with_distance": n / n_wd * 100,
                                 "n_inrace_official": len(sub), "n_without_distance": int(ct["unknown"])})
    write_csv(pd.DataFrame(seg_rows), "dropout_point_segments.csv",
              "RQ2 — section 3 segment shares of official in-race DNFs with a distance value (distance classification; "
              "variant t2_rule: the segments of the hazard's run-DNF definition, DNFs at or beyond the finish distance apart)",
              float_format=CSV_FLOAT)

    # Distance completed descriptives (in-race DNFs with a distance value used, i.e. after the course rule)
    dnf_dist = dnf_official[dnf_official["total_distance_km"].notna()]
    out.append(f"\nDistance completed by in-race DNFs with a distance value used (after the course rule; km, official source, "
               f"n={len(dnf_dist):,}):")
    for rt in ["im", "him"]:
        sub = dnf_dist[dnf_dist["race_type"] == rt]
        if len(sub) > 0:
            desc = sub["total_distance_km"].describe(percentiles=[.10, .25, .50, .75, .90])
            out.append(f"\n  {rt.upper()} (n={len(sub):,}):")
            out.append("  " + desc.round(3).to_string().replace("\n", "\n  "))
            rel = km_past(sub["total_distance_km"], sub["bike_end"])
            out.append(f"  position relative to the event's cycling completion (km): median {rel.median():.4f}; "
                       f"at or before it {ratio((rel <= 0).sum(), len(sub))}")

    # --- Method comparison ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 4: METHOD COMPARISON")
    out.append("=" * 60)

    both = dnf_official[   # in-race DNFs with a segment by splits and by distance
        (~dnf_official["last_completed_segment"].astype(str).isin(["unknown", "finish_time"])) &
        (dnf_official["dropout_distance"].astype(str) != "unknown")
    ].copy()
    if len(both) > 0:
        out.append(f"\nRecords with both methods available: {len(both):,}")
        out.append("\nNote: both methods read timing mats. The split method gives the last segment with a")
        out.append("recorded split time; the distance method gives the segment that contains the last mat")
        out.append("crossed (section 3 classification, event-specific boundaries). A DNF whose last split is T2")
        out.append("and whose last mat lies in the run is 'T2' by splits and 'run' by distance.")

        for rt in ["im", "him"]:
            sub = both[both["race_type"] == rt]
            agreement = (sub["last_completed_segment"].astype(str) == sub["dropout_distance"].astype(str)).mean() * 100
            out.append(f"\n{rt.upper()} (n={len(sub):,}) — cross-tabulation of split-based × distance-based segment")
            out.append(f"  label agreement, capped by construction (the T1/T2 split labels cannot match the "
                       f"swim/bike/run distance labels): {agreement:.1f}%")

            ct = pd.crosstab(
                sub["last_completed_segment"].astype(str),
                sub["dropout_distance"].astype(str),
                margins=True,
            )
            out.append(ct.to_string())

            # Key percentages
            for seg, dist_col in [("T2", "run"), ("T1", "bike"), ("T2", "bike")]:
                if seg in ct.index and dist_col in ct.columns:
                    out.append(f"  → {seg} by splits, {dist_col} by distance: "
                               f"{ratio(ct.loc[seg, dist_col], ct.loc[seg, 'All'])}")
            t2b = sub[(sub["last_completed_segment"].astype(str) == "T2") & (sub["dropout_distance"].astype(str) == "bike")]
            out.append(f"  → T2 by splits and bike by distance, at the cycling-completion distance: "
                       f"{ratio(t2b['run_class'].isin(['at_t2', 'at_t2_long', 'at_no_t2']).sum(), len(t2b)) if len(t2b) else '0'}")

    # --- Running position at the last checkpoint (the hazard's run-DNF definition) ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 5: RUNNING DISTANCE AT THE LAST CHECKPOINT (in-race run-segment DNFs)")
    out.append("=" * 60)
    out.append("\nRun DNFs exactly as in dropout_hazard.py (main analysis, T2 rule; _common.run_position, RUN_RULES):")
    out.append("km_run = total_distance_km - the event's cycling-completion distance, at two decimals (last checkpoint")
    out.append("crossed: a lower bound of the distance run). Run DNF = in-race DNF (finish-time DNFs excluded) with")
    out.append("  (a) km_run = 0 and a T2 split > 0 of at most the T2 duration limit of its event type (crossed the timing")
    out.append("      mat at the transition exit; enters the run at km 0), or")
    out.append("  (b) km_run > 0 and before the event's finish distance (km_run below the event's run length).")
    out.append("Not run DNFs: km_run = 0 with a T2 split longer than the limit or without a T2 split (withdrew in the")
    out.append("transition); records at or beyond the finish distance (no interval left to run). Sensitivities of the hazard:")
    out.append("strict rule, (b) only; T2 rule without the duration limit, (a) with any T2 split > 0, plus (b).")
    out.append("The distribution locates the last recorded checkpoint, so its shape also reflects where the")
    out.append("timing mats are placed; dropout_hazard.py estimates the interval-censored hazard per km.\n")

    # T2 duration limit and the DNFs at the cycling-completion distance by T2 split
    out.append(f"T2 duration limit (_common.t2_limits): the {T2_LIMIT_QUANTILE * 100:.0f}th percentile of t2_sec among the "
               "official-source finishers with a T2 split > 0 of the event type:")
    t2_rows = []
    for rt in ["im", "him"]:
        f = fin_t2[rt]
        out.append(f"  {rt.upper()}: limit {t2_lim[rt]:,.2f} s = {t2_lim[rt] / 60:.2f} min; finishers with a T2 split {len(f):,}, "
                   f"median T2 {f.median() / 60:.2f} min")
    out.append("In-race DNFs with a distance value at the cycling-completion distance, by T2 split (n; median T2 split, min):")
    for rt in ["im", "him"]:
        sub = dnf_official[(dnf_official["race_type"] == rt) & dnf_official["total_distance_km"].notna()]
        g = {k: sub.loc[sub["run_class"] == k, "t2_sec"].astype("float64") for k in ["at_t2", "at_t2_long", "at_no_t2"]}
        over60 = int((g["at_t2_long"] > 3600).sum())
        out.append(f"  {rt.upper()}: at_t2 (T2 split within the limit) {len(g['at_t2']):,}, median {g['at_t2'].median() / 60:.2f} min; "
                   f"at_t2_long (T2 split longer than the limit) {len(g['at_t2_long']):,}, median "
                   f"{g['at_t2_long'].median() / 60:.2f} min (over 60 min: {over60:,}); at_no_t2 (no T2 split) {len(g['at_no_t2']):,}")
        t2_rows.append({"race_type": rt, "t2_limit_sec": t2_lim[rt], "t2_limit_min": t2_lim[rt] / 60,
                        "n_finishers_with_t2": len(fin_t2[rt]), "median_t2_min_finishers": fin_t2[rt].median() / 60,
                        "n_at_t2": len(g["at_t2"]), "median_t2_min_at_t2": g["at_t2"].median() / 60,
                        "n_at_t2_long": len(g["at_t2_long"]), "median_t2_min_at_t2_long": g["at_t2_long"].median() / 60,
                        "n_at_t2_long_over_60_min": over60, "n_at_no_t2": len(g["at_no_t2"])})
    write_csv(pd.DataFrame(t2_rows), "dropout_point_t2.csv",
              "RQ2 — T2 duration limit (99th percentile of the official-source finishers' T2 splits) and the in-race DNFs at "
              "the cycling-completion distance by T2 split", float_format=CSV_FLOAT)

    # Run-segment DNFs of the main rule recorded at km 0 (T2-completers), by season; dropout_point_by_season.csv
    wd = dnf_official[dnf_official["total_distance_km"].notna()]
    run_all = wd[wd["run_class"].isin(RUN_RULES["t2_rule"])]
    km0_season = by_season(run_all, (run_all["run_class"] == "at_t2").astype(int), pd.Series(1, index=run_all.index),
                           ("n_run_dnf_km0", "n_run_dnf", "pct_run_dnf_km0"))
    out.append("\nRun-segment DNFs of the main rule (T2 rule) recorded at km 0, by season: at km 0 (T2-completers) / run DNFs")
    out.append("= share:")
    out.append(season_table(km0_season, "n_run_dnf_km0", "n_run_dnf", "pct_run_dnf_km0"))
    seasons = cov_season.merge(km0_season, on=["race_year", "race_type"], how="outer").sort_values(["race_type", "race_year"])
    for c in ("n_inrace_with_recorded_distance", "n_inrace_official", "n_run_dnf_km0", "n_run_dnf"):
        seasons[c] = seasons[c].fillna(0).astype(int)
    write_csv(seasons[["race_type", "race_year", "n_inrace_official", "n_inrace_with_recorded_distance",
                       "pct_with_recorded_distance", "n_run_dnf", "n_run_dnf_km0", "pct_run_dnf_km0"]],
              "dropout_point_by_season.csv",
              "RQ2 — by season and event type (race_type 'all' = both): official-source in-race DNFs carrying a distance value "
              "as recorded (section 3) and run-segment DNFs of the main rule recorded at km 0 (section 5)",
              float_format=CSV_FLOAT)

    for rt in ["im", "him"]:
        sub = dnf_official[(dnf_official["race_type"] == rt) & dnf_official["total_distance_km"].notna()]
        rc = sub["run_class"].value_counts().reindex(list(RUN_CLASS_LABELS)).fillna(0).astype(int)
        run = sub[sub["run_class"].isin(RUN_RULES["t2_rule"])]
        out.append(f"\n{rt.upper()}: in-race DNFs with a distance value: {len(sub):,}")
        for k, lab in RUN_CLASS_LABELS.items():
            out.append(f"  {k:10s} {rc[k]:>7,}  {lab}")
        out.append(f"{rt.upper()} run-segment in-race DNFs: n = {len(run):,}")
        out.append(f"  (T2-completers at km 0: {rc['at_t2']:,}; beyond the cycling completion: {rc['run']:,}; dropped: "
                   f"{rc['at_t2_long']:,} at the cycling-completion distance with a T2 split longer than the limit, "
                   f"{rc['at_no_t2']:,} at the cycling-completion distance without a T2 split, {rc['at_finish']:,} at the "
                   f"finish distance, {rc['beyond']:,} beyond the finish distance)")
        out.append(f"  strict rule (beyond the cycling completion only): n = {rc['run']:,}")
        out.append(f"  T2 rule without the duration limit (any T2 split at the cycling-completion distance): n = "
                   f"{int(sub['run_class'].isin(RUN_RULES['t2_no_limit']).sum()):,}")
        if len(run) > 0:
            desc = run["km_run"].describe(percentiles=[.10, .25, .50, .75, .90])
            out.append("  " + desc.round(3).to_string().replace("\n", "\n  "))
            # Cumulative thresholds
            for thr in [0, 1, 3, 5, 10, 21.1]:
                share = (run["km_run"] <= thr).mean() * 100
                out.append(f"  % with the last checkpoint within the first {thr} km of the run: {share:.2f}%")
            strict = run[run["run_class"] == "run"]["km_run"]
            out.append(f"  strict rule: median {strict.median():.3f} km (P25 {strict.quantile(.25):.3f}, P75 {strict.quantile(.75):.3f})")
            # The same statistics in the events with at least one run checkpoint beyond km 0 (at least one run
            # DNF with km_run > 0): in the other events every run DNF is at km 0, the only run checkpoint observed.
            ev_run = run["event_id"].astype(str)
            ev_beyond = set(ev_run[(run["run_class"] == "run").to_numpy()])
            rr = run[ev_run.isin(ev_beyond).to_numpy()]
            n_ev = ev_run.nunique()
            out.append(f"  events with a run checkpoint beyond km 0 (at least one run DNF with km_run > 0): {len(ev_beyond):,} of "
                       f"{n_ev:,} events with run DNFs; run DNFs in them: n = {len(rr):,} (T2-completers at km 0: "
                       f"{int((rr['run_class'] == 'at_t2').sum()):,}); events whose run DNFs are all at km 0: "
                       f"{n_ev - len(ev_beyond):,} ({len(run) - len(rr):,} T2-completers)")
            q = rr["km_run"].quantile([.10, .25, .50, .75, .90])
            out.append(f"  events with a run checkpoint beyond km 0, km_run: mean {rr['km_run'].mean():.3f}, P10 {q[.10]:.3f}, "
                       f"P25 {q[.25]:.3f}, median {q[.50]:.3f}, P75 {q[.75]:.3f}, P90 {q[.90]:.3f}, max {rr['km_run'].max():.3f}")
            for thr in [0, 1, 3, 5, 10, 21.1]:
                share = (rr["km_run"] <= thr).mean() * 100
                out.append(f"  % within the first {thr} km of the run (events with a run checkpoint beyond km 0): {share:.2f}%")

    # --- Save ---
    output_text = "\n".join(out)
    (RESULTS_DIR / "dropout_point.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(f"Saved to {RESULTS_DIR / 'dropout_point.txt'}")
    print(output_text)


if __name__ == "__main__":
    main()
