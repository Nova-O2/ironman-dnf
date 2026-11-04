"""
Professional athletes: rank at the end of the cycling segment.

Describes the rank in the overall field at the end of the cycling segment (bike_rank)
of IRONMAN professional in-race DNFs (DNFs without a recorded finish time) and compares it with
that of professional finishers (shares outside the top N, odds ratio of being outside the top 10,
Mann-Whitney U), plus the annual IRONMAN male professional DNF rate and curation statistics on
professional DNFs (pro_curation.txt: split coverage, the rate without the DNFs that have no split,
and the recorded distance of IRONMAN professional in-race DNFs).

Output: results/pro_rank.txt, results/pro_curation.txt + manuscript/tables/TableS4.md
(distribution by rank band).

Coverage caveat: the analysis is restricted to IM PRO in-race DNFs with a valid `bike_rank`
value at a recorded cycling split — that is, records satisfying BOTH
`bike_sec > 0` AND `bike_rank` in the range [1, 99998]. The dataset uses 99999
as a placeholder for missing rank (pro_rank.txt section 1 gives the share covered);
the remainder have no recorded cycling split or no recorded rank at that checkpoint.
Finish-time DNFs (a recorded finish time) are excluded: they completed the course.

Distances (pro_curation.txt section 3) are positioned against each event's own cycling completion
(modal finisher distances; _common.event_course_table, nominal fallback), at two decimals. The recorded
distances of events whose finishers' distances do not form a usable course (basis nominal_incoherent or
nominal_scale) are not used: _common.attach_course sets them to missing before the records with a distance
are selected.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_dataset, event_course_table, attach_course, km_past, fmt_half_up, pct_half_up,
    reproducibility_header, RESULTS_DIR, TABLE_DIR, COURSE_COLUMNS,
)


def in_race(frame: pd.DataFrame) -> pd.Series:
    """DNFs without a recorded finish time (overall_sec missing or 0)."""
    return ~(frame["overall_sec"].fillna(0) > 0)


def compute_pro_curation_stats():
    """Compute curation/integrity statistics on professional DNFs.

    Persists:
      - share of professional DNFs with a recorded swim split (race participation confirmed)
      - share of professional DNFs with NO recorded split times (possible DNS miscoded as DNF)
      - adjusted IRONMAN male PRO DNF rate after excluding the no-split PROs from
        both numerator and denominator
      - median total_distance_km of IRONMAN PRO in-race DNFs with distance data, and their
        position relative to the event's own cycling completion

    Output: results/pro_curation.txt
    """
    cols = list(dict.fromkeys([
        "finish_status", "race_type", "age_group",
        "swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec", "overall_sec",
        "total_distance_km",
    ] + COURSE_COLUMNS))
    print("Loading dataset for PRO curation stats...")
    df = load_dataset(cols)
    course = event_course_table(df)

    # All PROs (MPRO/FPRO) across both distances
    pro = df[df["age_group"].isin(["MPRO", "FPRO"])].copy()
    pro_starts = pro[pro["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()
    pro_dnf = pro[pro["finish_status"] == "DNF"].copy()

    # Split columns considered (transitions + main segments; overall_sec is not a "split")
    split_cols = ["swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec"]

    def has_any_split(frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=frame.index)
        for col in split_cols:
            mask = mask | (frame[col].fillna(0) > 0)
        return mask

    def has_swim_split(frame: pd.DataFrame) -> pd.Series:
        return frame["swim_sec"].fillna(0) > 0

    out = []
    out.append(reproducibility_header("PRO Curation / Integrity Statistics"))
    out.append("Subset: all professional (MPRO/FPRO) records in IRONMAN + IRONMAN 70.3.")
    out.append("Splits considered: swim_sec, t1_sec, bike_sec, t2_sec, run_sec (>0 = present).")

    # --- Section 1: swim split coverage among all PRO DNFs (both IM + HIM) ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 1: SWIM-SPLIT COVERAGE AMONG PROFESSIONAL DNFs")
    out.append("=" * 60)

    n_pro_dnf = len(pro_dnf)
    n_with_swim = has_swim_split(pro_dnf).sum()
    n_with_any = has_any_split(pro_dnf).sum()
    n_no_splits = n_pro_dnf - n_with_any

    pct_with_swim = n_with_swim / n_pro_dnf * 100 if n_pro_dnf else 0
    pct_no_splits = n_no_splits / n_pro_dnf * 100 if n_pro_dnf else 0

    out.append(f"\nTotal professional DNFs (MPRO+FPRO, IM+HIM): {n_pro_dnf:,}")
    out.append(f"  with recorded swim split (swim_sec > 0):       {n_with_swim:,} ({pct_with_swim:.4f}%)")
    out.append(f"  with any recorded split (any segment > 0):     {n_with_any:,} ({n_with_any / n_pro_dnf * 100:.4f}%)")
    out.append(f"  with NO recorded splits (possible DNS→DNF):    {n_no_splits:,} ({pct_no_splits:.4f}%)")

    # Broken down by distance for context
    for rt, label in [("im", "IRONMAN"), ("him", "IRONMAN 70.3")]:
        sub = pro_dnf[pro_dnf["race_type"] == rt]
        n = len(sub)
        nsw = has_swim_split(sub).sum()
        nas = has_any_split(sub).sum()
        nns = n - nas
        out.append(
            f"  [{label}] n={n:,} | swim split: {nsw:,} ({nsw/n*100:.4f}%) | "
            f"no splits: {nns:,} ({nns/n*100:.4f}%)"
        )

    # --- Section 2: IRONMAN male PRO adjusted DNF rate after excluding no-split PROs ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 2: IRONMAN MALE PRO DNF RATE — ADJUSTED")
    out.append("=" * 60)
    out.append("\nAdjustment: exclude professional DNFs with NO recorded splits")
    out.append("from BOTH numerator and denominator (treated as possible DNS miscoded as DNF).")

    im_mpro = pro_starts[(pro_starts["race_type"] == "im") & (pro_starts["age_group"] == "MPRO")].copy()
    im_mpro_dnf = im_mpro[im_mpro["finish_status"] == "DNF"].copy()

    n_starts_raw = len(im_mpro)
    n_dnf_raw = len(im_mpro_dnf)
    rate_raw = n_dnf_raw / n_starts_raw * 100 if n_starts_raw else 0

    dnf_with_any_split = has_any_split(im_mpro_dnf)
    n_dnf_adj = dnf_with_any_split.sum()
    n_dnf_excluded = n_dnf_raw - n_dnf_adj
    n_starts_adj = n_starts_raw - n_dnf_excluded
    rate_adj = n_dnf_adj / n_starts_adj * 100 if n_starts_adj else 0

    out.append(f"\nIRONMAN male PRO (MPRO, race_type=im):")
    out.append(f"  Raw:      starts={n_starts_raw:,}, DNF={n_dnf_raw:,}, rate={rate_raw:.4f}%")
    out.append(f"  Excluded: DNFs with no recorded splits = {n_dnf_excluded:,}")
    out.append(f"  Adjusted: starts={n_starts_adj:,}, DNF={n_dnf_adj:,}, rate={rate_adj:.4f}%")

    # Also compute for the other 3 PRO cells for completeness
    out.append("\nAdjusted rates for all four PRO cells:")
    for rt, ag, label in [
        ("im", "MPRO", "IRONMAN male"),
        ("im", "FPRO", "IRONMAN female"),
        ("him", "MPRO", "IRONMAN 70.3 male"),
        ("him", "FPRO", "IRONMAN 70.3 female"),
    ]:
        cell = pro_starts[(pro_starts["race_type"] == rt) & (pro_starts["age_group"] == ag)].copy()
        cell_dnf = cell[cell["finish_status"] == "DNF"].copy()
        n_s = len(cell)
        n_d = len(cell_dnf)
        rt_raw = n_d / n_s * 100 if n_s else 0
        d_any = has_any_split(cell_dnf).sum()
        d_excl = n_d - d_any
        s_adj = n_s - d_excl
        rt_adj = d_any / s_adj * 100 if s_adj else 0
        out.append(
            f"  {label:>22}: raw={rt_raw:.4f}% (n={n_s:,}) | "
            f"adjusted={rt_adj:.4f}% (n={s_adj:,}, excluded {d_excl} no-split DNFs)"
        )

    # --- Section 3: distance of IRONMAN PRO in-race DNFs with distance data ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3: MEDIAN TOTAL DISTANCE — IRONMAN PROFESSIONAL IN-RACE DNFs")
    out.append("=" * 60)
    out.append("\nSubset: IRONMAN PRO (MPRO+FPRO) in-race DNFs (no recorded finish time) with valid total_distance_km")
    out.append("(i.e. total_distance_km not null after the course rule of _common.attach_course); finish-time DNFs excluded.")

    im_pro_dnf = pro_dnf[pro_dnf["race_type"] == "im"].copy()
    im_pro_inrace = attach_course(im_pro_dnf[in_race(im_pro_dnf)].copy(), course)   # unusable-course distances set to missing
    d = im_pro_inrace[im_pro_inrace["total_distance_km"].notna()].copy()

    n_with_dist = len(d)
    n_total = len(im_pro_inrace)
    median_dist = d["total_distance_km"].astype("float64").median() if n_with_dist else float("nan")
    rel = km_past(d["total_distance_km"], d["bike_end"])

    out.append(f"\nIM PRO DNFs (total): {len(im_pro_dnf):,}; finish-time DNFs: {len(im_pro_dnf) - n_total:,}")
    out.append(f"IM PRO in-race DNFs (total):           {n_total:,}")
    out.append(f"IM PRO in-race DNFs with total_distance_km: {n_with_dist:,} "
               f"({n_with_dist/n_total*100:.4f}% of IM PRO in-race DNFs)")
    out.append(f"  (recorded distances not used under the course rule, _common.attach_course: "
               f"{int(im_pro_inrace['distance_not_used'].sum()):,})")
    out.append(f"Median total_distance_km:              {median_dist:.4f} km")
    out.append(f"Position relative to the event's cycling completion (km past it; event-specific, at two decimals): "
               f"median {rel.median():.4f}; P25 {rel.quantile(.25):.4f}; P75 {rel.quantile(.75):.4f}")
    out.append(f"  before the cycling completion: {int((rel < 0).sum()):,} ({(rel < 0).mean()*100:.4f}%); "
               f"at it: {int((rel == 0).sum()):,} ({(rel == 0).mean()*100:.4f}%); "
               f"beyond it: {int((rel > 0).sum()):,} ({(rel > 0).mean()*100:.4f}%)")
    out.append(f"(Reference: the event's own cycling-completion distance; nominal 184.11 km where the event's finishers carry no distance)")

    # Distance distribution
    desc = d["total_distance_km"].astype("float64").describe(percentiles=[.10, .25, .50, .75, .90])
    out.append("\nDistance distribution (km):")
    out.append("  " + desc.round(4).to_string().replace("\n", "\n  "))

    # --- Save ---
    output_text = "\n".join(out)
    (RESULTS_DIR / "pro_curation.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(f"Saved to {RESULTS_DIR / 'pro_curation.txt'}")
    print(output_text)

    return {
        "pct_pro_dnf_with_swim_split": pct_with_swim,
        "pct_pro_dnf_no_splits": pct_no_splits,
        "im_male_pro_rate_raw": rate_raw,
        "im_male_pro_rate_adjusted": rate_adj,
        "im_pro_dnf_median_distance_km": median_dist,
    }


def write_table_s4(valid_dnf: pd.DataFrame, valid_fin: pd.DataFrame, n_dnf_total: int, n_fin_total: int) -> None:
    """Write TableS4.md — bike_rank bands for IRONMAN® PRO in-race DNFs next to PRO finishers.
    Keeps the markdown source under source control so that re-running
    the script regenerates the table with current data. The note is
    descriptive; interpretation belongs to the Discussion. Every percentage is computed from the
    counts and rounded half up (_common.pct_half_up).
    """
    bands = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 99998)]
    band_labels = ["1–5", "6–10", "11–20", "21–50", "51–100", "101 or worse"]
    n_dnf, n_fin = len(valid_dnf), len(valid_fin)
    rows = []
    for (lo, hi), lbl in zip(bands, band_labels):
        nd = int(((valid_dnf["bike_rank"] >= lo) & (valid_dnf["bike_rank"] <= hi)).sum())
        nf = int(((valid_fin["bike_rank"] >= lo) & (valid_fin["bike_rank"] <= hi)).sum())
        rows.append(f"| {lbl} | {nd:,} | {pct_half_up(nd, n_dnf)} | {nf:,} | {pct_half_up(nf, n_fin)} |")

    out10_d = pct_half_up((valid_dnf["bike_rank"] > 10).sum(), n_dnf)
    out10_f = pct_half_up((valid_fin["bike_rank"] > 10).sum(), n_fin)
    out20_d = pct_half_up((valid_dnf["bike_rank"] > 20).sum(), n_dnf)
    out20_f = pct_half_up((valid_fin["bike_rank"] > 20).sum(), n_fin)

    md = f"""**Table S4.** Rank in the overall field at the end of the cycling segment among IRONMAN® professional athletes with a recorded cycling split and valid rank data: in-race DNFs (DNFs without a recorded finish time, n = {n_dnf:,}) and finishers (n = {n_fin:,}).

| Rank band | DNF, n | DNF, % | Finishers, n | Finishers, % |
|:---|---:|---:|---:|---:|
{chr(10).join(rows)}
| **Total** | **{n_dnf:,}** | **100.0** | **{n_fin:,}** | **100.0** |

Outside the top 10 at the end of the cycling segment: {out10_d}% of professional in-race DNFs versus {out10_f}% of professional finishers; outside the top 20: {out20_d}% versus {out20_f}%.
The rank is the position in the overall field (all categories) at the end of the cycling segment; the dataset uses 99,999 as a placeholder for missing rank, and records with that value or without a cycling split are excluded, leaving {n_dnf:,} of {n_dnf_total:,} professional in-race DNFs and {n_fin:,} of {n_fin_total:,} professional finishers. DNFs with a recorded finish time are not included.
"""
    (TABLE_DIR / "TableS4.md").write_text(md, encoding="utf-8", newline="\n")
    print(f"Saved to {TABLE_DIR / 'TableS4.md'}")


def main():
    cols = ["finish_status", "race_type", "age_group", "race_year", "bike_sec", "bike_rank", "overall_sec"]
    print("Loading dataset...")
    df = load_dataset(cols)

    out = []
    out.append(reproducibility_header("Professional athletes — rank at the end of the cycling segment (in-race DNFs vs finishers)"))

    # PRO IM starters
    pro_im = df[df["age_group"].isin(["MPRO", "FPRO"]) & (df["race_type"] == "im")
                & df["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()   # starters only (FIN + DNF + DQ)
    pro_im_dnf_all = pro_im[pro_im["finish_status"] == "DNF"].copy()
    pro_im_dnf = pro_im_dnf_all[in_race(pro_im_dnf_all)].copy()       # in-race DNFs only
    pro_im_fin = pro_im[pro_im["finish_status"] == "FIN"].copy()

    out.append("=" * 60)
    out.append("SECTION 1: SAMPLE AND COVERAGE")
    out.append("=" * 60)
    out.append(f"\nIM PRO starts (all finish_status in FIN/DNF/DQ): {len(pro_im):,}")
    out.append(f"IM PRO DNFs (total): {len(pro_im_dnf_all):,}; finish-time DNFs (recorded finish time, excluded): "
               f"{len(pro_im_dnf_all) - len(pro_im_dnf):,}; in-race DNFs: {len(pro_im_dnf):,}")
    out.append(f"IM PRO FINishers (total): {len(pro_im_fin):,}")

    # Valid subset: bike_sec > 0 (completed bike timing) AND bike_rank real (1..99998)
    def valid_rank(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[
            (frame["bike_sec"].fillna(0) > 0)
            & (frame["bike_rank"].notna())
            & (frame["bike_rank"] > 0)
            & (frame["bike_rank"] < 99999)
        ].copy()

    valid_dnf = valid_rank(pro_im_dnf)
    valid_fin = valid_rank(pro_im_fin)

    coverage_dnf = len(valid_dnf) / len(pro_im_dnf) * 100 if len(pro_im_dnf) else 0
    coverage_fin = len(valid_fin) / len(pro_im_fin) * 100 if len(pro_im_fin) else 0
    out.append("")
    out.append("Valid subset: bike_sec > 0 AND bike_rank real (1..99998):")
    out.append(f"  In-race DNFs with valid bike_rank: {len(valid_dnf):,} ({coverage_dnf:.4f}% of in-race DNFs)")
    out.append(f"  FINs with valid bike_rank: {len(valid_fin):,} ({coverage_fin:.4f}% of FINs)")

    # --- Descriptive distribution ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 2: BIKE_RANK DISTRIBUTION AT THE END OF THE CYCLING SEGMENT (PRO IN-RACE DNFs)")
    out.append("=" * 60)

    desc = valid_dnf["bike_rank"].describe(percentiles=[.10, .25, .50, .75, .90])
    out.append("\nIM PRO in-race DNFs with valid bike_rank:")
    out.append("  " + desc.round(2).to_string().replace("\n", "\n  "))

    # Cumulative % outside top N
    out.append("\nCumulative % of PRO in-race DNFs OUTSIDE top-N at the end of the bike segment:")
    for n in [5, 10, 20, 30, 50]:
        pct = (valid_dnf["bike_rank"] > n).sum() / len(valid_dnf) * 100
        out.append(f"  outside top {n}: {pct:.4f}%")

    # Reference: FIN distribution
    out.append("\nReference — IM PRO FINishers bike_rank distribution:")
    desc_fin = valid_fin["bike_rank"].describe(percentiles=[.10, .25, .50, .75, .90])
    out.append("  " + desc_fin.round(2).to_string().replace("\n", "\n  "))

    # --- Finisher comparator: the base rate matters. "x% of DNFs were outside the top 10" says
    # nothing without the same share among professionals who went on to finish. bike_rank is the
    # rank in the overall field at the end of the cycling segment (not the professional-only rank).
    out.append("\nFinisher comparator — % OUTSIDE top-N at the end of the bike segment, in-race DNFs vs FINishers:")
    out.append(f"  {'top-N':<8}{'DNF %':>10}{'FIN %':>10}{'diff pp':>10}")
    for n in [5, 10, 20, 30, 50]:
        p_d = (valid_dnf["bike_rank"] > n).mean() * 100
        p_f = (valid_fin["bike_rank"] > n).mean() * 100
        out.append(f"  {n:<8}{p_d:>10.4f}{p_f:>10.4f}{p_d - p_f:>10.4f}")

    # Odds of being outside the top 10 (DNF vs FIN), Wald 95% CI, and Mann-Whitney on bike_rank
    a = int((valid_dnf["bike_rank"] > 10).sum()); b = len(valid_dnf) - a
    c = int((valid_fin["bike_rank"] > 10).sum()); d = len(valid_fin) - c
    or_ = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_lo, ci_hi = np.exp(np.log(or_) - 1.96 * se), np.exp(np.log(or_) + 1.96 * se)
    mw = mannwhitneyu(valid_dnf["bike_rank"], valid_fin["bike_rank"], alternative="two-sided")
    out.append(f"\n  Outside top 10 — in-race DNF {a:,}/{len(valid_dnf):,} vs FIN {c:,}/{len(valid_fin):,}: "
               f"OR = {or_:.10g} (95% CI {ci_lo:.10g}-{ci_hi:.10g}); Wald CI on log(OR), SE {se:.10g}")
    out.append(f"  Mann-Whitney U on bike_rank (in-race DNF vs FIN): U = {mw.statistic:,.0f}, p = {mw.pvalue:.10g}; "
               f"median DNF {valid_dnf['bike_rank'].median():.1f} vs FIN {valid_fin['bike_rank'].median():.1f}")

    # --- TableS4: Distribution by rank band (for manuscript) ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3: TableS4 — DISTRIBUTION BY RANK BAND (PRO IN-RACE DNFs)")
    out.append("=" * 60)

    bands = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 99998)]
    band_labels = ["1–5", "6–10", "11–20", "21–50", "51–100", "101+"]

    rows = []
    n_total = len(valid_dnf)
    for (lo, hi), lbl in zip(bands, band_labels):
        n_band = ((valid_dnf["bike_rank"] >= lo) & (valid_dnf["bike_rank"] <= hi)).sum()
        pct = n_band / n_total * 100 if n_total else 0
        rows.append((lbl, n_band, pct))
    out.append(f"\nIM PRO in-race DNFs with valid bike_rank (n = {n_total:,}):")
    out.append(f"  {'Rank band':<10}{'N':>8}{'%':>10}")
    for lbl, n, pct in rows:
        out.append(f"  {lbl:<10}{n:>8,}{pct:>10.4f}")

    # --- Section 4: annual IM male PRO DNF rate — range and median over the seasons with
    # at least 100 starts, quoted in Results.
    out.append("\n" + "=" * 60)
    out.append("SECTION 4: IRONMAN MALE PRO DNF RATE BY YEAR (starters = FIN + DNF + DQ)")
    out.append("=" * 60)
    mpro = pro_im[pro_im["age_group"] == "MPRO"]
    by_year = mpro.groupby("race_year", observed=True).agg(
        starts=("finish_status", "size"),
        dnf=("finish_status", lambda s: int((s == "DNF").sum())),
    )
    by_year["rate_pct"] = by_year["dnf"] / by_year["starts"] * 100
    out.append("\n" + by_year.round(4).to_string())
    full = by_year[by_year["starts"] >= 100]
    out.append(f"\nYears with >= 100 MPRO starts: {len(full)} ({full.index.min()}-{full.index.max()}); "
               f"annual rate range {full['rate_pct'].min():.4f}%-{full['rate_pct'].max():.4f}%, "
               f"median {full['rate_pct'].median():.4f}%")
    out.append(f"All years (incl. small fields): range {by_year['rate_pct'].min():.1f}%-{by_year['rate_pct'].max():.1f}%")

    # Save text results
    output_text = "\n".join(out)
    (RESULTS_DIR / "pro_rank.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(f"Saved to {RESULTS_DIR / 'pro_rank.txt'}")

    # Save TableS4 (markdown, for manuscript)
    write_table_s4(valid_dnf, valid_fin, len(pro_im_dnf), len(pro_im_fin))

    print(output_text)


if __name__ == "__main__":
    main()
    print("\n" + "#" * 60)
    print("# Running PRO curation stats ...")
    print("#" * 60 + "\n")
    compute_pro_curation_stats()
