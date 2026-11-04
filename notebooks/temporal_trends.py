"""
RQ3: Temporal trends in DNF rates (2002-2026, final season partial).

Annual series per event type (starters = FIN + DNF + DQ; professionals included):
  in-race DNF rate      in-race DNFs (DNFs without a recorded finish time) / starters: the PRIMARY series of
                        the trend tests, as in-race DNFs are the outcome of the regression (RQ4) and the event
                        of the hazard (RQ2)
  total DNF rate        DNFs / starters (finish-time DNFs included): secondary series
  finish-time DNF rate  DNFs with a recorded finish time / starters (the total rate minus the in-race rate)
Trend tests on the three series of each event type, over its seasons except 2020 (COVID-19 pandemic, much reduced
field) and the partial 2026 season: ordinary least-squares slope of the annual rate on calendar year with a t-based
95% CI, R² and p; Mann-Kendall test (tau, p) and Sen's slope against calendar year. The in-race series is also
tested within the seasons up to 2009 and over 2010-2025 (RESTRICTED_SEASONS; 2020 excluded), and its rate is pooled
by period (PERIODS: in-race DNFs over starters, summed over the seasons of the period; 2020 and 2026 excluded). The
tests are fitted on the unrounded annual rates. By season, the share of the starters and of the finish-time DNFs
that come from the official source is persisted (section 1e). Year-to-year variation around the pandemic is
described, not modelled (sections 3-4).
Produces: results/temporal_trends.txt, temporal_trends_annual.csv (season x event type: counts, rates and
official-source shares at full precision), temporal_trends_tests.csv (trend statistics of the tested series at full
precision; column seasons: 'all' = the seasons above, '<first season>-2009' and '2010-2025' = the restricted in-race
tests) and temporal_trends_periods.csv (pooled in-race DNF rate by period, with numerators and denominators)
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats
import pymannkendall as mk

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_dataset, infer_gender, reproducibility_header, write_csv, RESULTS_DIR,
)

EXCLUDED_SEASONS = (2020, 2026)   # 2020: COVID-19 pandemic, much reduced field; 2026: partial season
# Tested series: key -> (rate column of the annual table, label); the primary series first
SERIES = {
    "inrace": ("inrace_rate", "in-race DNF rate (primary)"),
    "total": ("DNF_rate", "total DNF rate (secondary)"),
    "finish_time": ("finish_time_rate", "finish-time DNF rate (total minus in-race)"),
}
# The in-race series is also tested within these windows: (first season, last season); None = the first season of the
# event type; 2020 excluded, as in the trend tests.
RESTRICTED_SEASONS = ((None, 2009), (2010, 2025))
# Pooled in-race DNF rate by period: (first season, last season); None = the first season of the event type.
# 2020 and the partial 2026 season are excluded, as in the trend tests.
PERIODS = ((None, 2009), (2010, 2019), (2021, 2025))
TYPE_LABELS = {"im": "IM", "him": "HIM"}
CSV_FLOAT = "%.10g"
OUT_ANNUAL = "temporal_trends_annual.csv"
OUT_TESTS = "temporal_trends_tests.csv"
OUT_PERIODS = "temporal_trends_periods.csv"


def trend_tests(years, rates) -> dict:
    """OLS slope (t-based 95% CI, R², p), Mann-Kendall test and Sen's slope vs calendar year."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, rates)
    dof = len(years) - 2
    t_q = stats.t.ppf(0.975, dof)
    mk_result = mk.original_test(rates)
    sen = stats.theilslopes(rates, years)
    return {"n_years": len(years), "first_season": int(min(years)), "last_season": int(max(years)),
            "ols_slope": slope, "ols_se": std_err, "t_quantile": t_q, "df": dof,
            "ols_ci_low": slope - t_q * std_err, "ols_ci_high": slope + t_q * std_err,
            "ols_r2": r_value ** 2, "ols_p": p_value,
            "mk_trend": mk_result.trend, "mk_tau": mk_result.Tau, "mk_p": mk_result.p,
            "sen_slope": sen.slope, "sen_ci_low": sen.low_slope, "sen_ci_high": sen.high_slope}


def fmt_p(p: float) -> str:
    """p-value for the text output: six decimals, or three significant digits below 1e-6 (full precision in the CSV)."""
    return f"{p:.6f}" if p >= 1e-6 else f"{p:.3e}"


def test_lines(lab: str, slabel: str, r: dict) -> list[str]:
    """Text lines of one set of trend tests (event type label, series label, trend_tests result)."""
    return [
        f"\n{lab} — Linear regression, {slabel} (n={r['n_years']} years, {r['first_season']}-{r['last_season']}):",
        f"  Slope: {r['ols_slope']:.6f} pp/year (95% CI: {r['ols_ci_low']:.6f} to {r['ols_ci_high']:.6f}; "
        f"t quantile {r['t_quantile']:.4f}, df = {r['df']}; SE {r['ols_se']:.6f})",
        f"  R²: {r['ols_r2']:.4f}",
        f"  p-value: {fmt_p(r['ols_p'])}",
        f"\n{lab} — Mann-Kendall trend test, {slabel}:",
        f"  Classification (pymannkendall, two-sided, alpha = 0.05): {r['mk_trend']}",
        f"  Tau: {r['mk_tau']:.4f}",
        f"  p-value: {fmt_p(r['mk_p'])}",
        f"  Sen's slope: {r['sen_slope']:.6f} pp/year (vs calendar year; 95% CI {r['sen_ci_low']:.6f} "
        f"to {r['sen_ci_high']:.6f})",
    ]


def main():
    cols = ["finish_status", "race_type", "age_group", "race_year", "overall_sec", "event_id", "event_name", "source"]
    print("Loading dataset...")
    df = load_dataset(cols)
    df = infer_gender(df)
    df = df[df["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()
    df["is_dnf"] = (df["finish_status"] == "DNF").astype(int)
    has_time = df["overall_sec"].notna() & (df["overall_sec"] > 0)
    df["is_inrace"] = ((df["is_dnf"] == 1) & ~has_time).astype(int)        # DNF without a recorded finish time
    df["is_finish_time"] = ((df["is_dnf"] == 1) & has_time).astype(int)    # DNF with a recorded finish time
    df["is_official"] = (df["source"].astype(str) == "official").astype(int)
    df["is_finish_time_official"] = df["is_finish_time"] * df["is_official"]

    out = []
    out.append(reproducibility_header("RQ3 — Temporal Trends (incl. COVID-era variation)"))
    out.append("NOTE: starters = FIN + DNF + DQ, professionals included. Total DNF rate = DNFs / starters (finish-time")
    out.append("DNFs included); in-race DNF rate = in-race DNFs (DNFs without a recorded finish time) / starters;")
    out.append("finish-time DNF rate = DNFs with a recorded finish time / starters. The trend tests use the in-race")
    out.append("DNF rate as the primary series (in-race DNFs are the outcome of the regression and the event of the")
    out.append("hazard), the total DNF rate as the secondary series and the finish-time DNF rate, the difference")
    out.append("between the two, as a third series. The logistic regression (RQ4) also excludes professionals.")
    out.append("")

    # --- Annual rates ---
    out.append("=" * 60)
    out.append("SECTION 1: ANNUAL DNF RATES BY EVENT TYPE (total, in-race and finish-time)")
    out.append("=" * 60)
    annual = df.groupby(["race_year", "race_type"], observed=True).agg(
        N_starts=("is_dnf", "count"), N_DNF=("is_dnf", "sum"), N_inrace=("is_inrace", "sum"),
        N_finish_time=("is_finish_time", "sum"), N_starts_official=("is_official", "sum"),
        N_finish_time_official=("is_finish_time_official", "sum"))
    # unrounded rates (%): the trend tests fit these values; temporal_trends_annual.csv persists them
    annual["DNF_rate"] = annual["N_DNF"] / annual["N_starts"] * 100
    annual["inrace_rate"] = annual["N_inrace"] / annual["N_starts"] * 100
    annual["finish_time_rate"] = annual["N_finish_time"] / annual["N_starts"] * 100
    # official-source shares (%) of the starters and of the finish-time DNFs (missing when the season has none)
    annual["official_starts_pct"] = annual["N_starts_official"] / annual["N_starts"] * 100
    annual["official_finish_time_pct"] = (annual["N_finish_time_official"] / annual["N_finish_time"]
                                          * 100).where(annual["N_finish_time"] > 0)
    rate_cols = ["N_starts", "N_DNF", "DNF_rate", "N_inrace", "inrace_rate", "N_finish_time", "finish_time_rate"]
    source_cols = ["N_starts_official", "official_starts_pct", "N_finish_time_official", "official_finish_time_pct"]
    annual = annual[rate_cols + source_cols]
    a_all = annual.reset_index()
    a_all["race_type"] = a_all["race_type"].astype(str)
    a_all["in_trend_tests"] = ~a_all["race_year"].isin(EXCLUDED_SEASONS)
    shown = a_all[["race_year", "race_type"] + rate_cols].copy()
    for c in ("DNF_rate", "inrace_rate", "finish_time_rate"):
        shown[c] = shown[c].round(4)                                       # rounded for display only
    out.append("\nRates in % of starters (rounded here for display; full precision in " + OUT_ANNUAL + "):")
    out.append("\n" + shown.to_string(index=False))
    write_csv(a_all.rename(columns={"N_starts": "starters", "N_DNF": "dnf", "DNF_rate": "dnf_rate_pct",
                                    "N_inrace": "inrace_dnf", "inrace_rate": "inrace_rate_pct",
                                    "N_finish_time": "finish_time_dnf", "finish_time_rate": "finish_time_rate_pct",
                                    "N_starts_official": "starters_official", "official_starts_pct": "starters_official_pct",
                                    "N_finish_time_official": "finish_time_dnf_official",
                                    "official_finish_time_pct": "finish_time_dnf_official_pct"}),
              OUT_ANNUAL, "RQ3 — annual DNF rates by season × event type (total, in-race and finish-time; % of starters, "
              "full precision; in_trend_tests = season used by the trend tests) and the official-source shares of the "
              "starters and of the finish-time DNFs (% of each, full precision; missing without finish-time DNFs)",
              float_format=CSV_FLOAT)

    # --- Trend tests ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 1b: TREND TESTS (in-race DNF rate, primary; total DNF rate, secondary; finish-time DNF rate)")
    out.append("=" * 60)
    out.append(f"\nSeasons: every season of the event type except {EXCLUDED_SEASONS[0]} (COVID-19 pandemic, much reduced")
    out.append(f"field) and the partial {EXCLUDED_SEASONS[1]} season.")
    out.append("Linear regression: OLS slope of the annual rate on calendar year; its 95% CI uses the t quantile with")
    out.append("n - 2 degrees of freedom, the distribution of its p-value.")
    out.append("Mann-Kendall: non-parametric test for monotonic trend (Mann 1945, Kendall 1975); Sen's slope =")
    out.append("median of the pairwise slopes against calendar year (scipy.stats.theilslopes; 2020 is excluded, so")
    out.append("the years are not equally spaced). The classification line is pymannkendall's label of the")
    out.append("two-sided test at alpha = 0.05.")
    out.append(f"Both are fitted on the unrounded annual rates ({OUT_ANNUAL}); full-precision statistics in {OUT_TESTS}.")
    test_rows = []
    for series, (col, slabel) in SERIES.items():
        out.append(f"\n--- {slabel[0].upper() + slabel[1:]} ---")
        for rt in ["im", "him"]:
            d = a_all[(a_all["race_type"] == rt) & a_all["in_trend_tests"]].sort_values("race_year")
            r = trend_tests(d["race_year"].to_numpy(), d[col].to_numpy(dtype=float))
            test_rows.append({"series": series, "race_type": rt, "seasons": "all", **r})
            out += test_lines(TYPE_LABELS[rt], slabel, r)

    # --- Highest and lowest annual rate of the in-race and total series; event with the highest in-race DNF share in
    # the season of the highest rate ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 1c: HIGHEST AND LOWEST ANNUAL RATE OF THE IN-RACE AND TOTAL SERIES; EVENT WITH THE HIGHEST")
    out.append("IN-RACE DNF SHARE IN THE SEASON OF THE HIGHEST RATE")
    out.append("=" * 60)
    out.append("\nSeasons of the trend tests (2020 and the partial 2026 excluded). In-race DNF = DNF without a")
    out.append("recorded finish time; the event's share is over its starters. The season rate without that event is a")
    out.append("rate of the same series.")
    for series in ("inrace", "total"):
        col, slabel = SERIES[series]
        out.append(f"\n{slabel[0].upper() + slabel[1:]}:")
        num = "is_inrace" if series == "inrace" else "is_dnf"
        for rt in ["im", "him"]:
            a = a_all[(a_all["race_type"] == rt) & a_all["in_trend_tests"]]
            top = a.loc[a[col].idxmax()]
            low = a.loc[a[col].idxmin()]
            year = int(top["race_year"])
            season = df[(df["race_type"] == rt) & (df["race_year"] == year)]
            ev = season.groupby("event_id", observed=True).agg(name=("event_name", "first"), starters=("is_dnf", "size"),
                                                               inrace=("is_inrace", "sum"))
            ev["share"] = 100 * ev["inrace"] / ev["starters"]
            top_ev = ev["share"].idxmax()
            e = ev.loc[top_ev]
            rest = season[season["event_id"] != top_ev]
            out.append(f"  {TYPE_LABELS[rt]}: highest annual rate {year} {top[col]:.4f}%; lowest annual rate "
                       f"{int(low['race_year'])} {low[col]:.4f}%; event with the highest in-race DNF share of starters in "
                       f"{year}: {e['name']} {int(e['inrace']):,} / {int(e['starters']):,} = {e['share']:.4f}%; season rate "
                       f"without it: {100 * rest[num].mean():.4f}% ({int(rest[num].sum()):,} / {len(rest):,})")

    # --- In-race DNF rate before and from 2010, and by period ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 1d: IN-RACE DNF RATE — TREND TESTS BEFORE AND FROM 2010 AND POOLED RATE BY PERIOD")
    out.append("=" * 60)
    out.append(f"\nTrend tests of the in-race DNF rate restricted to the seasons up to 2009 and to 2010-2025 ({EXCLUDED_SEASONS[0]} excluded),")
    out.append(f"as in section 1b (rows with seasons = '<first season>-<last season>' in {OUT_TESTS}).")
    col, slabel = SERIES["inrace"]
    for rt in ["im", "him"]:
        a = a_all[(a_all["race_type"] == rt) & a_all["in_trend_tests"]]
        for w_lo, w_hi in RESTRICTED_SEASONS:
            w_lo = int(a["race_year"].min()) if w_lo is None else w_lo
            d = a[a["race_year"].between(w_lo, w_hi)].sort_values("race_year")
            r = trend_tests(d["race_year"].to_numpy(), d[col].to_numpy(dtype=float))
            test_rows.append({"series": "inrace", "race_type": rt, "seasons": f"{w_lo}-{w_hi}", **r})
            out += test_lines(TYPE_LABELS[rt], f"{slabel}, seasons {w_lo}-{w_hi}", r)
    write_csv(pd.DataFrame(test_rows), OUT_TESTS,
              "RQ3 — trend tests of the annual in-race (primary), total (secondary) and finish-time DNF rates, seasons "
              f"{EXCLUDED_SEASONS[0]} and {EXCLUDED_SEASONS[1]} excluded (seasons = 'all'), and of the in-race rate within "
              "the seasons up to 2009 and over 2010-2025 (seasons = first and last season of the window) (OLS slope with "
              "t-based 95% CI, R², p; Mann-Kendall tau and p; Sen's slope vs calendar year)", float_format=CSV_FLOAT)

    out.append("\nPooled in-race DNF rate by period: in-race DNFs / starters summed over the seasons of the period")
    out.append(f"({EXCLUDED_SEASONS[0]} and {EXCLUDED_SEASONS[1]} excluded; the first period starts at the first season of the event type):")
    period_rows = []
    for rt in ["im", "him"]:
        a = a_all[(a_all["race_type"] == rt) & a_all["in_trend_tests"]]
        for p_lo, p_hi in PERIODS:
            p_lo = int(a["race_year"].min()) if p_lo is None else p_lo
            p = a[a["race_year"].between(p_lo, p_hi)]
            n_st, n_in = int(p["N_starts"].sum()), int(p["N_inrace"].sum())
            period_rows.append({"race_type": rt, "period": f"{p_lo}-{p_hi}", "first_season": int(p["race_year"].min()),
                                "last_season": int(p["race_year"].max()), "n_seasons": len(p), "starters": n_st,
                                "inrace_dnf": n_in, "inrace_rate_pct": n_in / n_st * 100})
            out.append(f"  {TYPE_LABELS[rt]} {p_lo}-{p_hi} ({len(p)} seasons): {n_in:,} / {n_st:,} = {n_in / n_st * 100:.4f}%")
    write_csv(pd.DataFrame(period_rows), OUT_PERIODS,
              f"RQ3 — pooled in-race DNF rate by period (in-race DNFs / starters over the seasons of the period; "
              f"{EXCLUDED_SEASONS[0]} and {EXCLUDED_SEASONS[1]} excluded; % full precision)", float_format=CSV_FLOAT)

    # --- Source of the starters and of the finish-time DNFs by season ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 1e: OFFICIAL-SOURCE SHARE OF THE STARTERS AND OF THE FINISH-TIME DNFs BY SEASON")
    out.append("=" * 60)
    out.append(f"\nofficial / all = share (%), by season and event type (full precision in {OUT_ANNUAL}):")
    out.append(f"  {'season':6s} {'type':4s} {'starters from the official source':>40s} {'finish-time DNFs from the official source':>44s}")
    for r in a_all.sort_values(["race_type", "race_year"]).itertuples(index=False):
        ft = ("—" if r.N_finish_time == 0 else
              f"{int(r.N_finish_time_official):,} / {int(r.N_finish_time):,} = {r.official_finish_time_pct:.4f}%")
        out.append(f"  {int(r.race_year):<6d} {r.race_type:4s} "
                   f"{f'{int(r.N_starts_official):,} / {int(r.N_starts):,} = {r.official_starts_pct:.4f}%':>40s} {ft:>44s}")
    for rt in ["im", "him", "all"]:
        a = a_all if rt == "all" else a_all[a_all["race_type"] == rt]
        n_ft, n_ft_off = int(a["N_finish_time"].sum()), int(a["N_finish_time_official"].sum())
        n_st, n_st_off = int(a["N_starts"].sum()), int(a["N_starts_official"].sum())
        lab = "both event types" if rt == "all" else TYPE_LABELS[rt]
        out.append(f"  all seasons, {lab}: starters from the official source {n_st_off:,} / {n_st:,} = "
                   f"{n_st_off / n_st * 100:.4f}%; finish-time DNFs from the official source {n_ft_off:,} / {n_ft:,} = "
                   f"{n_ft_off / n_ft * 100:.4f}%")

    # --- Annual DNF rate by sex ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 2: ANNUAL DNF RATE BY DISTANCE × SEX (total DNF rate)")
    out.append("=" * 60)

    annual_g = df.groupby(["race_year", "race_type", "gender"], observed=True)["is_dnf"].agg(N_starts="count", N_DNF="sum", DNF_rate="mean")
    annual_g["DNF_rate"] = (annual_g["DNF_rate"] * 100).round(4)
    # Only Male/Female, skip Unknown
    annual_g = annual_g.reset_index()
    annual_g = annual_g[annual_g["gender"].isin(["Male", "Female"])]
    out.append("\n" + annual_g.to_string(index=False))

    # --- Variation around COVID-19 ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3: VARIATION AROUND COVID-19 (2017-2019 vs 2021-2023; total DNF rate)")
    out.append("=" * 60)
    out.append("\nNote: descriptive comparison only.")
    out.append("Separating a COVID-specific effect from natural fluctuation is not possible.")
    out.append("See year-by-year data above for full context.")

    pre = df[df["race_year"].between(2017, 2019)]
    post = df[df["race_year"].between(2021, 2023)]

    for label, sub in [("Pre-COVID (2017-2019)", pre), ("Post-COVID (2021-2023)", post)]:
        g = sub.groupby(["race_type", "gender"], observed=True)["is_dnf"].agg(N_starts="count", N_DNF="sum", DNF_rate="mean")
        g["DNF_rate"] = (g["DNF_rate"] * 100).round(4)
        out.append(f"\n{label}:")
        out.append(g.to_string())

    # Delta
    out.append("\nDelta (post - pre) in DNF rate (pp):")
    for rt in ["im", "him"]:
        for g in ["Male", "Female"]:
            pre_rate = pre[(pre["race_type"] == rt) & (pre["gender"] == g)]["is_dnf"].mean() * 100
            post_rate = post[(post["race_type"] == rt) & (post["gender"] == g)]["is_dnf"].mean() * 100
            delta = post_rate - pre_rate
            out.append(f"  {rt.upper()} {g}: {pre_rate:.4f}% → {post_rate:.4f}% (Δ = {delta:+.4f} pp)")

    # --- The 2024-2025 seasons ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 4: THE 2024-2025 SEASONS (total and in-race DNF rates)")
    out.append("=" * 60)
    recent = shown[shown["race_year"].between(2024, 2025)]
    out.append("\n" + recent[["race_year", "race_type", "N_starts", "N_DNF", "DNF_rate", "N_inrace", "inrace_rate"]]
               .to_string(index=False))

    # --- Save ---
    output_text = "\n".join(out)
    (RESULTS_DIR / "temporal_trends.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(f"Saved to {RESULTS_DIR / 'temporal_trends.txt'}")
    print(output_text)


if __name__ == "__main__":
    main()
