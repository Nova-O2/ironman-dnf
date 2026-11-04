"""
Generate publication-ready tables in Markdown format.
Tables saved to manuscript/tables/ for Pandoc export.

Every value is read from the CSV results written by the analysis scripts
(notebooks/results/*.csv). No number is typed here, so the tables follow the
data through any change of the dataset or of the analysis.

Rounding: every printed value is formatted once, from a full-precision source, rounded half up
at the printed precision (_common.fmt_half_up; percentages of counts computed exactly by
_common.pct_half_up). No value is rounded twice.

Main tables:
  Table 1: Sample descriptives by event type × sex         <- descriptives_table1.csv, excluded_race_years.csv
                                                              (excluded events by reason, footnote)
  Table 2: Logistic regression ORs — IRONMAN® model            <- prediction_im_or.csv, prediction_summary.csv
  Table 3: Logistic regression ORs — IRONMAN® 70.3 model       <- prediction_him_or.csv, prediction_summary.csv
                                                                  (n, events and AUC-ROC of each model)
  (The age × sex DNF rates are shown in Figure 1 only.)

Supplementary tables:
  Table S1: Athlete origin × race location cross-tabulation    <- athlete_vs_venue_crosstab.csv
  Table S2: Home vs away DNF rate by continent                 <- athlete_vs_venue_home_away.csv
  Table S5–S8: sensitivity analyses S1–S4                      <- sensitivity_or.csv, sensitivity_summary.csv
  Table S4: written by pro_rank_analysis.py (bike_rank bands)
  Table S3: run-segment dropout hazard by band + sensitivities <- dropout_hazard_bands.csv, dropout_hazard_contrasts.csv,
                                                                  dropout_hazard_cohort_seasons.csv (seasons of the
                                                                  years_le_2019 cohort), dropout_point_t2.csv (T2 limit)
"""

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import RESULTS_DIR, TABLE_DIR, SEED, STOPPED_SHARE, fmt_half_up, pct_half_up
from descriptives import AGE_BINS_TABLE


OR_FOOTNOTE = ("\n*OR, odds ratio; CI, confidence interval; AUC-ROC, area under the receiver operating characteristic curve. "
               "CIs and p-values from standard errors clustered by event (race-year).*\n")
OUTCOME = "in-race DNF (DNF without a recorded finish time) versus finisher or disqualified"
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
CONT_SHORT = {"North America": "N. America", "South America": "S. America"}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name, comment="#")


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else fmt_half_up(p, 3)


def fmt_n(n) -> str:
    return f"{int(n):,}"


def write(name: str, md: str) -> None:
    (TABLE_DIR / name).write_text(md, encoding="utf-8", newline="\n")
    print(f"  → {name}")


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------
def table1():
    t = read("descriptives_table1.csv").set_index(["race_type", "gender"])
    exc = read("excluded_race_years.csv")
    n_dup = int((exc["reason"] == "duplicate").sum())
    n_non = int((exc["reason"] == "non_ironman").sum())
    n_stop = int((exc["reason"] == "stopped").sum())
    n_fmt = len(exc) - n_dup - n_non - n_stop   # olympic_5150, aquabike, bike_run, do_not_use

    def cell(rt, g, col):
        return t.loc[(rt, g), col]

    def unknown_sex(rt, col):
        """Athletes of unknown sex in a Total: Total minus Male minus Female."""
        return int(cell(rt, "Total", col) - cell(rt, "Male", col) - cell(rt, "Female", col))

    def mark(rt, g, col):
        """Dagger on every Total that includes athletes of unknown sex."""
        return "†" if g == "Total" and unknown_sex(rt, col) > 0 else ""

    def n_pct(rt, g, n_col, den_col):
        """Count and its percentage of den_col, computed exactly from the two counts."""
        return f"{fmt_n(cell(rt, g, n_col))}{mark(rt, g, n_col)} ({pct_half_up(cell(rt, g, n_col), cell(rt, g, den_col))}%)"

    unk_st = {rt: unknown_sex(rt, "starters") for rt in ("im", "him")}
    unk_fin = {rt: unknown_sex(rt, "finishers") for rt in ("im", "him")}
    row = lambda label, f: f"| **{label}** | " + " | ".join(f(rt, g) for rt in ("im", "him") for g in ("Male", "Female", "Total")) + " |"
    lines = [
        "**Table 1.** Sample characteristics and finish status distribution by event type and sex for IRONMAN® and IRONMAN® 70.3 events (2002–2026, final season partial).",
        "",
        "| | Full | | | Half | | |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| | **Male** | **Female** | **Total** | **Male** | **Female** | **Total** |",
        row("Starters", lambda rt, g: fmt_n(cell(rt, g, "starters")) + mark(rt, g, "starters")),
        row("Finishers", lambda rt, g: fmt_n(cell(rt, g, "finishers")) + mark(rt, g, "finishers")),
        row("DNF", lambda rt, g: n_pct(rt, g, "dnf", "starters")),
        row("DNS", lambda rt, g: n_pct(rt, g, "dns", "n_records") if g == "Total" else "—"),
        row("DQ", lambda rt, g: n_pct(rt, g, "dq", "n_records") if g == "Total" else "—"),
        row("Finish-time DNF", lambda rt, g: n_pct(rt, g, "finish_time_dnf", "dnf")),
        row("PRO DNF rate", lambda rt, g: f"{pct_half_up(cell(rt, g, 'pro_dnf'), cell(rt, g, 'pro_starters'))}%" if g != "Total" else "—"),
        "",
        "Full = IRONMAN® (3.86 km swim, 180.25 km cycle, 42.20 km run); Half = IRONMAN® 70.3 (1.93 km swim, 90.12 km cycle, 21.10 km run).",
        f"†Totals include athletes of unknown sex (starters: IRONMAN® n = {unk_st['im']:,}, of whom {unk_fin['im']:,} finishers; "
        f"IRONMAN® 70.3 n = {unk_st['him']:,}, of whom {unk_fin['him']:,} finishers), so Male + Female does not add up to the Total.",
        "DNF rates computed as DNF / Starters (excluding DNS). DNS and DQ percentages are relative to all records with a finish status (including DNS).",
        "Finish-time DNF = DNF with a recorded overall finish time; percentages are relative to total DNFs for that event × sex. PRO DNF rate = DNF / starters among professional athletes (MPRO, FPRO).",
        f"Excludes {n_dup} duplicated events, {n_fmt} events in non-standard formats, {n_non} races of another brand recorded as IRONMAN® events, and {n_stop} events stopped or recorded wholesale as DNF (Methods, Study Population).",
        "",
    ]
    write("Table1.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# Tables 2 and 3 (and the OR body shared with S5–S8)
# ---------------------------------------------------------------------------
def var_label(v: str, with_ref: bool = True) -> str | None:
    if v == "female":
        return "Female (ref: Male)"
    if v.startswith("age_"):
        ab = v[4:].replace("-", "–")
        return f"Age {ab} (ref: 30–34)" if (ab == "18–24" and with_ref) else f"Age {ab}"
    if v.startswith("ath_"):
        c = v[4:].replace("_", " ")
        c = CONT_SHORT.get(c, c)
        return f"Athlete: {c} (ref: N. America)" if (c == "Africa" and with_ref) else f"Athlete: {c}"
    if v.startswith("race_") and v != "race_year":
        c = v[5:].replace("_", " ")
        c = CONT_SHORT.get(c, c)
        return f"Race: {c} (ref: N. America)" if (c == "Africa" and with_ref) else f"Race: {c}"
    if v == "race_year":
        return "Year (per year)"
    if v == "is_him":
        return "IRONMAN® 70.3 (ref: IRONMAN®)"
    if v == "is_pro":
        return "Professional (ref: Age-group)"
    return None  # const


def or_rows(df: pd.DataFrame, include_age: bool = True) -> list[str]:
    """Markdown rows in canonical order: female, extra flags, ages, athlete, race, year."""
    df = df.set_index("variable")
    order = ["female", "is_him", "is_pro"]
    if include_age:
        order += [f"age_{ab}" for ab in AGE_BINS_TABLE if ab != "30-34"]
    order += [f"ath_{c.replace(' ', '_')}" for c in CONTINENTS if c != "North America"]
    order += [f"race_{c.replace(' ', '_')}" for c in CONTINENTS if c != "North America"]
    order += ["race_year"]
    rows = []
    for v in order:
        if v not in df.index:
            continue
        r = df.loc[v]
        rows.append(f"| {var_label(v)} | {fmt_half_up(r['OR'], 3)} | {fmt_half_up(r['CI_low'], 3)}–{fmt_half_up(r['CI_high'], 3)} | {fmt_p(r['p'])} |")
    return rows


def table_main_or(rt: str, number: int, caption: str, fname: str):
    s = read("prediction_summary.csv").set_index("race_type").loc[rt]
    lines = [
        f"**Table {number}.** {caption.format(n=fmt_n(s['n']), k=fmt_n(s['n_clusters']))}",
        "",
        "| Variable | OR | 95% CI | p |",
        "|---|---:|---|---|",
        *or_rows(read(f"prediction_{rt}_or.csv")),
        "| | | | |",
        f"| **AUC-ROC** | **{fmt_half_up(s['auc'], 3)}** | | |",
        f"| **n** | **{fmt_n(s['n'])}** | | |",
        OR_FOOTNOTE,
    ]
    write(fname, "\n".join(lines))


def table2():
    table_main_or("im", 2, f"Logistic regression odds ratios for independent risk factors associated with {OUTCOME} in IRONMAN® events (n = {{n}} age-group starters; {{k}} events). Professional athletes, finish-time DNFs (DNFs with a recorded finish time), and athletes with unknown sex, an age-group code that fits none of the 5-year bins, unmapped continent of origin or unmapped race location excluded. Reference categories: male, age 30–34, North American athlete origin, North American race location.", "Table2.md")


def table3():
    table_main_or("him", 3, f"Logistic regression odds ratios for independent risk factors associated with {OUTCOME} in IRONMAN® 70.3 events (n = {{n}} age-group starters; {{k}} events). Exclusions and reference categories as in Table 2.", "Table3.md")


# ---------------------------------------------------------------------------
# Tables S1 and S2
# ---------------------------------------------------------------------------
def table_s1():
    ct = read("athlete_vs_venue_crosstab.csv")
    ct = ct[ct["N_starts"] >= 1000].set_index(["athlete_continent", "event_continent"])
    lines = [
        "**Table S1.** DNF rates (%) by athlete continent of origin (rows) and race location continent (columns). Both event types pooled.",
        "",
        "| Athlete origin (rows) / race location (columns) | " + " | ".join(CONT_SHORT.get(c, c) for c in CONTINENTS) + " |",
        "|---|" + "---:|" * len(CONTINENTS),
    ]
    for a in CONTINENTS:
        cells = []
        for e in CONTINENTS:
            cells.append(pct_half_up(ct.loc[(a, e), "N_DNF"], ct.loc[(a, e), "N_starts"]) if (a, e) in ct.index else "—")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", "*Values are DNF rates (%). Only combinations with n ≥ 1,000 starters shown. — = fewer than 1,000 starters.*", ""]
    write("TableS1.md", "\n".join(lines))


def table_s2():
    ha = read("athlete_vs_venue_home_away.csv").set_index(["athlete_continent", "is_home"])
    lines = [
        "**Table S2.** DNF rates for athletes competing at home (athlete continent matches race continent) versus away, by continent of origin. Both event types pooled.",
        "",
        "| Athlete origin | Home (%) | n Home | Away (%) | n Away | Δ (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in CONTINENTS + ["Overall"]:
        h, w = ha.loc[(a, "Home")], ha.loc[(a, "Away")]
        # Δ computed exactly from the counts, then rounded once
        delta = (Decimal(int(h["N_DNF"])) * 100 / Decimal(int(h["N_starts"]))
                 - Decimal(int(w["N_DNF"])) * 100 / Decimal(int(w["N_starts"])))
        d_txt = fmt_half_up(abs(delta), 1)
        sign = "" if d_txt == "0.0" else ("−" if delta < 0 else "+")
        vals = [pct_half_up(h["N_DNF"], h["N_starts"]), fmt_n(h["N_starts"]), pct_half_up(w["N_DNF"], w["N_starts"]),
                fmt_n(w["N_starts"]), f"{sign}{d_txt}"]
        if a == "Overall":
            vals = [f"**{v}**" for v in vals]
            a = "**Overall**"
        lines.append(f"| {a} | " + " | ".join(vals) + " |")
    lines += ["", "*Home = athlete continent matches race continent. Includes both event types. Δ = Home − Away (percentage points), computed from the counts before rounding.*", ""]
    write("TableS2.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# Tables S5–S8 (sensitivity analyses)
# ---------------------------------------------------------------------------
def sens_block(analysis: str, model_contains: str, label: str, ors: pd.DataFrame, summ: pd.DataFrame) -> list[str]:
    m = summ[(summ["analysis"] == analysis) & summ["model"].str.startswith(model_contains)]
    assert len(m) == 1, (analysis, model_contains, len(m))
    m = m.iloc[0]
    body = ors[(ors["analysis"] == analysis) & (ors["model"] == m["model"])]
    rows = or_rows(body)   # all fitted coefficients, age groups included
    head = (f"| **{label}** (n = {fmt_n(m['n'])}; DNF = {fmt_n(m['n_dnf'])} [{pct_half_up(m['n_dnf'], m['n'])}%]; "
            f"AUC-ROC = {fmt_half_up(m['auc'], 3)}; {fmt_n(m['n_clusters'])} events) | | | |")
    return [head] + rows


def table_sens(fname: str, caption: str, blocks: list[tuple[str, str, str]]):
    ors, summ = read("sensitivity_or.csv"), read("sensitivity_summary.csv")
    lines = [caption, "", "| Variable | OR | 95% CI | p |", "|---|---:|---|---|"]
    for analysis, model_contains, label in blocks:
        lines += sens_block(analysis, model_contains, label, ors, summ)
    lines.append(OR_FOOTNOTE)
    write(fname, "\n".join(lines))


def table_s5():
    table_sens("TableS5.md",
               "**Table S5.** Sensitivity analysis: logistic regression with finish-time DNFs included in the outcome (DNF, with or without a recorded finish time, versus finisher or disqualified). Full = IRONMAN®; Half = IRONMAN® 70.3. Reference categories as in Tables 2–3.",
               [("S1", "IM —", "Full"), ("S1", "HIM —", "Half")])


def table_s6():
    summ = read("sensitivity_summary.csv")
    m = summ[summ["analysis"] == "S2"].iloc[0]
    table_sens("TableS6.md",
               f"**Table S6.** Sensitivity analysis: combined-event model (IRONMAN® + IRONMAN® 70.3) with event type as a covariate, fitted on a 20% simple random sample of the pooled starters (seed = {SEED}). n = {fmt_n(m['n'])}; DNF = {fmt_n(m['n_dnf'])} ({pct_half_up(m['n_dnf'], m['n'])}%); AUC-ROC = {fmt_half_up(m['auc'], 3)}. Outcome and reference categories as in Tables 2–3.",
               [("S2", "Combined", "Combined")])


def table_s7():
    table_sens("TableS7.md",
               f"**Table S7.** Sensitivity analysis: logistic regression excluding the 2020 and 2021 seasons (COVID-19 pandemic period). Full = IRONMAN®; Half = IRONMAN® 70.3. Outcome and reference categories as in Tables 2–3.",
               [("S3", "IM —", "Full"), ("S3", "HIM —", "Half")])


def table_s8():
    table_sens("TableS8.md",
               f"**Table S8.** Sensitivity analysis: logistic regression including professional athletes with a professional status indicator. Full = IRONMAN®; Half = IRONMAN® 70.3. Outcome and reference categories as in Tables 2–3; professional athletes, who have no age group, are coded in the reference age group (30–34).",
               [("S4", "IM —", "Full"), ("S4", "HIM —", "Half")])


# ---------------------------------------------------------------------------
# Table S3 — run-segment dropout hazard
# ---------------------------------------------------------------------------
SENS_LABELS = {
    "main": "Main analysis (events with ≥ 30 run-segment DNFs)",
    "min_dnf_20": "Events with ≥ 20 run-segment DNFs",
    "min_dnf_50": "Events with ≥ 50 run-segment DNFs",
    "exclude_2026": "Excluding the 2026 season",
    "years_le_2019": "Seasons up to 2019",   # replaced by the seasons the cohort covers (season_label)
    "years_ge_2021": "Seasons from 2021",
    "at_risk_without_finish_time": "Finish-time DNFs removed from the at-risk set",
    "start_mat_0.5": "Start of the run within 0.5 km of cycling completion",
    "start_mat_1.5": "Start of the run within 1.5 km of cycling completion",
    "strict": "Strict rule: no DNF recorded at the cycling-completion distance enters the run",
    "t2_no_limit": "T2 rule without the duration limit",
    "including_stopped": "Stopped events kept",
}


def season_label(seasons: pd.DataFrame, analysis: str, rt: str) -> str:
    """'Seasons first–last' of a cohort, from the persisted seasons of its events (dropout_hazard_cohort_seasons.csv)."""
    s = seasons[(seasons["analysis"] == analysis) & (seasons["race_type"] == rt)]
    first, last = int(s["first_season"].iloc[0]), int(s["last_season"].iloc[0])
    return f"Season {first}" if first == last else f"Seasons {first}–{last}"


def table_s3():
    bands = read("dropout_hazard_bands.csv")
    con = read("dropout_hazard_contrasts.csv")
    seasons = read("dropout_hazard_cohort_seasons.csv")
    t2 = read("dropout_point_t2.csv").set_index("race_type")
    t2_min = {rt: fmt_half_up(t2.loc[rt, "t2_limit_min"], 1) for rt in ("im", "him")}
    ratio_ci = lambda r: f"{fmt_half_up(r['ratio'], 2)} ({fmt_half_up(r['ci_low'], 2)}–{fmt_half_up(r['ci_high'], 2)})"
    under = lambda r: " (underpowered)" if str(r["underpowered"]) == "True" else ""
    n_under = int(con["underpowered"].astype(str).eq("True").sum())
    main_bands = bands[bands["analysis"] == "main"]
    n_no_cp = {rt: int(main_bands.loc[main_bands["race_type"] == rt, "n_events_excluded_no_checkpoint"].iloc[0])
               for rt in ("im", "him")}
    lines = [
        "**Table S3.** Interval-censored dropout hazard along the running segment (in-race DNFs per 10,000 athlete-km; official-source events). Panel A: pooled hazard by band of interval start, main analysis. Panel B: hazard in the interval from the start of the run to the first checkpoint versus the remainder of the run, and per-kilometre contrast (km 0–3 versus km 3 to the end), with the 95% confidence interval from a cluster bootstrap over events (1,000 replicates), for the main analysis and the sensitivity analyses.",
        "",
        "**A. Pooled hazard by band**",
        "",
        "| Event type | Band of interval start (km) | Events | Intervals | DNFs | Athlete-km | Hazard per 10,000 athlete-km |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for rt, name in (("im", "IRONMAN®"), ("him", "IRONMAN® 70.3")):
        b = bands[(bands["race_type"] == rt) & (bands["analysis"] == "main")]
        for _, r in b.iterrows():
            lines.append(f"| {name} | {r['band_start_km']} | {fmt_n(r['n_events'])} | {fmt_n(r['n_intervals'])} | {fmt_n(r['dnf'])} | "
                         f"{fmt_half_up(r['athlete_km'], 0, thousands=True)} | {fmt_half_up(r['hazard_per_1e4_athlete_km'], 1)} |")
    lines += [
        "",
        "**B. First interval versus the remainder of the run, and per-kilometre contrast (km 0–3 versus km 3 to the end)**",
        "",
        "| Event type | Analysis | Events (start of the run observed) | First interval | Remainder | Ratio (95% CI) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for rt, name in (("im", "IRONMAN®"), ("him", "IRONMAN® 70.3")):
        c = con[(con["race_type"] == rt) & (con["contrast"] == "first_interval_vs_rest_between")]
        c = c.assign(_order=c["analysis"].map({k: i for i, k in enumerate(SENS_LABELS)})).sort_values("_order")
        for _, r in c.iterrows():
            lab = (season_label(seasons, r["analysis"], rt) if r["analysis"] == "years_le_2019"
                   else SENS_LABELS.get(r["analysis"], r["analysis"]))
            lines.append(f"| {name} | {lab} | {fmt_n(r['n_events'])} ({fmt_n(r['n_events_with_start_mat'])}) | "
                         f"{fmt_half_up(r['h_first_per_1e4'], 1)} | {fmt_half_up(r['h_rest_per_1e4'], 1)} | {ratio_ci(r)}{under(r)} |")
        for analysis, prefix in (("main", "Main analysis"), ("strict", "Strict rule"),
                                 ("t2_no_limit", "T2 rule without the duration limit"),
                                 ("including_stopped", "Stopped events kept")):
            w = con[(con["race_type"] == rt) & (con["contrast"] == "first_interval_vs_rest_within") & (con["analysis"] == analysis)].iloc[0]
            lines.append(f"| {name} | {prefix}, events in which the start of the run was observed (within-event comparison) | {fmt_n(w['n_events'])} ({fmt_n(w['n_events_with_start_mat'])}) | "
                         f"{fmt_half_up(w['h_first_per_1e4'], 1)} | {fmt_half_up(w['h_rest_per_1e4'], 1)} | {ratio_ci(w)}{under(w)} |")
            k = con[(con["race_type"] == rt) & (con["contrast"] == "curve_km0-3_vs_km3-end_excl_terminal") & (con["analysis"] == analysis)].iloc[0]
            lines.append(f"| {name} | {prefix}, per-kilometre curve, km 0–3 versus km 3 to the end (terminal bins excluded) | {fmt_n(k['n_events'])} | "
                         f"{fmt_half_up(k['h_first_per_1e4'], 1)} | {fmt_half_up(k['h_rest_per_1e4'], 1)} | {ratio_ci(k)}{under(k)} |")
    lines += [
        "",
        "*The run of each event starts at its cycling-completion distance (the modal swim + cycling distance of its finishers) and ends at its finish distance. Run-segment DNFs are in-race DNFs recorded beyond the cycling completion and before the finish distance, plus those recorded at the cycling-completion distance with a T2 split of at most the 99th percentile of the T2 splits of the official-source finishers of their event type "
        f"({t2_min['im']} min in IRONMAN®, {t2_min['him']} min in IRONMAN® 70.3), "
        "who crossed the timing mat at the transition exit and entered the run at km 0 (T2 rule); the T2 rule without the duration limit also counts those with a longer T2 split, and the strict rule counts none of the DNFs recorded at the cycling-completion distance. The first interval runs from the start of the run (km 0, observed in an event when at least one such athlete withdrew there, or a checkpoint within 1 km of the cycling completion) to the first checkpoint at or beyond 1 km; events in which the start of the run was not observed contribute only to the remainder in the between-event comparison, and events without a checkpoint at or beyond 1 km (0.5 or 1.5 km in the sensitivity analyses that move the start of the run) are excluded from both first-interval comparisons and kept in the per-kilometre curve. "
        f"Panel A also excludes the events without a checkpoint at or beyond 1 km ({n_no_cp['im']} IRONMAN® and {n_no_cp['him']} IRONMAN® 70.3 events). "
        "Hazard = DNFs whose last recorded checkpoint is the interval start ÷ (athletes at risk × interval width in km). Athletes at risk = finishers + finish-time DNFs + run-segment DNFs recorded at or beyond the interval start. The per-kilometre contrast excludes each event's terminal bin (the last 1-km bin of its run, which ends at or contains its finish). The recorded distances of events whose finishers' distances do not form a usable course are not used. "
        f"Events stopped or recorded wholesale as DNF (in-race DNFs in at least {STOPPED_SHARE * 100:.0f}% of starters) are excluded except in the sensitivity analysis that keeps them. A contrast based on fewer than 10 events in which the start of the run was observed is flagged as underpowered" + ("; none was.*" if n_under == 0 else f" ({n_under} contrasts, marked in the table).*"),
        "",
    ]
    write("TableS3.md", "\n".join(lines))


def main():
    print("Generating publication-ready tables...")
    table1()
    table2()
    table3()
    table_s1()
    table_s2()
    table_s5()
    table_s6()
    table_s7()
    table_s8()
    table_s3()
    print(f"\nAll tables saved to {TABLE_DIR} (Table S4 is written by pro_rank_analysis.py)")


if __name__ == "__main__":
    main()
