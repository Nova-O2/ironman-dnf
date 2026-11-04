"""
RQ1: DNF rates by sex, age group, region, and race distance; the analysed population.
Produces: results/descriptives.txt (section 0: excluded records and analysed totals),
          descriptives_table1.csv (Table 1), descriptives_age_gender.csv (Figure 1),
          excluded_race_years.csv, shortened_editions.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_dataset, infer_gender, assign_continent, exclude_race_years, duplicate_overlap, brand_rule_match,
    event_course_table, shortened_editions, edition_inrace_share, reproducibility_header, write_csv, RESULTS_DIR,
    TOTAL_RECORDS, COURSE_COLUMNS, DUPLICATE_RACE_YEARS, NONSTANDARD_FORMAT_EVENTS, NONSTANDARD_FORMAT_RULES,
    NON_IRONMAN_BRANDS, NON_IRONMAN_EVENTS, EXCLUDED_EVENTS, EXCLUSION_LABELS, SHORTENED_DISTANCE, SHORTENED_TIME,
    STOPPED_EDITIONS, STOPPED_SHARE,
)

AGE_BINS_TABLE = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                  "50-54", "55-59", "60-64", "65-69", "70+"]
CSV_FLOAT = "%.10g"
STARTER_STATUS = ["FIN", "DNF", "DQ"]


def consolidate_70plus(age_bin: str) -> str:
    """Collapse every bin from 70 years upward into '70+' (Figure 1, RQ4)."""
    return "70+" if age_bin in ("70-74", "75-79", "80-84", "80+", "85-89", "85+", "90+", "70+") else age_bin


def map_age_bin(ag: str) -> str:
    """Map age_group codes to standardized 5-year bins for analysis.

    Uses original 5-year bins from the dataset (M18-24, M25-29, M30-34, etc.)
    rather than decade aggregation — higher resolution for DNF rate patterns.
    Edge cases (M80+, F70&+) mapped to their base bin (80+, 70+).
    """
    ag = str(ag).strip()
    if ag in ("MPRO", "FPRO"):
        return "PRO"
    # Standard M/F + age range: M40-44, F25-29, etc.
    if len(ag) >= 5 and ag[0] in ("M", "F") and "-" in ag[1:]:
        try:
            parts = ag[1:].split("-")
            lo = int(parts[0])
            hi = int(parts[1])
            return f"{lo}-{hi}"
        except (ValueError, IndexError):
            pass
    # Edge cases: M80+, F70&+, M85+, M90+
    if ag[0:1] in ("M", "F") and ("+" in ag or "&" in ag):
        try:
            num = int("".join(c for c in ag[1:] if c.isdigit()))
            return f"{num}+"
        except ValueError:
            pass
    return "Other"


def excluded_records_section(full: pd.DataFrame, kept: pd.DataFrame, course: pd.DataFrame) -> list[str]:
    """Section 0: race-years excluded before every analysis (by reason and event type), the overlap
    of the duplicated copies, the non-standard-format editions, the non-IRONMAN races, the
    stopped-edition rule and its distribution, the shortened editions (kept) and the analysed totals.
    Writes excluded_race_years.csv and shortened_editions.csv."""
    out = ["=" * 60, "SECTION 0: EXCLUDED RECORDS (race-years removed before every analysis)", "=" * 60]
    out.append("\nThe records of these race-years (event_ids) are dropped by _common.load_dataset, so every analysis")
    out.append("and a rerun on the deposited file use the same population. Reasons:")
    for k, v in EXCLUSION_LABELS.items():
        out.append(f"  {k}: {v}")
    out.append("Starters = FIN + DNF + DQ.")

    eid = full["event_id"].astype("string")
    fs = full["finish_status"].astype("string")
    exc = full[eid.isin(list(EXCLUDED_EVENTS)).fillna(False).to_numpy()].copy()
    exc["reason"] = exc["event_id"].astype(str).map(EXCLUDED_EVENTS)
    exc["is_starter"] = exc["finish_status"].astype(str).isin(STARTER_STATUS)
    exc["is_dnf"] = exc["finish_status"].astype(str) == "DNF"
    # in-race DNF = DNF without a recorded finish time (the quantity of the stopped-edition rule)
    exc["is_inrace_dnf"] = exc["is_dnf"] & ~(exc["overall_sec"].astype("Float64").fillna(0) > 0).to_numpy(dtype=bool)
    ev = exc.groupby("event_id", observed=True).agg(
        event_name=("event_name", "first"), source=("source", "first"), race_type=("race_type", "first"),
        race_year=("race_year", "first"), reason=("reason", "first"), records=("reason", "size"),
        starters=("is_starter", "sum"), dnf=("is_dnf", "sum"), inrace_dnf=("is_inrace_dnf", "sum"))
    ev["event_name"] = ev["event_name"].astype(str)
    ev["kept_event_id"] = [DUPLICATE_RACE_YEARS[e][0] if e in DUPLICATE_RACE_YEARS else "" for e in ev.index]
    ev = ev.reset_index().sort_values(["reason", "race_year", "event_name"])
    write_csv(ev[["event_id", "event_name", "source", "race_type", "race_year", "reason", "kept_event_id",
                  "records", "starters", "dnf", "inrace_dnf"]], "excluded_race_years.csv",
              "Analysed population — race-years excluded at load (one row per event_id; inrace_dnf = DNFs without a "
              "recorded finish time)")
    present = set(full["event_id"].dropna().astype(str))
    absent = sorted(set(EXCLUDED_EVENTS) - present)
    out.append(f"\nListed race-years absent from this file: {len(absent)}" + (f" ({absent})" if absent else ""))

    out.append("\nExcluded race-years by reason and event type:")
    out.append(f"  {'reason':14s} {'type':4s} {'race-years':>10s} {'records':>9s} {'starters':>9s} {'DNF':>7s}")
    for reason in list(EXCLUSION_LABELS):
        for rt in ["im", "him"]:
            g = ev[(ev["reason"] == reason) & (ev["race_type"].astype(str) == rt)]
            out.append(f"  {reason:14s} {rt:4s} {len(g):>10,} {int(g['records'].sum()):>9,} {int(g['starters'].sum()):>9,} "
                       f"{int(g['dnf'].sum()):>7,}")
    for rt in ["im", "him", "all"]:
        g = ev if rt == "all" else ev[ev["race_type"].astype(str) == rt]
        out.append(f"  {'total':14s} {rt:4s} {len(g):>10,} {int(g['records'].sum()):>9,} {int(g['starters'].sum()):>9,} "
                   f"{int(g['dnf'].sum()):>7,}")
    n_dup = int((ev["reason"] == "duplicate").sum())
    n_fmt = int(ev["reason"].isin(list(NONSTANDARD_FORMAT_RULES)).sum())
    n_non = int((ev["reason"] == "non_ironman").sum())
    n_stop = int((ev["reason"] == "stopped").sum())
    assert n_dup + n_fmt + n_non + n_stop == len(ev), "excluded race-year without a reason"
    out.append(f"Excluded race-years: {len(ev):,} (duplicated copies {n_dup}, non-standard formats {n_fmt}, "
               f"non-IRONMAN races {n_non}, stopped or wholesale-coded editions {n_stop})")
    out.append(f"Excluded records: {int(ev['records'].sum()):,}; excluded starters: {int(ev['starters'].sum()):,}; "
               f"excluded DNFs: {int(ev['dnf'].sum()):,}")

    # Duplicated race-years: overlap of the two copies and the completeness criteria of the rule
    ov = duplicate_overlap(full)
    out.append("\nDuplicated race-years — overlap of the two copies (finisher key = bib, overall, swim and cycling times")
    out.append("of records with an overall time > 0 and a bib) and the completeness criteria of the copy kept")
    out.append("(_common.DUPLICATE_RACE_YEARS: rule 1 subset, rule 2 event type contradicts finish times, rule 3 records >")
    out.append("finishers > split times, rule 4 series name):")
    for r in ov.itertuples(index=False):
        out.append(f"  DROPPED {r.dropped_name} [{r.dropped_event_id}]: records {r.dropped_records:,}, finishers "
                   f"{r.dropped_finishers:,}, DNF {r.dropped_dnf:,}, split times {r.dropped_split_times:,}")
        out.append(f"     KEPT {r.kept_name} [{r.kept_event_id}]: records {r.kept_records:,}, finishers "
                   f"{r.kept_finishers:,}, DNF {r.kept_dnf:,}, split times {r.kept_split_times:,}")
        out.append(f"     shared finisher keys {r.shared_finisher_keys:,} = {r.shared_pct_of_dropped:.4f}% of the dropped copy, "
                   f"{r.shared_pct_of_kept:.4f}% of the kept copy ({r.shared_pct_of_smaller:.4f}% of the smaller); {r.rule}")
    out.append(f"  minimum shared share of the smaller copy: {ov['shared_pct_of_smaller'].min():.4f}%")

    # Non-standard formats: list with the modal course of the finishers (official source)
    out.append("\nNon-standard-format editions (event-name rule; modal swim / cycling / run / total distance of the")
    out.append("finishers, km, when the edition has finishers with distances):")
    fmt = ev[ev["reason"].isin(list(NONSTANDARD_FORMAT_RULES))]
    fin = full[(fs == "FIN").fillna(False).to_numpy() & eid.isin(list(NONSTANDARD_FORMAT_EVENTS)).fillna(False).to_numpy()]
    def _mode(s: pd.Series) -> str:
        s = s.dropna().astype("float64").round(4)
        return f"{s.mode().min():g}" if len(s) else "—"
    for r in fmt.itertuples(index=False):
        f = fin[fin["event_id"].astype(str) == r.event_id]
        crs = "/".join(_mode(f[c]) for c in ["swim_distance_km", "bike_distance_km", "run_distance_km", "total_distance_km"])
        out.append(f"  {r.reason:12s} {r.event_name} ({r.race_type}, {r.source}): records {r.records:,}, starters {r.starters:,}, "
                   f"DNF {r.dnf:,}, finishers {len(f):,}; course {crs}")

    # Races of another brand recorded as IRONMAN® events: the brand rules and the races they match
    non = ev[ev["reason"] == "non_ironman"]
    names = full[["event_id", "event_name"]].dropna().drop_duplicates("event_id")
    out.append("\nNon-IRONMAN races (_common.NON_IRONMAN_EVENTS): races of another brand recorded as IRONMAN® events in the data, identified")
    out.append("by event-name rules (case-insensitive) for these brands: " + ", ".join(NON_IRONMAN_BRANDS)
               + f"; event names matched in the file, both sources: {int(names['event_name'].astype(str).map(brand_rule_match).notna().sum())} "
               "(load_dataset fails unless the rules match exactly the races listed):")
    for r in non.itertuples(index=False):
        out.append(f"  {NON_IRONMAN_EVENTS[r.event_id]}: {r.event_name} ({r.race_type}, {r.source}) [{r.event_id}]: records {r.records:,}, "
                   f"starters {r.starters:,}, DNF {r.dnf:,}")
    out.append(f"  total: {len(non)} races, records {int(non['records'].sum()):,}, starters {int(non['starters'].sum()):,}, "
               f"DNF {int(non['dnf'].sum()):,}")

    # Stopped or wholesale-coded editions: the rule and the distribution that justifies its threshold
    share = edition_inrace_share(exclude_race_years(full, keep_stopped=True))   # after exclusions 1-3
    thr = STOPPED_SHARE * 100
    out.append(f"\nStopped or wholesale-coded editions (_common.STOPPED_EDITIONS): editions of either source whose in-race DNFs "
               f"(no recorded finish time) are >= {thr:.0f}% of their starters, the rule evaluated on the editions left after the "
               "duplicates, the non-standard formats and the non-IRONMAN races are removed (load_dataset fails unless it flags "
               "exactly the editions listed):")
    for e in STOPPED_EDITIONS:
        if e in share.index:
            r = share.loc[e]
            out.append(f"  {r['event_name']} ({r['source']}) [{e}]: starters {int(r['starters']):,}, in-race DNFs "
                       f"{int(r['inrace_dnf']):,} ({r['inrace_share_pct']:.4f}%)")
    for scope, sel in [("all editions, both sources (the scope of the rule)", pd.Series(True, index=share.index)),
                       ("official-source editions", share["source"] == "official"),
                       ("secondary-source editions", share["source"] == "coachcox")]:
        x = share[sel]
        at_or_above = x["inrace_dnf"] / x["starters"] >= STOPPED_SHARE   # the rule of _common.check_stopped_rule
        below = x[~at_or_above]
        top = below["inrace_share_pct"].idxmax()
        out.append(f"  Distribution of the in-race DNF share of starters, {scope}: editions with starters {len(x):,}; at or above "
                   f"{thr:.0f}%: {int(at_or_above.sum())}; next-highest edition {below.loc[top, 'inrace_share_pct']:.4f}% "
                   f"({below.loc[top, 'event_name']}); 99.5th percentile of the editions below the threshold "
                   f"{below['inrace_share_pct'].quantile(0.995):.4f}% (99th {below['inrace_share_pct'].quantile(0.99):.4f}%, "
                   f"median {below['inrace_share_pct'].median():.4f}%)")

    # Shortened editions (kept)
    sh = shortened_editions(kept, course)
    write_csv(sh.reset_index()[["event_id", "event_name", "source", "race_type", "n_finishers", "median_finish_h",
                                "typical_median_h", "time_ratio", "basis", "total", "distance_share", "criterion"]],
              "shortened_editions.csv", "Analysed population — editions held over a shortened course (kept)",
              float_format=CSV_FLOAT)
    out.append("\nShortened editions (KEPT in every analysis): official-source event with its own course and a modal finisher")
    out.append(f"total distance below {SHORTENED_DISTANCE:.0%} of nominal (distance), or median finisher overall time below "
               f"{SHORTENED_TIME:.0%} of the typical")
    out.append("median of its event type, the median of the event medians (time; both sources), or an event name that")
    out.append("declares a partial course (name):")
    typ = sh.groupby("race_type")["typical_median_h"].first() if len(sh) else pd.Series(dtype=float)
    out.append("  typical median finisher time (h): " + ", ".join(f"{k} {v:.4f}" for k, v in typ.items()))
    for rt in ["im", "him"]:
        s = sh[sh["race_type"] == rt]
        out.append(f"  {rt.upper()}: {len(s)} editions (by criterion: " + ", ".join(f"{c} {n}" for c, n in s["criterion"].value_counts().items()) + ")")
        for e, r in s.iterrows():
            dist = f"{r['distance_share'] * 100:.1f}% of nominal distance" if pd.notna(r["distance_share"]) else "no distance"
            out.append(f"    {r['event_name']} ({r['source']}): median finish {r['median_finish_h']:.3f} h "
                       f"({r['time_ratio'] * 100:.1f}% of typical), {dist}; criterion: {r['criterion']}")
    out.append(f"Shortened editions: {len(sh)} (IM {int((sh['race_type'] == 'im').sum())}, HIM {int((sh['race_type'] == 'him').sum())})")
    no_swim = course[(course["basis"] == "own") & (course["swim"] == 0)]
    out.append(f"Editions held without a swim (official source, own course with a modal finisher swim distance of 0; kept, "
               f"not counted as shortened): {len(no_swim)} (IM {int((no_swim['race_type'] == 'im').sum())}, "
               f"HIM {int((no_swim['race_type'] == 'him').sum())})")

    # Analysed totals
    k_fs = kept["finish_status"].astype("string")
    k_eid = kept["event_id"].astype("string")
    out.append("\nAnalysed population after the exclusions:")
    out.append(f"  Records analysed: {len(kept):,} (of {len(full):,} in the file)")
    st_mask = k_fs.isin(STARTER_STATUS).fillna(False)
    for rt in ["im", "him"]:
        m = (kept["race_type"].astype("string") == rt).fillna(False)
        out.append(f"  {rt.upper()}: records {int(m.sum()):,}, starters {int((m & st_mask).sum()):,}, "
                   f"DNF {int((m & (k_fs == 'DNF').fillna(False)).sum()):,}")
    n_ev_file = full["event_id"].dropna().astype(str).nunique()
    n_ev_kept = k_eid.dropna().nunique()
    n_ev_starters = k_eid[st_mask.to_numpy()].dropna().nunique()
    ev_type = kept.dropna(subset=["event_id"]).drop_duplicates("event_id")
    out.append(f"  Race-years (event_ids) in the file: {n_ev_file:,}; excluded: {len(ev):,}; analysed: {n_ev_kept:,} "
               f"(IM {int((ev_type['race_type'].astype(str) == 'im').sum()):,}, HIM {int((ev_type['race_type'].astype(str) == 'him').sum()):,}; "
               f"official {int((ev_type['source'].astype(str) == 'official').sum()):,}, secondary source "
               f"{int((ev_type['source'].astype(str) == 'coachcox').sum()):,})")
    out.append(f"  Analysed race-years with at least one starter: {n_ev_starters:,}")
    names = full.dropna(subset=["event_id"]).drop_duplicates("event_id")
    key = names["event_name"].astype(str) + "|" + names["race_year"].astype(str) + "|" + names["race_type"].astype(str)
    dup_keys = key[key.duplicated(keep=False)]
    out.append(f"  Race-years as (event name, year, event type) in the file: {key.nunique():,}; event names that carry two "
               f"event_ids: " + (", ".join(sorted(set(dup_keys.str.split('|').str[0]))) or "none")
               + " (two races in the same year, April and November 2008)")
    return out


def main():
    cols = [
        "finish_status", "race_type", "age_group", "country", "overall_sec", "bike_sec",
        "event_id", "event_name", "source", "race_year", "bib",
        "swim_sec", "t1_sec", "t2_sec", "run_sec",
        "swim_distance_km", "bike_distance_km", "run_distance_km", "total_distance_km",
    ]
    print("Loading dataset (whole file; the exclusions are counted here)...")
    full = load_dataset(list(dict.fromkeys(cols + COURSE_COLUMNS)), exclude=False)
    df = exclude_race_years(full)
    course = event_course_table(df)
    out = []
    out.append(reproducibility_header("RQ1 — DNF Descriptives"))
    out.append("NOTE: Sections 1-4 include PROs and finish-time DNFs (DNFs with a recorded finish time; full population).")
    out.append("Section 3b (Figure 1) and the logistic regression (RQ4) exclude both.")
    out.append("")
    n_file = len(full)
    out.append(f"Records in the file read: {n_file:,} (reference count of the merged dataset file: {TOTAL_RECORDS:,})")
    out.append(f"Records in the excluded race-years (section 0): {n_file - len(df):,}; records analysed: {len(df):,}")
    out += [""] + excluded_records_section(full, df, course) + [""]

    df = infer_gender(df)
    n_total = len(df)
    n_nostatus = int(df["finish_status"].isna().sum())
    df = df[df["finish_status"].isin(["FIN", "DNF", "DNS", "DQ"])].copy()
    # Finish-time DNF = DNF with a recorded overall finish time (in-race DNF = without one)
    df["is_finish_time_dnf"] = (df["finish_status"] == "DNF") & df["overall_sec"].notna() & (df["overall_sec"] > 0)
    df["is_pro"] = df["age_group"].astype(str).isin(["MPRO", "FPRO"])

    out.append(f"Records without a finish status (excluded from every analysis): {n_nostatus:,} ({n_nostatus/n_total*100:.2f}% of the records analysed)")
    out.append(f"Records with a finish status in FIN/DNF/DNS/DQ: {len(df):,}")
    out.append("")

    # --- Overall finish status ---
    out.append("=" * 60)
    out.append("SECTION 1: OVERALL FINISH STATUS")
    out.append("=" * 60)

    status_all = df["finish_status"].value_counts()
    status_pct = df["finish_status"].value_counts(normalize=True) * 100
    out.append("\nAll records (% relative to total records including DNS):")
    out.append(pd.DataFrame({"N": status_all, "%": status_pct.round(1)}).to_string())

    for rt in ["im", "him"]:
        sub = df[df["race_type"] == rt]
        s = sub["finish_status"].value_counts()
        p = sub["finish_status"].value_counts(normalize=True) * 100
        out.append(f"\n{rt.upper()} only (n={len(sub):,}, % relative to all records):")
        out.append(pd.DataFrame({"N": s, "%": p.round(1)}).to_string())

    # DNF rates relative to STARTERS (excluding DNS) — used in manuscript
    out.append("\n--- DNF rates relative to STARTERS (excl DNS) ---")
    starters_all = df[df["finish_status"].isin(["FIN", "DNF", "DQ"])]
    for rt in ["im", "him"]:
        sub = starters_all[starters_all["race_type"] == rt]
        n_dnf = (sub["finish_status"] == "DNF").sum()
        rate = n_dnf / len(sub) * 100
        out.append(f"  {rt.upper()}: {n_dnf:,} / {len(sub):,} = {rate:.4f}%")

    # --- Starters only (exclude DNS) ---
    starters = df[df["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()
    starters["is_dnf"] = (starters["finish_status"] == "DNF").astype(int)

    # --- DNF rate by sex ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 2: DNF RATE BY SEX")
    out.append("=" * 60)

    for rt in ["im", "him", "all"]:
        sub = starters if rt == "all" else starters[starters["race_type"] == rt]
        g = sub.groupby("gender", observed=True)["is_dnf"].agg(N_starts="count", N_DNF="sum", DNF_rate="mean")
        g["DNF_rate"] = (g["DNF_rate"] * 100).round(4)
        out.append(f"\n{'ALL DISTANCES' if rt == 'all' else rt.upper()}:")
        out.append(g.to_string())

    # --- DNF rate by age group ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3: DNF RATE BY AGE GROUP")
    out.append("=" * 60)

    starters["age_bin"] = starters["age_group"].astype(str).apply(map_age_bin)

    for rt in ["im", "him"]:
        sub = starters[starters["race_type"] == rt]
        g = sub.groupby(["age_bin", "gender"], observed=True)["is_dnf"].agg(N_starts="count", N_DNF="sum", DNF_rate="mean")
        g["DNF_rate"] = (g["DNF_rate"] * 100).round(4)
        out.append(f"\n{rt.upper()} — DNF rate by age group (5-year bins) × sex:")
        out.append(g.sort_index().to_string())

    # --- Table 1 block: finish status by event type × sex ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3a: TABLE 1 BLOCK — FINISH STATUS BY EVENT TYPE × SEX")
    out.append("=" * 60)
    out.append("\nStarters = FIN + DNF + DQ. DNF rate = DNF / starters. DNS and DQ shares are relative")
    out.append("to all records of the event type (including DNS). Finish-time DNF share is relative to the")
    out.append("DNFs of the same cell. PRO rate = DNF / starters among MPRO (male) or FPRO (female).")
    out.append("Percentages are persisted at full precision in descriptives_table1.csv (rounded here for display).")
    t1_rows = []
    for rt in ["im", "him"]:
        all_rt = df[df["race_type"] == rt]
        for gender in ["Male", "Female", "Total"]:
            cell = all_rt if gender == "Total" else all_rt[all_rt["gender"] == gender]
            st = cell[cell["finish_status"].isin(["FIN", "DNF", "DQ"])]
            n_dnf = int((st["finish_status"] == "DNF").sum())
            n_ft = int(st["is_finish_time_dnf"].sum())
            pro = st[st["is_pro"]]
            pro_dnf = int((pro["finish_status"] == "DNF").sum())
            t1_rows.append({
                "race_type": rt, "gender": gender,
                "n_records": len(cell),
                "starters": len(st),
                "finishers": int((st["finish_status"] == "FIN").sum()),
                "dnf": n_dnf,
                "dnf_rate_pct": n_dnf / len(st) * 100 if len(st) else float("nan"),
                "dns": int((cell["finish_status"] == "DNS").sum()),
                "dns_pct_of_records": (cell["finish_status"] == "DNS").mean() * 100,
                "dq": int((st["finish_status"] == "DQ").sum()),
                "dq_pct_of_records": (cell["finish_status"] == "DQ").mean() * 100,
                "finish_time_dnf": n_ft,
                "finish_time_pct_of_dnf": n_ft / n_dnf * 100 if n_dnf else float("nan"),
                "pro_starters": len(pro),
                "pro_dnf": pro_dnf,
                "pro_dnf_rate_pct": pro_dnf / len(pro) * 100 if len(pro) else float("nan"),
            })
    t1 = pd.DataFrame(t1_rows)
    out.append("\n" + t1.round(4).to_string(index=False))
    write_csv(t1, "descriptives_table1.csv", "RQ1 — Table 1 block (finish status by event type × sex)", float_format=CSV_FLOAT)

    # --- Figure 1 block: age bin (70+ consolidated) × sex, two variants ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3b: FIGURE 1 BLOCK — DNF RATE BY AGE BIN (70+ CONSOLIDATED) × SEX")
    out.append("=" * 60)
    out.append("\nVariant 'all': starters of known sex whose age-group code falls in one of the eleven age bins")
    out.append("(professional categories and unrecognised codes fall outside them). Variant 'excl_pro_finish_time':")
    out.append("the same with finish-time DNFs removed — the population of Figure 1, which plots this variant.")
    out.append("The logistic regression starts from it and further drops athletes of unknown continent of origin")
    out.append("and starters at unmapped race locations.")
    t2_rows = []
    base = starters[starters["gender"].isin(["Male", "Female"])].copy()
    base["age_bin"] = base["age_bin"].apply(consolidate_70plus)
    # Starters of known sex whose age-group code fits none of the eleven bins (map_age_bin -> 'Other';
    # professional categories are counted apart): excluded from Figure 1 and from the regression.
    other = base[base["age_bin"] == "Other"]
    out.append("\nStarters of known sex whose age-group code fits none of the eleven age bins ('Other'; excluded")
    out.append("from Figure 1 and from the regression; professional categories not included):")
    for rt, label in [("im", "IRONMAN"), ("him", "IRONMAN 70.3")]:
        o = other[other["race_type"] == rt]
        out.append(f"  {label}: {len(o):,} (Female {int((o['gender'] == 'Female').sum()):,}, "
                   f"Male {int((o['gender'] == 'Male').sum()):,}; finish-time DNFs among them "
                   f"{int(o['is_finish_time_dnf'].sum()):,})")
    codes = other["age_group"].astype(str).value_counts()
    out.append("  codes: " + ", ".join(f"{k} ({v:,})" for k, v in codes.head(10).items())
               + (f", {len(codes) - 10} more" if len(codes) > 10 else ""))
    base = base[base["age_bin"].isin(AGE_BINS_TABLE)]
    for variant, frame in [("all", base), ("excl_pro_finish_time", base[~base["is_pro"] & ~base["is_finish_time_dnf"]])]:
        g = frame.groupby(["race_type", "age_bin", "gender"], observed=True)["is_dnf"].agg(["count", "sum"])
        for (rt, ab, gender), r in g.iterrows():
            t2_rows.append({"variant": variant, "race_type": rt, "age_bin": ab, "gender": gender,
                            "N_starts": int(r["count"]), "N_DNF": int(r["sum"]),
                            "DNF_rate_pct": r["sum"] / r["count"] * 100})
    t2 = pd.DataFrame(t2_rows)
    for variant in ["all", "excl_pro_finish_time"]:
        piv = t2[t2["variant"] == variant].pivot_table(index="age_bin", columns=["race_type", "gender"],
                                                       values="DNF_rate_pct").reindex(AGE_BINS_TABLE)
        out.append(f"\nDNF rate (%) — variant '{variant}':")
        out.append(piv.round(1).to_string())
    write_csv(t2, "descriptives_age_gender.csv", "RQ1 — Figure 1 block (age bin × sex × event type, two variants)",
              float_format=CSV_FLOAT)

    # --- Cycling time among age-group finishers (Discussion: hours of prior cycling) ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3c: CYCLING SPLIT AMONG AGE-GROUP FINISHERS (hours)")
    out.append("=" * 60)
    fin = df[(df["finish_status"] == "FIN") & ~df["is_pro"] & (df["bike_sec"].fillna(0) > 0)]
    for rt in ["im", "him"]:
        h = fin.loc[fin["race_type"] == rt, "bike_sec"].astype("float64") / 3600
        out.append(f"  {rt.upper()} (n = {len(h):,}): P10={h.quantile(.10):.3f} P25={h.quantile(.25):.3f} "
                   f"P50={h.quantile(.50):.3f} P75={h.quantile(.75):.3f} P90={h.quantile(.90):.3f}")

    # --- DNF rate by continent ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 4: DNF RATE BY ATHLETE ORIGIN (CONTINENTAL)")
    out.append("=" * 60)

    starters = assign_continent(starters)
    n_mapped = (starters["continent"] != "Unknown").sum()
    n_unknown = (starters["continent"] == "Unknown").sum()
    out.append(f"\nCoverage: {n_mapped:,} mapped ({n_mapped/len(starters)*100:.1f}%), {n_unknown:,} Unknown ({n_unknown/len(starters)*100:.1f}%)")
    out.append("Source: 'country' text field (athlete origin, not race location).")

    starters_mapped = starters[starters["continent"] != "Unknown"]
    g = starters_mapped.groupby(["continent", "race_type"], observed=True)["is_dnf"].agg(N_starts="count", N_DNF="sum", DNF_rate="mean")
    g["DNF_rate"] = (g["DNF_rate"] * 100).round(4)
    out.append("\nDNF rate by athlete continent × distance:")
    out.append(g.sort_values("N_starts", ascending=False).to_string())

    # --- Save ---
    output_text = "\n".join(out)
    (RESULTS_DIR / "descriptives.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(f"Saved to {RESULTS_DIR / 'descriptives.txt'}")
    print(output_text)


if __name__ == "__main__":
    main()
