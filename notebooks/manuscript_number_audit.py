"""Audit every quantitative claim of the manuscript against notebooks/results/.

Modelled on the number audit of the companion data descriptor. Each
claim is registered with the value as printed in the manuscript, the tolerance
it deserves, where it appears, and an extractor that reads the current value
from the persisted results (txt via regex, csv via pandas). The script never
touches the dataset: if a number cannot be read from results/, it is not
traceable and the claim fails by construction.

Status per claim:
  ok    printed value within tolerance of the persisted value (a printed bound, Bound('<', 0.001) or
        Bound('>', 99), is ok when the persisted value satisfies the inequality at full precision)
  FAIL  the manuscript and the results disagree -> rewrite the prose or regenerate
  NEW   claim registered before its printed value exists (printed = None); the
        current value is listed so that the prose can be written from it
  SKIP  the claim reads an external source that is absent here (the companion data
        paper's QA outputs, or results/deposit_rerun_audit.csv); listed separately
  ERROR the extractor failed (file missing in results/, pattern not found)

Printed values below are those of the current manuscript over dataset v4.0.0. Whenever the prose changes a number, re-register it here. Exit
status is non-zero while any FAIL or ERROR remains, or when the audited results were not all produced by the
current analysis code on one input file (stale or mixed results, an ERROR; a '# Dataset:' line that is 'mixed' or
'absent' counts as mixed), so the audit can serve as a gate; the Makefile does not run it
(`make all` only builds the documents): run it as step 11 of the pipeline (notebooks/README.md) before building.
The CSV header records the dataset of the audited results (the '# Dataset:' line shared by every results file, or
'mixed' / 'absent'), the audit-code fingerprint and the analysis-code fingerprint of the audited results.

Claims about the dataset's construction (source shares, cross-source agreement,
internal consistency, race-years) are read from the companion data paper's
quality-assessment outputs for version 4.0.0 (COMPANION_RESULTS: the folder named by the environment
variable IRONMAN_DNF_COMPANION_RESULTS, or else notebooks/results_v4 of a clone of the public repository
ironman-data next to this project, or of the companion paper's working copy ironman-data-paper; these claims
are SKIP when the folder is absent); they are not recomputed here because the raw per-source files live with
the companion paper.

Claims of the Data availability statement (deposit_*) read results/deposit_rerun_audit.csv: this audit run on
the results of a rerun of the pipeline on the deposited file (notebooks/README.md, Reproduction check). The file
backs those claims only when it was produced on results of the deposited file (its '# Dataset:' line names
ironman_dataset_deidentified.csv), under the current analysed population, by the current audit code (the '# Audit
code:' fingerprint, sha256 of notebooks/*.py except generate_figures.py, that this script writes into its CSV
header) and on results produced by the current analysis code (the '# Audited results, analysis code:' line, read
from the headers of the audited results); otherwise every deposit claim is ERROR. On the deposit only the record
counts of DEPOSIT_RECORD_COUNTS may differ, each downwards by at most the omitted records; every other registered
claim must keep its value at the printed precision, and every bound claim must still hold
(deposit_changed_estimates). Claims that read the companion QA outputs and were SKIP in the deposit audit are not
compared.

Run:  python3 notebooks/manuscript_number_audit.py
"""

import math
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import RESULTS_DIR, CONTINENT_MAP, reproducibility_header, code_sha256, AUDIT_CODE_EXCLUDE, ANALYSIS_CODE_EXCLUDE

OUT_CSV = RESULTS_DIR / "manuscript_number_audit.csv"
SIBLINGS = Path(__file__).resolve().parents[2]   # the folder that holds this project and the companion repository


def companion_results() -> Path:
    """Folder of the companion data descriptor's QA outputs on dataset v4.0.0: IRONMAN_DNF_COMPANION_RESULTS when set,
    otherwise the first that exists of ../ironman-data/notebooks/results_v4 (a clone of the public repository next to
    this project) and ../ironman-data-paper/notebooks/results_v4 (the companion paper's working copy); the first of
    the two when neither exists (claims reading it are then SKIP)."""
    if os.environ.get("IRONMAN_DNF_COMPANION_RESULTS"):
        return Path(os.environ["IRONMAN_DNF_COMPANION_RESULTS"]).resolve()
    candidates = [SIBLINGS / repo / "notebooks" / "results_v4" for repo in ("ironman-data", "ironman-data-paper")]
    return next((c for c in candidates if c.exists()), candidates[0])


COMPANION_RESULTS = companion_results()


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
_TXT_CACHE: dict[str, str] = {}


def txt(name: str) -> str:
    if name not in _TXT_CACHE:
        _TXT_CACHE[name] = (RESULTS_DIR / name).read_text(encoding="utf-8")
    return _TXT_CACHE[name]


def grab(name: str, pattern: str, group: int = 1, flags: int = re.M) -> float:
    """First regex match in a results txt, as float (commas stripped)."""
    m = re.search(pattern, txt(name), flags)
    if not m:
        raise LookupError(f"{name}: pattern not found: {pattern!r}")
    return float(m.group(group).replace(",", ""))


def grab_section(name: str, start: str, pattern: str, group: int = 1, after: str = "") -> float:
    """Regex match restricted to the text after the first occurrence of `start`
    (itself searched after the first occurrence of `after`, when given)."""
    body = txt(name)
    if after:
        j = body.find(after)
        if j < 0:
            raise LookupError(f"{name}: marker not found: {after!r}")
        body = body[j:]
    i = body.find(start)
    if i < 0:
        raise LookupError(f"{name}: section marker not found: {start!r}")
    m = re.search(pattern, body[i:], re.M)
    if not m:
        raise LookupError(f"{name} after {start!r}: pattern not found: {pattern!r}")
    return float(m.group(group).replace(",", ""))


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name, comment="#")


def t1(rt: str, gender: str, col: str) -> float:
    d = csv("descriptives_table1.csv").set_index(["race_type", "gender"])
    return float(d.loc[(rt, gender), col])


def pred(rt: str, col: str) -> float:
    return float(csv("prediction_summary.csv").set_index("race_type").loc[rt, col])


def pred_or(rt: str, var: str, col: str = "OR") -> float:
    return float(csv(f"prediction_{rt}_or.csv").set_index("variable").loc[var, col])


def hazard(rt: str, contrast: str, col: str, analysis: str = "main") -> float:
    c = csv("dropout_hazard_contrasts.csv")
    r = c[(c.race_type == rt) & (c.contrast == contrast) & (c.analysis == analysis)]
    assert len(r) == 1
    return float(r.iloc[0][col])


ANNUAL_RATE = {"total": "dnf_rate_pct", "inrace": "inrace_rate_pct", "finish_time": "finish_time_rate_pct"}


def annual_series(rt: str, series: str = "total") -> pd.Series:
    """Annual rate (%) by season of event type rt (temporal_trends_annual.csv, full precision): series 'total' (DNF
    rate, the secondary series of the trend tests), 'inrace' (in-race DNF rate, primary) or 'finish_time'."""
    d = csv("temporal_trends_annual.csv")
    d = d[d.race_type == rt]
    return pd.Series(d[ANNUAL_RATE[series]].to_numpy(dtype=float), index=d["race_year"].astype(int).to_numpy())


def annual(rt: str, year: int, series: str = "total") -> float:
    s = annual_series(rt, series)
    if year not in s.index:
        raise LookupError(f"temporal_trends_annual.csv: no row for {rt} {year}")
    return float(s.loc[year])


def annual_count(rt: str, year: int, col: str = "starters") -> float:
    """A column of season `year`, event type rt (temporal_trends_annual.csv): a count (starters, dnf, inrace_dnf,
    finish_time_dnf, starters_official, finish_time_dnf_official) or an official-source share (starters_official_pct,
    finish_time_dnf_official_pct)."""
    d = csv("temporal_trends_annual.csv")
    r = d[(d.race_type == rt) & (d.race_year == year)]
    assert len(r) == 1
    return float(r.iloc[0][col])


def trend(rt: str, stat: str, series: str = "total", seasons: str = "all") -> float:
    """Trend statistic of the annual rate (temporal_trends_tests.csv, full precision): n_years, ols_slope, ols_ci_low,
    ols_ci_high, ols_r2, ols_p, mk_tau, mk_p, sen_slope, ...; series 'inrace' (primary), 'total' (secondary) or
    'finish_time'; seasons 'all' (every season but 2020 and 2026), or, in-race series only, '2002-2009' (IRONMAN®),
    '2004-2009' (IRONMAN® 70.3) and '2010-2025'."""
    d = csv("temporal_trends_tests.csv")
    r = d[(d.series == series) & (d.race_type == rt) & (d.seasons == seasons)]
    assert len(r) == 1
    return float(r.iloc[0][stat])


def period_rate(rt: str, period: str) -> float:
    """Pooled in-race DNF rate (%) of event type rt over the seasons of `period`, e.g. '2002-2009' or '2010-2019'
    (temporal_trends_periods.csv, full precision; 2020 and 2026 excluded)."""
    d = csv("temporal_trends_periods.csv")
    r = d[(d.race_type == rt) & (d.period == period)]
    assert len(r) == 1
    return float(r.iloc[0]["inrace_rate_pct"])


class SourceAbsent(FileNotFoundError):
    """An external source of a claim (companion QA output, deposit rerun audit) is not present."""


_COMPANION_READS = [0]   # calls of companion(): tells _deposit_pairs which claims read the companion QA outputs


def companion(name: str) -> pd.DataFrame:
    _COMPANION_READS[0] += 1
    path = COMPANION_RESULTS / name
    if not path.exists():
        shown = path.relative_to(SIBLINGS) if path.is_relative_to(SIBLINGS) else Path(COMPANION_RESULTS.name) / name
        raise SourceAbsent(f"companion QA output absent: {shown.as_posix()}")   # same note on every platform
    return pd.read_csv(path, comment="#")


def age_rate(rt: str, age_bin: str, gender: str, variant: str = "excl_pro_finish_time") -> float:
    d = csv("descriptives_age_gender.csv")
    r = d[(d.variant == variant) & (d.race_type == rt) & (d.age_bin == age_bin) & (d.gender == gender)]
    assert len(r) == 1
    return float(r.iloc[0]["DNF_rate_pct"])


def origin_rate(continent: str, rt: str) -> float:
    """DNF rate by athlete continent × event type (descriptives.txt section 4). pandas prints a
    continent label only on the first row of its group, so the second row is matched as a
    continuation line that follows the labelled row of the same continent."""
    try:
        return grab_section("descriptives.txt", "DNF rate by athlete continent",
                            rf"^{re.escape(continent)}\s+{rt}\s+\d+\s+\d+\s+([\d.]+)")
    except LookupError:
        return grab_section("descriptives.txt", "DNF rate by athlete continent",
                            rf"^{re.escape(continent)}\s+(?:im|him)\s+\d+\s+\d+\s+[\d.]+\n\s+{rt}\s+\d+\s+\d+\s+([\d.]+)")


def origin_counts(continent: str, rt: str) -> tuple[int, int]:
    """Starters and DNFs (N_starts, N_DNF) by athlete continent × event type (descriptives.txt section 4), the row
    matched as in origin_rate (the labelled row, or the continuation line that follows the labelled row of the same
    continent)."""
    for row in (rf"^{re.escape(continent)}\s+{rt}\s+(\d+)\s+(\d+)\s",
                rf"^{re.escape(continent)}\s+(?:im|him)\s+\d+\s+\d+\s+[\d.]+\n\s+{rt}\s+(\d+)\s+(\d+)\s"):
        try:
            n, n_dnf = (int(grab_section("descriptives.txt", "DNF rate by athlete continent", row, g)) for g in (1, 2))
            return n, n_dnf
        except LookupError:
            continue
    raise LookupError(f"descriptives.txt: row of {continent} ({rt}) not found after 'DNF rate by athlete continent'")


def home_away_delta(continent: str) -> float:
    d = csv("athlete_vs_venue_home_away.csv").set_index(["athlete_continent", "is_home"])
    home, away = d.loc[(continent, "Home")], d.loc[(continent, "Away")]
    return 100 * (home.N_DNF / home.N_starts - away.N_DNF / away.N_starts)


def curve_km(rt: str, km: int, col: str = "hazard_per_1e4", analysis: str = "main") -> float:
    c = csv("dropout_hazard_curve.csv")
    r = c[(c.race_type == rt) & (c.km_bin_start == km) & (c.analysis == analysis)]
    assert len(r) == 1
    return float(r.iloc[0][col])


def seg_share(rt: str, segment: str, variant: str = "distance") -> float:
    """Share (%) of official in-race DNFs with a distance value whose last mat lies in `segment`
    (dropout_point_segments.csv, persisted by dropout_point.py section 3). Variant 'distance' = the distance
    classification; variant 't2_rule' = the segments of the hazard's run-DNF definition: its run segment counts the
    DNFs at the cycling-completion distance whose T2 split is within the duration limit (longer T2 splits and no T2
    split stay in the cycling segment) plus those beyond the cycling completion and before the finish distance, and
    the DNFs at or beyond the finish distance are kept apart (segments 'at the finish distance' and 'beyond the
    finish distance')."""
    d = csv("dropout_point_segments.csv")
    r = d[(d.race_type == rt) & (d.variant == variant) & (d.segment == segment)]
    assert len(r) == 1
    return float(r.iloc[0]["pct_of_with_distance"])


def excluded(col: str, reason: str | None = None, rt: str | None = None) -> float:
    """Sum of `col` (or the number of race-years when col == 'race_years') over excluded_race_years.csv."""
    d = csv("excluded_race_years.csv")
    if reason is not None:
        d = d[d.reason == reason]
    if rt is not None:
        d = d[d.race_type == rt]
    return float(len(d)) if col == "race_years" else float(d[col].sum())


def course_top(rt: str, label: str, what: str = "value") -> float:
    """Most common own course position (dropout_point.txt section 3): its value or its number of events."""
    m = re.search(rf"^    {rt.upper()} {re.escape(label)}: ([\d.]+) \(([\d,]+)\)", txt("dropout_point.txt"), re.M)
    if not m:
        raise LookupError(f"dropout_point.txt: most common {label} ({rt}) not found")
    return float((m.group(1) if what == "value" else m.group(2)).replace(",", ""))


def season_value(rt: str, year: int, col: str) -> float:
    """dropout_point_by_season.csv: a column of season `year` for event type rt ('all' = both event types)."""
    d = csv("dropout_point_by_season.csv")
    r = d[(d.race_type == rt) & (d.race_year == year)]
    assert len(r) == 1
    return float(r.iloc[0][col])


def season_extreme(col: str, fn, years: tuple[int, int], rt: str = "all") -> float:
    """min or max of a dropout_point_by_season.csv column over the seasons years[0]..years[1] (event type rt)."""
    d = csv("dropout_point_by_season.csv")
    d = d[(d.race_type == rt) & d.race_year.between(*years)]
    return float(fn(d[col]))


def km0_share_other(fn, year: int = 2019, min_run: int = 1, rt: str = "all") -> float:
    """min or max share (%) of the run-segment DNFs of the main rule recorded at km 0 over the seasons other than
    `year` with at least `min_run` run-segment DNFs, by default every season with any (dropout_point_by_season.csv;
    rt 'all' = both event types)."""
    d = csv("dropout_point_by_season.csv")
    d = d[(d.race_type == rt) & (d.race_year != year) & (d.n_run_dnf >= min_run)]
    return float(fn(d["pct_run_dnf_km0"]))


def cohort_season(rt: str, analysis: str, which: str) -> float:
    """First or last season (which = 'first_season' / 'last_season') of a hazard cohort
    (dropout_hazard_cohort_seasons.csv; analysis 'main', 'years_le_2019' or 'years_ge_2021')."""
    d = csv("dropout_hazard_cohort_seasons.csv")
    d = d[(d.race_type == rt) & (d.analysis == analysis)]
    assert d[which].nunique() == 1
    return float(d[which].iloc[0])


def venue_ci_pattern() -> float:
    """1 when, over the race-location ORs of Europe, Asia and Africa in the event-type-specific sensitivity models
    (S1, S3, S4 = Supplementary Tables S5, S7, S8; both event types), every OR is above 1 and the cells whose 95% CI
    includes 1 are exactly IRONMAN® 70.3 Africa in S1, S3 and S4 and IRONMAN® 70.3 Asia in S1."""
    s = csv("sensitivity_or.csv")
    s = s[s.analysis.isin(["S1", "S3", "S4"]) & s.variable.isin(["race_Africa", "race_Asia", "race_Europe"])]
    if len(s) != 18:
        raise LookupError(f"sensitivity_or.csv: {len(s)} race-location cells, 18 expected")
    if not np.isfinite(s[["OR", "CI_low", "CI_high"]].to_numpy(dtype=float)).all():
        raise LookupError("sensitivity_or.csv: non-finite OR or CI in a race-location cell")
    incl = {(a, m.split(" ")[0], v) for a, m, v, lo in zip(s.analysis, s.model, s.variable, s.CI_low) if lo < 1}
    expected = {("S1", "HIM", "race_Africa"), ("S3", "HIM", "race_Africa"), ("S4", "HIM", "race_Africa"), ("S1", "HIM", "race_Asia")}
    return float(incl == expected and bool((s.OR > 1).all()))


def curve_dip_then_rise(rt: str, start: int, min_steps: int = 1) -> float:
    """1 when the main per-km hazard, from `start` km to the last bin plotted in Figure 3, has its minimum strictly
    inside that window and rises at every step from the minimum to the last plotted bin."""
    c = csv("dropout_hazard_curve.csv")
    c = c[(c.race_type == rt) & (c.analysis == "main")].sort_values("km_bin_start")
    first_term = c.loc[c["terminal_bin"].astype(bool), "km_bin_start"].min()
    h = c[(c.km_bin_start >= start) & (c.km_bin_start < first_term)]["hazard_per_1e4"].to_numpy(dtype=float)
    i = int(h.argmin())
    return float(0 < i <= len(h) - 1 - min_steps and all(b > a for a, b in zip(h[i:], h[i + 1:])))


def curve_last_plotted(rt: str, what: str = "hazard") -> float:
    """The last per-km bin plotted in Figure 3 (the bin before the first one dominated by terminal bins), main
    analysis: its hazard per 10,000 athlete-km (what='hazard') or its start in km (what='start')."""
    c = csv("dropout_hazard_curve.csv")
    c = c[(c.race_type == rt) & (c.analysis == "main")].sort_values("km_bin_start")
    first_term = c.loc[c["terminal_bin"].astype(bool), "km_bin_start"].min()
    last = c[c.km_bin_start < first_term].iloc[-1]
    return float(last["hazard_per_1e4"] if what == "hazard" else last["km_bin_start"])


def checkpoint_stat(rt: str, q: float) -> float:
    """Quantile q (0.5 = median) of the distinct run checkpoints per event of the main hazard cohort, computed at full
    precision from dropout_hazard_events.csv (dropout_hazard.txt prints it rounded to an integer)."""
    d = csv("dropout_hazard_events.csv")
    return float(d[(d.race_type == rt) & (d.analysis == "main")]["n_mats"].quantile(q))


def t2_limit_min(rt: str) -> float:
    """T2 duration limit (minutes, full precision) of an event type (dropout_point_t2.csv)."""
    return float(csv("dropout_point_t2.csv").set_index("race_type").loc[rt, "t2_limit_min"])


def t2_median(rt: str, group: str) -> float:
    """Median T2 split (minutes, full precision) of a group of dropout_point_t2.csv: 'at_t2' (in-race DNFs at the
    cycling-completion distance with a T2 split within the duration limit), 'at_t2_long' or 'finishers'."""
    d = csv("dropout_point_t2.csv").set_index("race_type")
    return float(d.loc[rt, f"median_t2_min_{group}"])


def course_own(rt: str) -> float:
    """Official-source events of event type rt with their own course positions (event_course.csv: basis own or
    own_nominal_run)."""
    d = csv("event_course.csv")
    return float(((d.race_type == rt) & d.basis.isin(["own", "own_nominal_run"])).sum())


DEPOSIT_RERUN = RESULTS_DIR / "deposit_rerun_audit.csv"
AUDIT_CODE_KEY = "# Audit code:"
RESULTS_CODE_KEY = "# Audited results, analysis code:"
ANALYSIS_CODE_KEY = "# Analysis code: sha256 "   # line of _common.reproducibility_header in every results file
DATASET_KEY = "# Dataset: "                      # line of _common.reproducibility_header in every results file
DEPOSIT_DATASET = "ironman_dataset_deidentified.csv"   # the deposited de-identified file


class DepositAuditInvalid(RuntimeError):
    """results/deposit_rerun_audit.csv exists but cannot back the deposit claims (results of another dataset, another
    analysed population, other audit code, or results produced by other analysis code): the deposit claims are ERROR,
    not SKIP."""


def _results_header(key: str) -> tuple[dict[str, str], list[str]]:
    """The header line starting with `key` in every results file this audit reads (every txt and csv of RESULTS_DIR
    except the audit CSVs): {file name: rest of the line} and the names of the files without it."""
    found, missing = {}, []
    for p in sorted(RESULTS_DIR.glob("*.txt")) + sorted(RESULTS_DIR.glob("*.csv")):
        if p.name in (OUT_CSV.name, DEPOSIT_RERUN.name):
            continue
        with p.open(encoding="utf-8") as fh:
            value = None
            for line in fh:
                if not line.startswith("#"):
                    break
                if line.startswith(key):
                    value = line[len(key):].strip()
                    break
        if value is None:
            missing.append(p.name)
        else:
            found[p.name] = value
    return found, missing


def _shared_value(found: dict[str, str], missing: list[str], short=lambda v: v) -> str:
    """The value shared by every results file, 'absent' when none carries the line, or 'mixed (...)' when they disagree
    or some lack it."""
    values = sorted(set(found.values()))
    if not found:
        return "absent"
    if missing or len(values) > 1:
        return f"mixed ({', '.join(short(v) for v in values)}; without the line: {len(missing)} files)"
    return values[0]


def results_analysis_code() -> str:
    """Analysis-code fingerprint recorded in the headers of the results this audit reads: the sha256 when all files
    carry the same one, 'absent' when none does, or 'mixed (...)' (results produced by different code)."""
    found, missing = _results_header(ANALYSIS_CODE_KEY)
    return _shared_value({k: v.split()[0] for k, v in found.items()}, missing, lambda v: v[:12])


def results_dataset() -> str:
    """Dataset of the results this audit reads: the '# Dataset:' value (file name and version) when every results
    file carries the same one, 'absent' when none does, or 'mixed (...)' (results of different input files)."""
    return _shared_value(*_results_header(DATASET_KEY))


def deposit_audit() -> pd.DataFrame:
    """results/deposit_rerun_audit.csv indexed by claim. SourceAbsent (the deposit claims are SKIP) when the file is
    absent; DepositAuditInvalid (ERROR) when it exists but audited results of another input than the deposited file
    (its '# Dataset:' line, the dataset of the audited results, does not name DEPOSIT_DATASET), was produced under
    another analysed population (its '# Analysed population:' line differs from the current one), by other audit code
    (its audit-code fingerprint differs from code_sha256(AUDIT_CODE_EXCLUDE) now) or on results produced by other
    analysis code (its audited-results fingerprint differs from code_sha256(ANALYSIS_CODE_EXCLUDE) now). Then rerun
    the deposit check (notebooks/README.md) and copy its audit CSV again."""
    if not DEPOSIT_RERUN.exists():
        raise SourceAbsent(f"deposit rerun audit absent: results/{DEPOSIT_RERUN.name}")
    with DEPOSIT_RERUN.open(encoding="utf-8") as fh:
        header = []
        for line in fh:
            if not line.startswith("#"):
                break
            header.append(line.rstrip("\n"))
    value = lambda key: next((h[len(key):].strip() for h in header if h.startswith(key)), "")
    pop_key = "# Analysed population:"
    current_pop = next(line for line in reproducibility_header("").splitlines() if line.startswith(pop_key))
    problems = []
    dataset = value(DATASET_KEY.strip())
    if (dataset.split() + [""])[0] != DEPOSIT_DATASET:
        problems.append(f"results of another dataset ({dataset[:60] or 'absent'}; the deposit audit needs {DEPOSIT_DATASET})")
    if next((h for h in header if h.startswith(pop_key)), "") != current_pop:
        problems.append(f"another analysed population ({value(pop_key)[:60]!r}...)")
    audit_code = (value(AUDIT_CODE_KEY).split() + ["", ""])[1]
    if audit_code != code_sha256(AUDIT_CODE_EXCLUDE):
        problems.append(f"other audit code (fingerprint {audit_code[:12] or 'absent'}, now {code_sha256(AUDIT_CODE_EXCLUDE)[:12]})")
    results_code = (value(RESULTS_CODE_KEY).split() + ["", ""])[1] if value(RESULTS_CODE_KEY).startswith("sha256 ") else ""
    if results_code != code_sha256(ANALYSIS_CODE_EXCLUDE):
        problems.append(f"results of other analysis code ({value(RESULTS_CODE_KEY)[:40] or 'absent'}, now "
                        f"{code_sha256(ANALYSIS_CODE_EXCLUDE)[:12]})")
    if problems:
        raise DepositAuditInvalid(f"results/{DEPOSIT_RERUN.name} cannot back the deposit claims: " + "; ".join(problems)
                                  + "; rerun the deposit check (notebooks/README.md) and copy its audit CSV again")
    return pd.read_csv(DEPOSIT_RERUN, comment="#").set_index("claim")


def deposit(claim: str) -> float:
    """Value recomputed on the deposited de-identified file (ironman_dataset_deidentified.csv).

    Steps 1-10 of the pipeline rerun with IRONMAN_DNF_DATA_PATH pointing at the deposit and
    the results/table/figure directories in scratch, then this audit run on that results directory;
    its CSV is kept as results/deposit_rerun_audit.csv (notebooks/README.md, Reproduction
    check). Backs the Data availability statement."""
    return float(deposit_audit().loc[claim, "actual"])


def fig1_n(rt: str) -> float:
    """Starters plotted in Figure 1: variant excl_pro_finish_time of descriptives_age_gender.csv, Male + Female."""
    d = csv("descriptives_age_gender.csv")
    d = d[(d["variant"] == "excl_pro_finish_time") & (d["race_type"] == rt) & d["gender"].isin(["Male", "Female"])]
    return float(d["N_starts"].sum())


def age_rise_breaks(start: str = "25-29") -> float:
    """Number of decreases in the DNF rate between consecutive age bins from `start` to 70+, summed over
    both sexes and both event types (Figure 1 population); 0 when the rate rises at every step."""
    d = csv("descriptives_age_gender.csv")
    d = d[d["variant"] == "excl_pro_finish_time"]
    bins = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
    bins = bins[bins.index(start):]
    breaks = 0
    for (_, _), g in d.groupby(["race_type", "gender"]):
        r = g.set_index("age_bin").reindex(bins)["DNF_rate_pct"].to_numpy(dtype=float)
        if not np.isfinite(r).all():
            raise LookupError("descriptives_age_gender.csv: an age bin is missing")
        breaks += int((r[1:] <= r[:-1]).sum())
    return float(breaks)


def annual_extreme(rt: str, fn, years: tuple[int, int] = (2002, 2025), series: str = "total") -> float:
    """min or max of the annual rate of `series` over the seasons years[0]..years[1] (2020 included)."""
    s = annual_series(rt, series)
    return float(fn(s[(s.index >= years[0]) & (s.index <= years[1])]))


def fm_ratios(rt: str) -> dict[str, float]:
    """Female-to-male DNF rate ratio by age bin (Figure 1 population)."""
    bins = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
    return {b: age_rate(rt, b, "Female") / age_rate(rt, b, "Male") for b in bins}


def fm_ratio_extreme(rt: str, first: str, last: str, fn) -> float:
    """min or max of the female-to-male ratio over the age bins first..last (inclusive)."""
    r = fm_ratios(rt)
    keys = list(r)
    return fn(r[b] for b in keys[keys.index(first):keys.index(last) + 1])


def curve_after3(rt: str, fn, start: int = 3, end: int | None = None) -> float:
    """min or max of the main per-km hazard over the bins from `start` km up to the last bin plotted in
    Figure 3 (the bins before the first one dominated by terminal bins), or up to `end` when given."""
    c = csv("dropout_hazard_curve.csv")
    c = c[(c.race_type == rt) & (c.analysis == "main")].sort_values("km_bin_start")
    first_term = c.loc[c["terminal_bin"].astype(bool), "km_bin_start"].min()
    c = c[(c.km_bin_start >= start) & (c.km_bin_start < first_term)]
    if end is not None:
        c = c[c.km_bin_start <= end]
    return float(fn(c["hazard_per_1e4"]))


def curve_argmax_after3(rt: str) -> float:
    """Start (km) of the per-km bin with the highest main hazard after the third kilometre (Figure 3 range)."""
    c = csv("dropout_hazard_curve.csv")
    c = c[(c.race_type == rt) & (c.analysis == "main")].sort_values("km_bin_start")
    first_term = c.loc[c["terminal_bin"].astype(bool), "km_bin_start"].min()
    c = c[(c.km_bin_start >= 3) & (c.km_bin_start < first_term)]
    return float(c.loc[c["hazard_per_1e4"].idxmax(), "km_bin_start"])


# Record counts that the omission of records from the deposited file (the redacted secondary-source records) lowers:
# the only registered claims allowed to differ on the deposit, each downwards by at most the omitted records.
DEPOSIT_RECORD_COUNTS = frozenset({
    "n_records", "n_starters", "n_starters_im", "n_starters_him", "n_dns", "n_dq", "n_dnf", "n_dnf_im", "n_dnf_him",
    "n_with_status", "inrace_n", "country_strings_distinct", "fig1_n_im", "fig1_n_him", "records_analysed_n",
})


def _deposit_pairs() -> list[tuple[str, float, float, float]]:
    """(claim, tolerance, canonical value, deposit value) for every registered claim other than the deposit_*
    claims. Fails when the deposit rerun audit lacks a claim or holds a non-finite value for it, so a claim
    registered after the last deposit rerun cannot be skipped silently: rerun the deposit check
    (notebooks/README.md) and copy its audit CSV again. A claim that reads the companion QA outputs (which do not
    depend on the input file) and was SKIP in the deposit audit, because they were absent where it ran, is left
    out rather than treated as stale."""
    dep = deposit_audit()
    pairs, missing = [], []
    for key, (printed, tol, _, fn) in CLAIMS.items():
        if key.startswith("deposit_"):
            continue
        reads_before = _COMPANION_READS[0]
        try:
            canon = fn()
        except SourceAbsent:
            continue
        if _COMPANION_READS[0] > reads_before and key in dep.index and str(dep.loc[key, "status"]) == "SKIP":
            continue
        if key not in dep.index or not math.isfinite(float(dep.loc[key, "actual"])):
            missing.append(key)
            continue
        pairs.append((key, tol, float(canon), float(dep.loc[key, "actual"])))
    if missing:
        raise LookupError(f"deposit rerun audit is stale: {len(missing)} claims absent or non-finite "
                          f"({', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''})")
    return pairs


def deposit_changed_estimates() -> float:
    """Registered claims other than the record counts of DEPOSIT_RECORD_COUNTS whose value on the deposited file
    differs from the canonical value at the printed precision (the decimals implied by the tolerance: tolerance 0 =
    integers, so booleans, years, ranks, medians of ranks and event counts must be equal; wider tolerances compared
    at integer precision), or, for a bound claim, whose bound holds on one file and not on the other."""
    n = 0
    for key, tol, canon, dep in _deposit_pairs():
        if key in DEPOSIT_RECORD_COUNTS:
            continue
        printed = CLAIMS[key][0]
        if isinstance(printed, Bound):
            n += printed.holds(canon) != printed.holds(dep)
            continue
        decimals = STANDARD_DECIMALS.get(next((s for s in STANDARD_DECIMALS if abs(tol - s) < 1e-12), None), 0)
        q = Decimal(1).scaleb(-decimals)
        if (Decimal(repr(canon)).quantize(q, rounding=ROUND_HALF_UP)
                != Decimal(repr(dep)).quantize(q, rounding=ROUND_HALF_UP)):
            n += 1
    return float(n)


def deposit_max_count_drop() -> float:
    """Largest drop of a record count (DEPOSIT_RECORD_COUNTS) on the deposited file. Fails if one rises, or falls by
    more than the records omitted from the deposit (n_records, canonical minus deposit), which the omission of
    records cannot produce."""
    unknown = sorted(k for k in DEPOSIT_RECORD_COUNTS if k not in CLAIMS or CLAIMS[k][1] != 0)
    if unknown:
        raise LookupError(f"DEPOSIT_RECORD_COUNTS lists claims that are not registered count claims: {', '.join(unknown)}")
    drops = {key: canon - dep for key, _, canon, dep in _deposit_pairs() if key in DEPOSIT_RECORD_COUNTS}
    omitted = drops["n_records"]
    bad = [f"{k} ({d:+g})" for k, d in drops.items() if not 0 <= d <= omitted]
    if bad:
        raise ValueError(f"record counts that rise, or fall by more than the {omitted:g} omitted records, on the deposited "
                         f"file: {', '.join(bad)}")
    return float(max(drops.values()))


def deposit_omitted(status: str) -> float:
    """Records omitted from the deposited file with a given finish status: canonical count minus the count on the
    deposit, from the registered count claims and the deposit rerun audit (fin = starters - DNF - DQ; dnf, dq, dns)."""
    def drop(claim: str) -> float:
        return CLAIMS[claim][3]() - deposit(claim)
    if status == "fin":
        return float(drop("n_starters") - drop("n_dnf") - drop("n_dq"))
    return float(drop({"dnf": "n_dnf", "dq": "n_dq", "dns": "n_dns"}[status]))


def stopped_lowest_share() -> float:
    """Lowest share (%) of in-race DNFs (DNFs without a recorded finish time) among the starters of the excluded
    stopped or wholesale-coded editions (the quantity of the stopped-edition rule)."""
    d = csv("excluded_race_years.csv")
    d = d[d.reason == "stopped"]
    return float((100 * d["inrace_dnf"] / d["starters"]).min())


def no_checkpoint_all_in(year: str) -> float:
    """1 when every event excluded from the first-interval contrasts for lacking a checkpoint at or beyond 1 km
    (dropout_hazard.txt section 1, both event types, main analysis) was held in `year`."""
    names = []
    for marker in ("IRONMAN\n", "IRONMAN 70.3"):
        body = txt("dropout_hazard.txt")
        body = body[body.find(marker):]
        i = body.find("events without a checkpoint at or beyond 1.0 km")
        if i < 0:
            raise LookupError("dropout_hazard.txt: list of events without a checkpoint at or beyond 1 km not found")
        for line in body[i:].split("\n")[1:]:
            if not line.startswith("    "):
                break
            names.append(line.strip())
    if not names:
        raise LookupError("dropout_hazard.txt: no excluded events listed")
    return float(all(n.startswith(year) for n in names))


def annual_argextreme(rt: str, fn, years: tuple[int, int], series: str = "total") -> float:
    """Season with the highest (fn=max) or lowest (fn=min) annual rate of `series` within `years`."""
    s = annual_series(rt, series)
    s = s[(s.index >= years[0]) & (s.index <= years[1])]
    return float(s.index[list(s.to_numpy()).index(fn(s.to_numpy()))])


def venue_share(rt: str, athlete: str, venues: list[str]) -> float:
    """Share (%) of the starts of athletes from `athlete` held on the `venues` continents (event type `rt`)."""
    d = csv("athlete_vs_venue_share_by_type.csv")
    d = d[(d.race_type == rt) & (d.athlete_continent == athlete)]
    return float(d[d.event_continent.isin(venues)]["share_pct"].sum())


def pro_adjust_drop(fn) -> float:
    """Smallest (fn=min) or largest (fn=max) drop, in percentage points, of the four professional DNF rates when
    the professional DNFs without any recorded split are removed (pro_curation.txt section 2)."""
    pairs = re.findall(r"raw=([\d.]+)% \(n=[\d,]+\) \| adjusted=([\d.]+)%", txt("pro_curation.txt"))
    if len(pairs) != 4:
        raise LookupError(f"pro_curation.txt: expected 4 adjusted rates, found {len(pairs)}")
    return float(fn(float(r) - float(a) for r, a in pairs))


def age_steeper(start: str = "45-49") -> float:
    """1 when, in all four sex-by-event-type series (Figure 1 population), the increase of the DNF rate between
    consecutive age bins grows at every step from `start` onward."""
    d = csv("descriptives_age_gender.csv")
    d = d[d["variant"] == "excl_pro_finish_time"]
    bins = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
    bins = bins[bins.index(start):]
    ok = True
    for (_, _), g in d.groupby(["race_type", "gender"]):
        if set(g["gender"]) - {"Male", "Female"}:
            continue
        r = g.set_index("age_bin").reindex(bins)["DNF_rate_pct"].to_numpy()
        steps = r[1:] - r[:-1]
        ok &= bool((steps[1:] > steps[:-1]).all())
    return float(ok)


def women_higher_every_age() -> float:
    """1 when the DNF rate of women exceeds that of men in every age bin of both event types (Figure 1 population: the
    female-to-male ratio is above 1 in all eleven bins)."""
    return float(all(r > 1 for rt in ("im", "him") for r in fm_ratios(rt).values()))


def origin_extremes(rt: str, top: tuple[str, ...], bottom: str) -> float:
    """1 when, by athlete continent of origin (descriptives.txt section 4; the continents of CONTINENT_MAP), every
    continent in `top` has a DNF rate of event type rt strictly above that of every other continent and `bottom` one
    strictly below that of every other continent. The rates are computed exactly from the counts (N_DNF / N_starts),
    not from the 4-decimal display: an exact tie fails the claim, and rates equal at four decimals are compared by
    their exact values."""
    rates = {}
    for c in sorted(set(CONTINENT_MAP.values())):
        n, n_dnf = origin_counts(c, rt)
        rates[c] = Fraction(n_dnf, n)
    others = [r for c, r in rates.items() if c not in top]
    return float(min(rates[c] for c in top) > max(others)
                 and all(rates[bottom] < r for c, r in rates.items() if c != bottom))


def or_age_rises(start: str = "35-39") -> float:
    """1 when, in both regression models (prediction_{im,him}_or.csv), the odds ratio of the age bins (relative to the
    30-34 reference) rises at every step from `start` to 70+."""
    bins = ["35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+"]
    bins = bins[bins.index(start):]
    return float(all(pred_or(rt, f"age_{a}") < pred_or(rt, f"age_{b}")
                     for rt in ("im", "him") for a, b in zip(bins, bins[1:])))


def crosstab_no_row(segment: str) -> float:
    """1 when neither cross-tabulation of the two localization methods (dropout_point.txt section 4, IRONMAN® and
    IRONMAN® 70.3; the rows of Figure 3a-b) has a row for `segment`, i.e. no in-race DNF of the cross-tabulations has
    it as its last completed segment by splits. The row labels are read between the 'last_completed_segment' header and
    the 'All' margin of each table."""
    body = txt("dropout_point.txt")
    body = body[body.index("SECTION 4: METHOD COMPARISON"):body.index("SECTION 5:")]
    tables = re.findall(r"^last_completed_segment[ \t]*\n((?:.*\n)*?)All[ \t]", body, re.M)
    if len(tables) != 2:
        raise LookupError(f"dropout_point.txt section 4: {len(tables)} cross-tabulations found, 2 expected")
    return float(segment not in [line.split()[0] for t in tables for line in t.splitlines() if line.strip()])


def run_restricted(rt: str, what: str) -> float:
    """Run-segment DNFs in the events with at least one run checkpoint beyond km 0 (dropout_point.txt section 5):
    'p25', 'median', 'p75' of km_run, or 'km0_pct', the share at km 0 computed from its counts."""
    start = "\nIM run-segment in-race" if rt == "im" else "\nHIM run-segment in-race"
    if what == "km0_pct":
        n = grab_section("dropout_point.txt", start, r"events with a run checkpoint beyond km 0 \(at least one run DNF with km_run > 0\): [\d,]+ of [\d,]+ events with run DNFs; run DNFs in them: n = ([\d,]+)")
        k0 = grab_section("dropout_point.txt", start, r"run DNFs in them: n = [\d,]+ \(T2-completers at km 0: ([\d,]+)\)")
        return 100 * k0 / n
    label = {"p25": "P25", "median": "median", "p75": "P75"}[what]
    return grab_section("dropout_point.txt", start, rf"events with a run checkpoint beyond km 0, km_run: [^\n]*?{label} ([\d.]+)")


class Bound:
    """A printed bound ('p < 0.001', 'more than 99% in every season'): the claim is ok when the persisted value
    satisfies the inequality at full precision, without rounding it to the printed precision first."""
    OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}

    def __init__(self, op: str, value: float) -> None:
        if op not in self.OPS:
            raise ValueError(f"Bound: unknown operator {op!r}")
        self.op, self.value = op, float(value)

    def holds(self, actual: float) -> bool:
        return math.isfinite(float(actual)) and self.OPS[self.op](float(actual), self.value)

    def __str__(self) -> str:
        return f"{self.op} {self.value:g}"


# ---------------------------------------------------------------------------
# Claims: key -> (printed, tolerance, location, extractor)
#   printed=None marks a NEW claim with no printed value yet; printed=Bound(op, value) a printed bound, checked as
#   an inequality at full precision (its tolerance slot, BOUND, is not used).
# ---------------------------------------------------------------------------
PCT, CNT, OR3, KM = 0.05, 0, 0.0005, 0.05
BOUND = 0.0

CLAIMS: dict[str, tuple[float | None, float, str, callable]] = {
    # --- dataset / population (Abstract, Methods §Dataset, §Study Population) ---
    "n_records": (3_632_323, CNT, "Methods §Dataset (whole file, before the exclusions)",
                  lambda: grab("descriptives.txt", r"Records in the file read: ([\d,]+)")),
    "n_nostatus": (10_342, CNT, "Methods §Study Population",
                   lambda: grab("descriptives.txt", r"Records without a finish status[^:]*: ([\d,]+)")),
    "n_starters": (3_192_507, CNT, "Abstract; Methods §Study Population; Results §Sample Characteristics; Figure 2 caption",
                   lambda: t1("im", "Total", "starters") + t1("him", "Total", "starters")),
    "n_starters_im": (1_159_245, CNT, "Results §Sample Characteristics", lambda: t1("im", "Total", "starters")),
    "n_starters_him": (2_033_262, CNT, "Results §Sample Characteristics", lambda: t1("him", "Total", "starters")),
    "n_dns": (383_300, CNT, "Methods §Study Population; Results §Sample Characteristics",
              lambda: t1("im", "Total", "dns") + t1("him", "Total", "dns")),
    "n_dq": (12_592, CNT, "Results §Sample Characteristics", lambda: t1("im", "Total", "dq") + t1("him", "Total", "dq")),
    "n_dnf": (208_513, CNT, "Abstract; Introduction (purpose); Results §Sample Characteristics; Discussion",
              lambda: t1("im", "Total", "dnf") + t1("him", "Total", "dnf")),
    "n_dnf_im": (106_810, CNT, "Results §Sample Characteristics; Results §Dropout Localization (Split-based method)", lambda: t1("im", "Total", "dnf")),
    "n_dnf_him": (101_703, CNT, "Results §Sample Characteristics; Results §Dropout Localization (Split-based method)", lambda: t1("him", "Total", "dnf")),
    "n_with_status": (3_575_807, CNT, "Results §Sample Characteristics ('records with a finish status')",
                      lambda: t1("im", "Total", "n_records") + t1("him", "Total", "n_records")),
    "dnf_pct_of_records": (5.8, PCT, "Results §Sample Characteristics ('208,513 (5.8%) were DNFs')",
                           lambda: 100 * (t1("im", "Total", "dnf") + t1("him", "Total", "dnf")) / (t1("im", "Total", "n_records") + t1("him", "Total", "n_records"))),
    "dns_pct_of_records": (10.7, PCT, "Results §Sample Characteristics",
                           lambda: 100 * (t1("im", "Total", "dns") + t1("him", "Total", "dns")) / (t1("im", "Total", "n_records") + t1("him", "Total", "n_records"))),
    "dnf_rate_im": (9.2, PCT, "Abstract; Results §Sample Characteristics; Conclusions; Table 1",
                    lambda: t1("im", "Total", "dnf_rate_pct")),
    "dnf_rate_him": (5.0, PCT, "Abstract; Results §Sample Characteristics; Conclusions; Table 1",
                     lambda: t1("him", "Total", "dnf_rate_pct")),
    "dnf_rate_im_female": (10.7, PCT, "Results §DNF Rates by Sex, Age, and Region; Table 1", lambda: t1("im", "Female", "dnf_rate_pct")),
    "dnf_rate_im_male": (8.9, PCT, "Results §DNF Rates by Sex, Age, and Region; Table 1", lambda: t1("im", "Male", "dnf_rate_pct")),
    "dnf_rate_him_female": (6.2, PCT, "Results §DNF Rates by Sex, Age, and Region; Table 1", lambda: t1("him", "Female", "dnf_rate_pct")),
    "dnf_rate_him_male": (4.6, PCT, "Results §DNF Rates by Sex, Age, and Region; Table 1", lambda: t1("him", "Male", "dnf_rate_pct")),

    # --- dataset construction (Methods §Dataset) — companion QA on v4.0.0 ---
    "race_years_total": (2180, CNT, "Methods §Dataset",
                         lambda: float(companion("r1_event_coverage.csv").iloc[0]["total_race_years"])),
    "official_share_pct": (69.7, PCT, "Methods §Dataset",
                           lambda: 100 * companion("r1_event_coverage.csv").iloc[0]["official_records_total"] / (companion("r1_event_coverage.csv").iloc[0]["official_records_total"] + companion("r1_event_coverage.csv").iloc[0]["supplementary_records"])),
    "aggregator_share_pct": (30.3, PCT, "Methods §Dataset",
                             lambda: 100 * companion("r1_event_coverage.csv").iloc[0]["supplementary_records"] / (companion("r1_event_coverage.csv").iloc[0]["official_records_total"] + companion("r1_event_coverage.csv").iloc[0]["supplementary_records"])),
    "cross_source_race_years": (1237, CNT, "Methods §Dataset",
                                lambda: float(companion("r1_cross_source_summary.csv").iloc[0]["race_years"])),
    "cross_source_pairs": (9_600_332, CNT, "Methods §Dataset",
                           lambda: float(companion("r1_cross_source_summary.csv").iloc[0]["split_pairs"])),
    "cross_source_agreement_pct": (99.2, PCT, "Methods §Dataset",
                                   lambda: float(companion("r1_cross_source_summary.csv").iloc[0]["split_agreement_pct"])),
    "internal_consistency_1s_pct": (98.3, PCT, "Methods §Dataset ('within one second')",
                                    lambda: float(companion("r1_internal_consistency.csv").set_index("tolerance_sec").loc[1, "pct_within"])),

    # --- finish-time / in-race DNFs (Methods §DNF Classification; Results §Finish-Time DNFs; Figure 4) ---
    "ft_n": (9199, CNT, "Results §DNFs with a Recorded Finish Time",
             lambda: grab("dropout_point.txt", r"^Finish-time DNFs: ([\d,]+) /")),
    "ft_pct": (4.4, PCT, "Abstract; Results §DNFs with a Recorded Finish Time",
               lambda: grab("dropout_point.txt", r"^Finish-time DNFs: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "inrace_n": (199_314, CNT, "Results §DNFs with a Recorded Finish Time (in-race DNFs)",
                    lambda: grab("dropout_point.txt", r"^In-race DNFs: ([\d,]+) /")),
    "inrace_pct": (95.6, PCT, "Results §DNFs with a Recorded Finish Time (in-race DNFs)",
                      lambda: grab("dropout_point.txt", r"^In-race DNFs: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_im_n": (1793, CNT, "Results §DNFs with a Recorded Finish Time; Figure 4 caption",
                lambda: grab("dropout_point.txt", r"^IM: finish-time ([\d,]+) /")),
    "ft_im_pct": (1.7, PCT, "Results §DNFs with a Recorded Finish Time; Results §Dropout Localization (Split-based method)",
                  lambda: grab("dropout_point.txt", r"^IM: finish-time [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_him_n": (7406, CNT, "Results §DNFs with a Recorded Finish Time; Figure 4 caption",
                 lambda: grab("dropout_point.txt", r"^HIM: finish-time ([\d,]+) /")),
    "ft_him_pct": (7.3, PCT, "Results §DNFs with a Recorded Finish Time; Results §Dropout Localization (Split-based method)",
                   lambda: grab("dropout_point.txt", r"^HIM: finish-time [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_him_share_of_ft_pct": (80.5, PCT, "Results §DNFs with a Recorded Finish Time ('80.5% of all finish-time DNFs')",
                               lambda: 100 * grab("dropout_point.txt", r"^HIM: finish-time ([\d,]+) /") / grab("dropout_point.txt", r"^Finish-time DNFs: ([\d,]+) /")),
    "ft_median_h_im": (15.5, PCT, "Results §DNFs with a Recorded Finish Time",
                       lambda: grab_section("dropout_point.txt", "\nIM: finish-time", r"P50=([\d.]+)")),
    "ft_median_h_him": (8.2, PCT, "Results §DNFs with a Recorded Finish Time",
                        lambda: grab_section("dropout_point.txt", "HIM: finish-time", r"P50=([\d.]+)")),
    "ft_all_five_splits_pct": (89.0, PCT, "Results §DNFs with a Recorded Finish Time ('among the 89.0% with all five split times recorded')",
                               lambda: grab("dropout_point.txt", r"all five split times recorded \(> 0\): [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_splits_within_5s_n": (8185, CNT, "Results §DNFs with a Recorded Finish Time ('8,185 of 8,189')",
                              lambda: grab("dropout_point.txt", r"within 5 s among those with all five splits: ([\d,]+) / ")),
    "ft_all_five_splits_n": (8189, CNT, "Results §DNFs with a Recorded Finish Time ('8,185 of 8,189')",
                             lambda: grab("dropout_point.txt", r"all five split times recorded \(> 0\): ([\d,]+) / ")),
    "ft_splits_within_5s_all_pct": (91.0, PCT, "Results §DNFs with a Recorded Finish Time ('in 91.0% of all finish-time DNFs')",
                                    lambda: grab("dropout_point.txt", r"within 5 s of overall_sec, all finish-time DNFs: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_with_distance_n": (6738, CNT, "Results §DNFs with a Recorded Finish Time",
                           lambda: grab("dropout_point.txt", r"with distance data: ([\d,]+) / ")),
    "ft_with_distance_pct": (73.2, PCT, "Results §DNFs with a Recorded Finish Time",
                             lambda: grab("dropout_point.txt", r"with distance data: [\d,]+ / [\d,]+ = ([\d.]+)% of finish-time DNFs")),
    "ft_at_full_distance_pct": (90.4, PCT, "Results §DNFs with a Recorded Finish Time ('at or above the finish distance of their event')",
                                lambda: grab("dropout_point.txt", r"at or above the event's modal finisher distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_within3km_pct": (92.1, PCT, "Results §DNFs with a Recorded Finish Time ('within 3 km of it'; two-sided, |distance - finish| <= 3 km)",
                         lambda: grab("dropout_point.txt", r"within 3 km of the event's modal finisher distance, two-sided \(\|distance - finish\| <= 3 km\): [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "inrace_swim_split_pct": (87.5, PCT, "Methods §DNF Classification (official source, in-race DNFs)",
                                 lambda: grab_section("dropout_point.txt", "Split times > 0 among in-race DNFs", r"^\s*swim\s+[\d.]+%\s+([\d.]+)%")),
    "inrace_bike_split_pct": (54.5, PCT, "Methods §DNF Classification (official source, in-race DNFs)",
                                 lambda: grab_section("dropout_point.txt", "Split times > 0 among in-race DNFs", r"^\s*bike\s+[\d.]+%\s+([\d.]+)%")),
    "inrace_run_split_pct": (0.1, PCT, "Methods §DNF Classification (official source, in-race DNFs)",
                                lambda: grab_section("dropout_point.txt", "Split times > 0 among in-race DNFs", r"^\s*run\s+[\d.]+%\s+([\d.]+)%")),

    # --- finish-time DNFs: typical cut-off, ranking, concentration by edition (dropout_point.txt section 1) ---
    "ft_im_below_cutoff_pct": (79.8, PCT, "Results §DNFs with a Recorded Finish Time",
                               lambda: grab("dropout_point.txt", r"^  IM: below 17\.0 h: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_him_below_cutoff_pct": (57.2, PCT, "Results §DNFs with a Recorded Finish Time",
                                lambda: grab("dropout_point.txt", r"^  HIM: below 8\.5 h: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_im_over_20h_n": (1, CNT, "Figure 4 caption ('One IRONMAN® record above 20 hours lies outside the axis')",
                         lambda: grab("dropout_point.txt", r"^  IM: overall time > 20\.0 h[^:]*: ([\d,]+)")),
    "ft_im_under_4_5h_n": (2, CNT, "Figure 4 caption (IRONMAN® finish-time DNFs below 4.5 h)",
                           lambda: grab("dropout_point.txt", r"^  IM: overall time < 4\.5 h \(Figure 4 left tail\): ([\d,]+)")),
    "ft_him_under_2h_n": (1, CNT, "Figure 4 caption (IRONMAN® 70.3 finish-time DNFs below 2 h)",
                          lambda: grab("dropout_point.txt", r"^  HIM: overall time < 2\.0 h \(Figure 4 left tail\): ([\d,]+)")),
    "ft_unranked_pct": (98.2, PCT, "Results §DNFs with a Recorded Finish Time",
                        lambda: grab("dropout_point.txt", r"^  unranked: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_editions_n": (835, CNT, "Results §DNFs with a Recorded Finish Time",
                      lambda: grab("dropout_point.txt", r"editions with >= 1 finish-time DNF: ([\d,]+) of")),
    "ft_editions_any_dnf_n": (2116, CNT, "Results §DNFs with a Recorded Finish Time",
                              lambda: grab("dropout_point.txt", r"editions with >= 1 finish-time DNF: [\d,]+ of ([\d,]+) editions")),
    "ft_editions_median": (5.0, 0.05, "Results §DNFs with a Recorded Finish Time ('median 5 per event')",
                           lambda: grab("dropout_point.txt", r"finish-time DNFs per edition \(editions with >= 1\): median ([\d.]+)")),
    "ft_top_decile_editions_n": (83, CNT, "Results §DNFs with a Recorded Finish Time",
                                 lambda: grab("dropout_point.txt", r"top 10% of those editions by count \(([\d,]+) editions\)")),
    "ft_top_decile_share_pct": (41.1, PCT, "Results §DNFs with a Recorded Finish Time",
                                lambda: grab("dropout_point.txt", r"top 10% of those editions by count \([\d,]+ editions\): [\d,]+ / [\d,]+ = ([\d.]+)%")),

    # --- age × sex (Results §DNF Rates; Figure 1, variant excl_pro_finish_time) ---
    "age_im_male_25_29": (6.4, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: age_rate("im", "25-29", "Male")),
    "age_im_male_55_59": (12.4, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: age_rate("im", "55-59", "Male")),
    "age_im_male_65_69": (22.6, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: age_rate("im", "65-69", "Male")),
    "age_im_male_70": (35.2, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: age_rate("im", "70+", "Male")),
    "age_im_female_25_29": (7.1, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: age_rate("im", "25-29", "Female")),
    "age_im_female_55_59": (15.7, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: age_rate("im", "55-59", "Female")),
    "age_im_female_65_69": (31.5, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: age_rate("im", "65-69", "Female")),
    "age_im_female_70": (49.8, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: age_rate("im", "70+", "Female")),
    "age_rise_breaks_from_25_29": (0, CNT, "Results §DNF Rates by Sex, Age, and Region ('increased progressively with age from the 25–29 age group onward'; decreases between consecutive bins, both sexes and event types)",
                                   lambda: age_rise_breaks("25-29")),
    "fm_ratio_im_70_max": (1.4, 0.05, "Results §DNF Rates by Sex, Age, and Region ('highest at 70+', IRONMAN®; NaN unless the 70+ ratio is the largest)",
                           lambda: (lambda r: r["70+"] if r["70+"] == max(r.values()) else float("nan"))(fm_ratios("im"))),
    "fm_ratio_im_30_34": (1.1, 0.05, "Results §DNF Rates by Sex, Age, and Region ('approximately 1.1 in the 25–34 age range')",
                          lambda: age_rate("im", "30-34", "Female") / age_rate("im", "30-34", "Male")),
    "cycling_median_h_im": (6.2, PCT, "Results §Sample Characteristics; Discussion §Early-run withdrawal and the cycling-to-running transition ('median cycling split of 6.2 hours')",
                            lambda: grab_section("descriptives.txt", "SECTION 3c", r"^\s*IM \(n = [\d,]+\): P10=[\d.]+ P25=[\d.]+ P50=([\d.]+)")),
    "cycling_median_h_him": (3.0, PCT, "Results §Sample Characteristics; Discussion §Early-run withdrawal and the cycling-to-running transition ('3.0 hours')",
                             lambda: grab_section("descriptives.txt", "SECTION 3c", r"HIM \(n = [\d,]+\): P10=[\d.]+ P25=[\d.]+ P50=([\d.]+)")),

    # --- athlete origin (Results §DNF Rates) ---
    "origin_im_asia": (11.5, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("Asia", "im")),
    "origin_im_north_america": (10.3, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("North America", "im")),
    "origin_im_europe": (8.7, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("Europe", "im")),
    "origin_im_africa": (8.7, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("Africa", "im")),
    "origin_im_south_america": (6.4, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("South America", "im")),
    "origin_im_oceania": (5.8, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("Oceania", "im")),
    "origin_him_asia": (7.2, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("Asia", "him")),
    "origin_him_africa": (6.9, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("Africa", "him")),
    "origin_him_south_america": (4.0, PCT, "Results §DNF Rates by Sex, Age, and Region", lambda: origin_rate("South America", "him")),
    "origin_him_extremes": (1, CNT, "Results §DNF Rates by Sex, Age, and Region ('In IRONMAN® 70.3 events, athletes from Asia (7.2%) and Africa (6.9%) had the highest rates and South American athletes the lowest'; among the six continents of origin)",
                            lambda: origin_extremes("him", ("Asia", "Africa"), "South America")),
    "home_away_north_america_pp": (-2.5, 0.05, "Results §DNF Rates by Sex, Age, and Region (home vs away); Table S2", lambda: home_away_delta("North America")),
    "home_away_oceania_pp": (-1.6, 0.05, "Results §DNF Rates by Sex, Age, and Region (home vs away); Table S2", lambda: home_away_delta("Oceania")),
    "home_away_south_america_pp": (-0.8, 0.05, "Results §DNF Rates by Sex, Age, and Region (home vs away); Table S2", lambda: home_away_delta("South America")),
    "home_away_africa_pp": (1.3, 0.05, "Results §DNF Rates by Sex, Age, and Region (home vs away); Table S2", lambda: home_away_delta("Africa")),
    "home_away_europe_pp": (0.7, 0.05, "Results §DNF Rates by Sex, Age, and Region (home vs away); Table S2", lambda: home_away_delta("Europe")),
    "home_away_asia_pp": (-0.7, 0.05, "Results §DNF Rates by Sex, Age, and Region (home vs away); Table S2", lambda: home_away_delta("Asia")),
    "home_away_overall_pp": (-0.3, 0.05, "Results §DNF Rates by Sex, Age, and Region (home vs away); Table S2", lambda: home_away_delta("Overall")),

    # --- professionals (Results §Professional Athletes; Table 1; Table S4) ---
    "pro_rate_im_male": (24.0, PCT, "Abstract; Results §Professional Athletes; Discussion §Professional athletes; Table 1",
                         lambda: t1("im", "Male", "pro_dnf_rate_pct")),
    "pro_rate_im_female": (17.7, PCT, "Results §Professional Athletes; Table 1",
                           lambda: t1("im", "Female", "pro_dnf_rate_pct")),
    "pro_rate_him_male": (12.0, PCT, "Results §Professional Athletes; Table 1",
                          lambda: t1("him", "Male", "pro_dnf_rate_pct")),
    "pro_rate_him_female": (9.6, PCT, "Results §Professional Athletes; Table 1",
                            lambda: t1("him", "Female", "pro_dnf_rate_pct")),
    "pro_im_male_starts": (14_102, CNT, "Results §Professional Athletes", lambda: t1("im", "Male", "pro_starters")),
    "pro_im_male_dnf": (3388, CNT, "Results §Professional Athletes", lambda: t1("im", "Male", "pro_dnf")),
    "pro_im_female_starts": (7469, CNT, "Results §Professional Athletes", lambda: t1("im", "Female", "pro_starters")),
    "pro_im_female_dnf": (1321, CNT, "Results §Professional Athletes", lambda: t1("im", "Female", "pro_dnf")),
    "pro_annual_seasons_n": (23, CNT, "Results §Professional Athletes ('23 seasons with at least 100 IRONMAN® male professional starts')",
                             lambda: grab("pro_rank.txt", r"Years with >= 100 MPRO starts: (\d+)")),
    "pro_annual_min_pct": (21.3, 0.05, "Results §Professional Athletes ('21.3% to 26.8%')",
                           lambda: grab("pro_rank.txt", r"annual rate range ([\d.]+)%-")),
    "pro_annual_max_pct": (26.8, 0.05, "Results §Professional Athletes", lambda: grab("pro_rank.txt", r"annual rate range [\d.]+%-([\d.]+)%")),
    "pro_annual_median_pct": (23.9, 0.05, "Results §Professional Athletes", lambda: grab("pro_rank.txt", r"annual rate range [\d.]+%-[\d.]+%, median ([\d.]+)%")),
    "pro_no_split_pct": (11.4, PCT, "Results §Professional Athletes ('11.4% of professional DNFs with no recorded split time in any segment')",
                         lambda: grab("pro_curation.txt", r"with NO recorded splits \(possible DNS→DNF\):\s+[\d,]+ \(([\d.]+)%\)")),
    "pro_no_split_im_pct": (7.6, PCT, "Results §Professional Athletes ('7.6% in IRONMAN®')",
                            lambda: grab("pro_curation.txt", r"\[IRONMAN\] n=[\d,]+ \| swim split: [\d,]+ \([\d.]+%\) \| no splits: [\d,]+ \(([\d.]+)%\)")),
    "pro_any_split_pct": (88.6, PCT, "Discussion §Professional athletes ('88.6% of professional DNFs had at least one recorded split')",
                          lambda: grab("pro_curation.txt", r"with any recorded split \(any segment > 0\):\s+[\d,]+ \(([\d.]+)%\)")),
    "pro_rate_im_male_adjusted": (22.7, PCT, "Results §Professional Athletes",
                                  lambda: grab("pro_curation.txt", r"IRONMAN male: raw=[\d.]+% \(n=[\d,]+\) \| adjusted=([\d.]+)%")),
    "pro_dist_n": (1393, CNT, "Results §Professional Athletes (IRONMAN® professional in-race DNFs with a distance value)", lambda: grab("pro_curation.txt", r"^IM PRO in-race DNFs with total_distance_km:\s+([\d,]+)")),
    "pro_dist_pct": (29.7, PCT, "Results §Professional Athletes (share of IRONMAN® professional in-race DNFs)", lambda: grab("pro_curation.txt", r"^IM PRO in-race DNFs with total_distance_km:\s+[\d,]+ \(([\d.]+)%")),
    "pro_dist_median_km": (183.9, KM, "Results §Professional Athletes", lambda: grab("pro_curation.txt", r"Median total_distance_km:\s+([\d.]+) km")),
    "pro_rank_dnf_n": (1414, CNT, "Results §Professional Athletes; Table S4", lambda: grab("pro_rank.txt", r"DNFs with valid bike_rank: ([\d,]+)")),
    "pro_rank_dnf_pct": (30.1, PCT, "Results §Professional Athletes; Discussion §Strengths and limitations (share of in-race DNFs)", lambda: grab("pro_rank.txt", r"DNFs with valid bike_rank: [\d,]+ \(([\d.]+)% of in-race DNFs\)")),
    "pro_rank_fin_n": (13_612, CNT, "Results §Professional Athletes; Table S4", lambda: grab("pro_rank.txt", r"FINs with valid bike_rank: ([\d,]+)")),
    "pro_rank_fin_pct": (81.0, PCT, "Results §Professional Athletes", lambda: grab("pro_rank.txt", r"FINs with valid bike_rank: [\d,]+ \(([\d.]+)% of FINs\)")),
    "pro_outside_top10_dnf_pct": (83.8, PCT, "Results §Professional Athletes; Table S4",
                                  lambda: grab("pro_rank.txt", r"^\s*10\s+([\d.]+)\s+[\d.]+\s+-?[\d.]+$")),
    "pro_outside_top10_fin_pct": (74.9, PCT, "Results §Professional Athletes; Table S4",
                                  lambda: grab("pro_rank.txt", r"^\s*10\s+[\d.]+\s+([\d.]+)\s+-?[\d.]+$")),
    "pro_outside_top10_or": (1.734, 0.0005, "Results §Professional Athletes; Discussion §Professional athletes", lambda: grab("pro_rank.txt", r"OR = ([\d.]+) \(95% CI")),
    "pro_outside_top10_or_ci_low": (1.498, 0.0005, "Results §Professional Athletes", lambda: grab("pro_rank.txt", r"OR = [\d.]+ \(95% CI ([\d.]+)-")),
    "pro_outside_top10_or_ci_high": (2.008, 0.0005, "Results §Professional Athletes", lambda: grab("pro_rank.txt", r"OR = [\d.]+ \(95% CI [\d.]+-([\d.]+)\)")),
    "pro_rank_dnf_p25": (15, CNT, "Results §Professional Athletes ('median rank of 39 (25th–75th percentile 15–138)')", lambda: grab_section("pro_rank.txt", "IM PRO in-race DNFs with valid bike_rank:", r"^  25%\s+([\d.]+)")),
    "pro_rank_dnf_p75": (138, CNT, "Results §Professional Athletes ('15–138'; 137.75 at full precision, printed as an integer rank)", lambda: grab_section("pro_rank.txt", "IM PRO in-race DNFs with valid bike_rank:", r"^  75%\s+([\d.]+)")),
    "pro_rank_fin_p25": (10, CNT, "Results §Professional Athletes ('versus 28 (10–94')", lambda: grab_section("pro_rank.txt", "Reference — IM PRO FINishers bike_rank distribution:", r"^  25%\s+([\d.]+)")),
    "pro_rank_fin_p75": (94, CNT, "Results §Professional Athletes ('versus 28 (10–94')", lambda: grab_section("pro_rank.txt", "Reference — IM PRO FINishers bike_rank distribution:", r"^  75%\s+([\d.]+)")),
    "pro_position_p25_km": (-89.6, KM, "Results §Professional Athletes ('median position was 0.0 km (25th–75th percentile −89.6 to 14.6 km)')", lambda: grab("pro_curation.txt", r"Position relative to the event's cycling completion[^\n]*P25 (-?[\d.]+);")),
    "pro_position_p75_km": (14.6, KM, "Results §Professional Athletes ('−89.6 to 14.6 km')", lambda: grab("pro_curation.txt", r"Position relative to the event's cycling completion[^\n]*P75 (-?[\d.]+)")),
    "cycling_split_im_p25_h": (5.6, PCT, "Results §Sample Characteristics ('6.2 hours (25th–75th percentile 5.6–6.9)')", lambda: grab_section("descriptives.txt", "SECTION 3c", r"^  IM \(n = [\d,]+\): P10=[\d.]+ P25=([\d.]+)")),
    "cycling_split_im_p75_h": (6.9, PCT, "Results §Sample Characteristics ('5.6–6.9')", lambda: grab_section("descriptives.txt", "SECTION 3c", r"^  IM \(n = [\d,]+\):[^\n]*P75=([\d.]+)")),
    "cycling_split_him_p25_h": (2.7, PCT, "Results §Sample Characteristics ('3.0 hours (2.7–3.3)')", lambda: grab_section("descriptives.txt", "SECTION 3c", r"^  HIM \(n = [\d,]+\): P10=[\d.]+ P25=([\d.]+)")),
    "cycling_split_him_p75_h": (3.3, PCT, "Results §Sample Characteristics ('2.7–3.3')", lambda: grab_section("descriptives.txt", "SECTION 3c", r"^  HIM \(n = [\d,]+\):[^\n]*P75=([\d.]+)")),
    "dist_him_p25_km": (1.9, KM, "Results §Dropout Localization (Distance-based method) ('a median of 76.1 km (25th–75th percentile 1.9–93.3 km)')", lambda: grab_section("dropout_point.txt", "Distance completed by in-race DNFs", r"HIM \(n=[\d,]+\):[\s\S]*?25%\s+([\d.]+)")),
    "dist_him_p75_km": (93.3, KM, "Results §Dropout Localization (Distance-based method) ('1.9–93.3 km')", lambda: grab_section("dropout_point.txt", "Distance completed by in-race DNFs", r"HIM \(n=[\d,]+\):[\s\S]*?75%\s+([\d.]+)")),
    "ft_time_im_p25_h": (13.3, PCT, "Results §DNFs with a Recorded Finish Time ('15.5 hours (25th–75th percentile 13.3–16.8)')", lambda: grab_section("dropout_point.txt", "\nIM: finish-time ", r"Finish-time DNF overall time \(hours\): P25=([\d.]+)")),
    "ft_time_im_p75_h": (16.8, PCT, "Results §DNFs with a Recorded Finish Time ('13.3–16.8')", lambda: grab_section("dropout_point.txt", "\nIM: finish-time ", r"Finish-time DNF overall time \(hours\):[^\n]*P75=([\d.]+)")),
    "ft_time_him_p25_h": (6.9, PCT, "Results §DNFs with a Recorded Finish Time ('8.2 hours (6.9–8.7)')", lambda: grab_section("dropout_point.txt", "\nHIM: finish-time ", r"Finish-time DNF overall time \(hours\): P25=([\d.]+)")),
    "ft_time_him_p75_h": (8.7, PCT, "Results §DNFs with a Recorded Finish Time ('6.9–8.7')", lambda: grab_section("dropout_point.txt", "\nHIM: finish-time ", r"Finish-time DNF overall time \(hours\):[^\n]*P75=([\d.]+)")),
    "ft_editions_p25": (2, CNT, "Results §DNFs with a Recorded Finish Time ('median 5 per event, 25th–75th percentile 2–14')", lambda: grab("dropout_point.txt", r"finish-time DNFs per edition \(editions with >= 1\): median [\d.]+ \(P25 ([\d.]+),")),
    "ft_editions_p75": (14, CNT, "Results §DNFs with a Recorded Finish Time ('2–14')", lambda: grab("dropout_point.txt", r"finish-time DNFs per edition \(editions with >= 1\): median [\d.]+ \(P25 [\d.]+, P75 ([\d.]+)")),
    "pro_median_rank_dnf": (39, CNT, "Results §Professional Athletes", lambda: grab("pro_rank.txt", r"median DNF ([\d.]+) vs FIN")),
    "pro_median_rank_fin": (28, CNT, "Results §Professional Athletes", lambda: grab("pro_rank.txt", r"median DNF [\d.]+ vs FIN ([\d.]+)")),
    "pro_rank_mw_p": (Bound("<", 0.001), BOUND, "Results §Professional Athletes ('Mann-Whitney U test, p < 0.001'; rank at the end of the cycling segment, in-race DNFs vs finishers)",
                      lambda: grab("pro_rank.txt", r"Mann-Whitney U on bike_rank \(in-race DNF vs FIN\): U = [\d,]+, p = ([\d.eE+-]+);")),

    # --- regression (Results §Risk Factor Identification; Tables 2-3; Figure 5 caption) ---
    "reg_n_im": (1_106_486, CNT, "Methods §Statistical Analysis (Logistic Regression); Results §Risk Factor Identification; Table 2; Figure 5 caption", lambda: pred("im", "n")),
    "reg_n_him": (1_954_178, CNT, "Methods §Statistical Analysis (Logistic Regression); Results §Risk Factor Identification; Table 3; Figure 5 caption", lambda: pred("him", "n")),
    "reg_rate_im": (8.7, PCT, "Results §Risk Factor Identification", lambda: pred("im", "dnf_rate_pct")),
    "reg_rate_him": (4.5, PCT, "Results §Risk Factor Identification", lambda: pred("him", "dnf_rate_pct")),
    "auc_im": (0.614, 0.0005, "Results §Risk Factor Identification; Table 2; Figure 5 caption; Discussion §Strengths and limitations", lambda: pred("im", "auc")),
    "auc_him": (0.593, 0.0005, "Results §Risk Factor Identification; Table 3; Figure 5 caption", lambda: pred("him", "auc")),
    "pseudo_r2_im": (0.025, 0.0005, "Results §Risk Factor Identification", lambda: grab("prediction_im.txt", r"Pseudo R²: ([\d.]+)")),
    "pseudo_r2_him": (0.015, 0.0005, "Results §Risk Factor Identification", lambda: grab("prediction_him.txt", r"Pseudo R²: ([\d.]+)")),
    "or_female_im": (1.265, OR3, "Results §Risk Factor Identification; Table 2", lambda: pred_or("im", "female")),
    "or_female_him": (1.387, OR3, "Results §Risk Factor Identification; Table 3", lambda: pred_or("him", "female")),
    "or_female_im_2dp": (1.27, 0.005, "Abstract; cover letter", lambda: pred_or("im", "female")),
    "or_female_him_2dp": (1.39, 0.005, "Abstract; cover letter", lambda: pred_or("him", "female")),
    "or_age70_im": (8.741, OR3, "Results §Risk Factor Identification; Table 2; Discussion §Age as a steep risk factor", lambda: pred_or("im", "age_70+")),
    "or_age70_him": (5.289, OR3, "Results §Risk Factor Identification; Table 3", lambda: pred_or("him", "age_70+")),
    "or_age70_im_2dp": (8.74, 0.005, "Abstract; cover letter", lambda: pred_or("im", "age_70+")),
    "or_age70_him_2dp": (5.29, 0.005, "Abstract; cover letter", lambda: pred_or("him", "age_70+")),
    "or_age35_im": (1.110, OR3, "Results §Risk Factor Identification; Discussion §Age as a steep risk factor", lambda: pred_or("im", "age_35-39")),
    "or_age35_him": (1.036, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "age_35-39")),
    "or_age_rises_from_35": (1, CNT, "Results §Risk Factor Identification ('a monotonic gradient from the 35–39 age group onward'); Discussion §Age as a steep risk factor ('The monotonic increase in DNF odds with age'); OR rising at every step from 35–39 to 70+, both models",
                             lambda: or_age_rises("35-39")),
    "or_age25_im": (0.968, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "age_25-29")),
    "or_age25_him": (0.959, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "age_25-29")),
    "or_age18_im": (1.098, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "age_18-24")),
    "or_age18_him": (1.050, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "age_18-24")),
    "or_ath_europe_im": (0.614, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "ath_Europe")),
    "or_ath_oceania_im": (0.491, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "ath_Oceania")),
    "or_ath_south_america_im": (0.680, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "ath_South_America")),
    "or_ath_africa_im": (0.627, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "ath_Africa")),
    "or_ath_asia_im": (0.824, OR3, "Results §Risk Factor Identification; Discussion §Regional variation", lambda: pred_or("im", "ath_Asia")),
    "or_ath_asia_him": (1.302, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "ath_Asia")),
    "or_ath_africa_him": (1.147, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "ath_Africa")),
    "or_ath_europe_him": (0.750, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "ath_Europe")),
    "or_ath_south_america_him": (0.776, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "ath_South_America")),
    "or_ath_oceania_him": (0.966, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "ath_Oceania")),
    "or_race_asia_im": (1.645, OR3, "Results §Risk Factor Identification; Table 2", lambda: pred_or("im", "race_Asia")),
    "or_race_europe_im": (1.665, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "race_Europe")),
    "or_race_africa_im": (1.550, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "race_Africa")),
    "or_race_oceania_im": (1.014, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "race_Oceania")),
    "or_race_south_america_im": (0.977, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "race_South_America")),
    "or_race_africa_him": (1.423, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "race_Africa")),
    "or_race_europe_him": (1.261, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "race_Europe")),
    "or_race_asia_him": (1.245, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "race_Asia")),
    "or_race_south_america_him": (1.099, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "race_South_America")),
    "or_race_oceania_him": (0.839, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "race_Oceania")),
    "or_year_im": (0.974, OR3, "Results §Risk Factor Identification", lambda: pred_or("im", "race_year")),
    "or_year_him": (0.993, OR3, "Results §Risk Factor Identification", lambda: pred_or("him", "race_year")),
    "sens_s2_n": (612_133, CNT, "Table S6 caption",
                  lambda: float(csv("sensitivity_summary.csv").set_index("analysis").loc["S2", "n"])),

    # --- dropout localisation (Methods; Results §Dropout Localization; Figure 3 caption) ---
    "dnf_with_distance_pct": (41.4, PCT, "Methods §Dropout Point Localization ('41.4% of all DNFs carry a value')",
                              lambda: grab("dropout_point.txt", r"all DNFs with a distance value: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "dnf_with_distance_im_pct": (34.4, PCT, "Methods §Dropout Point Localization (share of all IRONMAN® DNFs carrying a distance value)",
                                 lambda: grab("dropout_point.txt", r"^  IM DNFs with a distance value: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "dnf_with_distance_him_pct": (48.7, PCT, "Methods §Dropout Point Localization (share of all IRONMAN® 70.3 DNFs carrying a distance value)",
                                  lambda: grab("dropout_point.txt", r"^  HIM DNFs with a distance value: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "split_im_t2_pct": (39.3, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nIM (n=", r"^T2\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_im_t1_pct": (27.7, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nIM (n=", r"^T1\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_im_bike_pct": (10.6, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nIM (n=", r"^bike\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_im_swim_pct": (5.5, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nIM (n=", r"^swim\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_im_run_pct": (0.5, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nIM (n=", r"^run\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_im_unknown_pct": (14.8, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nIM (n=", r"^unknown\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_him_t2_pct": (28.2, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nHIM (n=", r"^T2\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_him_t1_pct": (18.3, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nHIM (n=", r"^T1\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_him_bike_pct": (14.6, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nHIM (n=", r"^bike\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_him_swim_pct": (8.3, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nHIM (n=", r"^swim\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_him_run_pct": (3.5, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nHIM (n=", r"^run\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "split_him_unknown_pct": (19.7, PCT, "Results §Dropout Localization (Split-based method)", lambda: grab_section("dropout_point.txt", "\nHIM (n=", r"^unknown\s+\d+\s+([\d.]+)", after="SECTION 2: LAST COMPLETED SEGMENT")),
    "dist_n_im": (35_190, CNT, "Results §Dropout Localization (Distance-based method)",
                  lambda: grab_section("dropout_point.txt", "Distance completed by in-race DNFs", r"^  IM \(n=([\d,]+)\)")),
    "dist_n_him": (43_206, CNT, "Results §Dropout Localization (Distance-based method)",
                   lambda: grab_section("dropout_point.txt", "Distance completed by in-race DNFs", r"HIM \(n=([\d,]+)\)")),
    "dist_median_im_km": (182.8, KM, "Results §Dropout Localization (Distance-based method)",
                          lambda: grab_section("dropout_point.txt", "Distance completed by in-race DNFs", r"^  IM \(n=[\d,]+\):[\s\S]*?50%\s+([\d.]+)")),
    "dist_p25_im_km": (91.1, KM, "Results §Dropout Localization (Distance-based method)",
                       lambda: grab_section("dropout_point.txt", "Distance completed by in-race DNFs", r"^  IM \(n=[\d,]+\):[\s\S]*?25%\s+([\d.]+)")),
    "dist_p75_im_km": (196.9, KM, "Results §Dropout Localization (Distance-based method)",
                       lambda: grab_section("dropout_point.txt", "Distance completed by in-race DNFs", r"^  IM \(n=[\d,]+\):[\s\S]*?75%\s+([\d.]+)")),
    "dist_median_him_km": (76.1, KM, "Results §Dropout Localization (Distance-based method)",
                           lambda: grab_section("dropout_point.txt", "Distance completed by in-race DNFs", r"HIM \(n=[\d,]+\):[\s\S]*?50%\s+([\d.]+)")),
    "dist_seg_im_bike_pct": (48.5, PCT, "Results §Dropout Localization (Distance-based method)", lambda: seg_share("im", "bike")),
    "dist_seg_im_run_pct": (37.4, PCT, "Results §Dropout Localization (Distance-based method)", lambda: seg_share("im", "run")),
    "dist_seg_im_swim_pct": (7.9, PCT, "Results §Dropout Localization (Distance-based method)", lambda: seg_share("im", "swim")),
    "dist_seg_im_no_mat_pct": (6.2, PCT, "Results §Dropout Localization (Distance-based method; 'had not crossed any timing mat', distance 0)", lambda: seg_share("im", "no timing mat crossed")),
    "dist_seg_him_bike_pct": (42.9, PCT, "Results §Dropout Localization (Distance-based method)", lambda: seg_share("him", "bike")),
    "dist_seg_him_run_pct": (24.5, PCT, "Results §Dropout Localization (Distance-based method)", lambda: seg_share("him", "run")),
    "dist_seg_him_swim_pct": (15.4, PCT, "Results §Dropout Localization (Distance-based method)", lambda: seg_share("him", "swim")),
    "dist_seg_him_no_mat_pct": (17.2, PCT, "Results §Dropout Localization (Distance-based method; 'had not crossed any timing mat', distance 0)", lambda: seg_share("him", "no timing mat crossed")),
    "crosstab_both_n": (67_449, CNT, "Methods §Dropout Point Localization; Results §Dropout Localization (Cross-tabulation of methods)",
                        lambda: grab("dropout_point.txt", r"Records with both methods available: ([\d,]+)")),
    "crosstab_both_pct": (32.3, PCT, "Methods §Dropout Point Localization ('32.3% of all DNFs')",
                          lambda: 100 * grab("dropout_point.txt", r"Records with both methods available: ([\d,]+)") / (t1("im", "Total", "dnf") + t1("him", "Total", "dnf"))),
    "crosstab_im_n": (32_138, CNT, "Methods §Dropout Point Localization; Figure 3a caption",
                      lambda: grab("dropout_point.txt", r"^IM \(n=([\d,]+)\) — cross-tabulation")),
    "crosstab_him_n": (35_311, CNT, "Methods §Dropout Point Localization; Figure 3b caption",
                       lambda: grab("dropout_point.txt", r"^HIM \(n=([\d,]+)\) — cross-tabulation")),
    "crosstab_no_run_row": (1, CNT, "Figure 3 caption ('no record had the run as its last completed segment'; rows of the cross-tabulations, both event types)",
                            lambda: crosstab_no_row("run")),
    "t2_in_run_im_pct": (86.6, PCT, "Results §Dropout Localization (Cross-tabulation of methods); Discussion §Early-run withdrawal and the cycling-to-running transition",
                         lambda: grab_section("dropout_point.txt", "\nIM (n=", after="SECTION 4", pattern=r"T2 by splits, run by distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "t1_in_bike_im_pct": (93.2, PCT, "Results §Dropout Localization (Cross-tabulation of methods)",
                          lambda: grab_section("dropout_point.txt", "\nIM (n=", after="SECTION 4", pattern=r"T1 by splits, bike by distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "t2_in_run_him_pct": (71.7, PCT, "Results §Dropout Localization (Cross-tabulation of methods); Discussion §Early-run withdrawal and the cycling-to-running transition",
                          lambda: grab_section("dropout_point.txt", "HIM (n=", after="SECTION 4", pattern=r"T2 by splits, run by distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "t1_in_bike_him_pct": (76.1, PCT, "Results §Dropout Localization (Cross-tabulation of methods)",
                           lambda: grab_section("dropout_point.txt", "HIM (n=", after="SECTION 4", pattern=r"T1 by splits, bike by distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),

    # --- run-segment DNFs and the hazard (Methods §Dropout Hazard; Results §Dropout hazard; Figure 3c-d; Table S3) ---
    "run_dnf_n_im": (14_820, CNT, "Results §Dropout Localization (Dropout hazard along the run)",
                     lambda: grab("dropout_point.txt", r"^IM run-segment in-race DNFs: n = ([\d,]+)")),
    "run_dnf_n_him": (13_793, CNT, "Results §Dropout Localization (Dropout hazard along the run)",
                      lambda: grab("dropout_point.txt", r"^HIM run-segment in-race DNFs: n = ([\d,]+)")),
    "hazard_at_mat_im": (3696, CNT, "Methods §Dropout Hazard Along the Run (DNFs at the cycling-completion distance not modelled: without a T2 split)", lambda: grab_section("dropout_hazard.txt", "IRONMAN\n", r"without a T2 split \(withdrew in the transition; not modelled\): ([\d,]+)")),
    "hazard_at_mat_him": (6261, CNT, "Methods §Dropout Hazard Along the Run (DNFs at the cycling-completion distance not modelled: without a T2 split)", lambda: grab_section("dropout_hazard.txt", "IRONMAN 70.3", r"without a T2 split \(withdrew in the transition; not modelled\): ([\d,]+)")),
    "hazard_events_im": (184, CNT, "Results §Dropout Localization (Dropout hazard along the run); Figure 3c caption; Table S3",
                         lambda: hazard("im", "first_interval_vs_rest_between", "n_events_cohort")),
    "hazard_events_him": (119, CNT, "Results §Dropout Localization (Dropout hazard along the run); Figure 3d caption; Table S3",
                          lambda: hazard("him", "first_interval_vs_rest_between", "n_events_cohort")),
    "hazard_events_with_start_mat_im": (177, CNT, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("im", "first_interval_vs_rest_between", "n_events_with_start_mat")),
    "hazard_events_with_start_mat_him": (112, CNT, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("him", "first_interval_vs_rest_between", "n_events_with_start_mat")),
    "hazard_cohort_dnf_im": (13_488, CNT, "Results §Dropout Localization (Dropout hazard along the run); Figure 3c caption",
                             lambda: grab_section("dropout_hazard.txt", "IRONMAN\n", r"run DNFs covered: ([\d,]+)")),
    "hazard_cohort_dnf_him": (5749, CNT, "Results §Dropout Localization (Dropout hazard along the run); Figure 3d caption",
                              lambda: grab_section("dropout_hazard.txt", "IRONMAN 70.3", r"run DNFs covered: ([\d,]+)")),
    "hazard_run_dnf_covered_pct_im": (91.0, PCT, "Results §Dropout Localization (Dropout hazard along the run); Discussion §Strengths and limitations",
                                      lambda: grab_section("dropout_hazard.txt", "IRONMAN\n", r"run DNFs covered: [\d,]+ \(([\d.]+)%")),
    "hazard_run_dnf_covered_pct_him": (41.7, PCT, "Results §Dropout Localization (Dropout hazard along the run); Discussion §Strengths and limitations",
                                       lambda: grab_section("dropout_hazard.txt", "IRONMAN 70.3", r"run DNFs covered: [\d,]+ \(([\d.]+)%")),
    "checkpoints_per_event_median_im": (12, CNT, "Results §Dropout Localization (Dropout hazard along the run)",
                                        lambda: checkpoint_stat("im", 0.5)),
    "checkpoints_per_event_median_him": (7, CNT, "Results §Dropout Localization (Dropout hazard along the run)",
                                         lambda: checkpoint_stat("him", 0.5)),
    "first_checkpoint_median_im_km": (2.9, KM, "Abstract; Results §Dropout Localization (Dropout hazard along the run); Figure 3 caption",
                                      lambda: grab_section("dropout_hazard.txt", "IRONMAN\n", r"first run checkpoint \(km past the event's cycling completion\): median ([\d.]+)")),
    "first_checkpoint_median_him_km": (2.9, KM, "Abstract; Results §Dropout Localization (Dropout hazard along the run); Figure 3 caption",
                                       lambda: grab_section("dropout_hazard.txt", "IRONMAN 70.3", r"first run checkpoint \(km past the event's cycling completion\): median ([\d.]+)")),
    "hazard_ratio_im": (1.54, 0.005, "Abstract; Results §Dropout Localization (Dropout hazard along the run); Table S3",
                        lambda: hazard("im", "first_interval_vs_rest_between", "ratio")),
    "hazard_ratio_im_ci_low": (1.34, 0.005, "Abstract; Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("im", "first_interval_vs_rest_between", "ci_low")),
    "hazard_ratio_im_ci_high": (1.78, 0.005, "Abstract; Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("im", "first_interval_vs_rest_between", "ci_high")),
    "hazard_first_im": (15.0, 0.05, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("im", "first_interval_vs_rest_between", "h_first_per_1e4")),
    "hazard_rest_im": (9.8, 0.05, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("im", "first_interval_vs_rest_between", "h_rest_per_1e4")),
    "hazard_ratio_within_im": (1.53, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("im", "first_interval_vs_rest_within", "ratio")),
    "hazard_ratio_within_im_ci_low": (1.33, 0.005, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("im", "first_interval_vs_rest_within", "ci_low")),
    "hazard_ratio_within_im_ci_high": (1.77, 0.005, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("im", "first_interval_vs_rest_within", "ci_high")),
    "hazard_ratio_him": (1.66, 0.005, "Abstract; Results §Dropout Localization (Dropout hazard along the run); Table S3",
                         lambda: hazard("him", "first_interval_vs_rest_between", "ratio")),
    "hazard_ratio_him_ci_low": (1.40, 0.005, "Abstract; Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "first_interval_vs_rest_between", "ci_low")),
    "hazard_ratio_him_ci_high": (1.95, 0.005, "Abstract; Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "first_interval_vs_rest_between", "ci_high")),
    "hazard_first_him": (18.3, 0.05, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "first_interval_vs_rest_between", "h_first_per_1e4")),
    "hazard_rest_him": (11.0, 0.05, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "first_interval_vs_rest_between", "h_rest_per_1e4")),
    "hazard_ratio_within_him": (1.67, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "first_interval_vs_rest_within", "ratio")),
    "hazard_ratio_within_him_ci_low": (1.41, 0.005, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("him", "first_interval_vs_rest_within", "ci_low")),
    "hazard_ratio_within_him_ci_high": (1.98, 0.005, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("him", "first_interval_vs_rest_within", "ci_high")),
    "hazard_curve_first_im": (16.2, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("im", "curve_km0-3_vs_km3-end_excl_terminal", "h_first_per_1e4")),
    "hazard_curve_rest_im": (9.7, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("im", "curve_km0-3_vs_km3-end_excl_terminal", "h_rest_per_1e4")),
    "hazard_curve_ratio_im": (1.67, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("im", "curve_km0-3_vs_km3-end_excl_terminal", "ratio")),
    "hazard_curve_ratio_im_ci_low": (1.47, 0.005, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("im", "curve_km0-3_vs_km3-end_excl_terminal", "ci_low")),
    "hazard_curve_ratio_im_ci_high": (1.89, 0.005, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("im", "curve_km0-3_vs_km3-end_excl_terminal", "ci_high")),
    "hazard_curve_first_him": (16.9, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "h_first_per_1e4")),
    "hazard_curve_rest_him": (11.1, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "h_rest_per_1e4")),
    "hazard_curve_ratio_him": (1.52, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ratio")),
    "hazard_curve_ratio_him_ci_low": (1.32, 0.005, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ci_low")),
    "hazard_curve_ratio_him_ci_high": (1.73, 0.005, "Results §Dropout Localization (Dropout hazard along the run)", lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ci_high")),
    "hazard_km0_im": (21.7, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("im", 0)),
    "hazard_km0_im_ci_low": (18.0, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("im", 0, "ci_low")),
    "hazard_km0_im_ci_high": (26.5, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("im", 0, "ci_high")),
    "hazard_km1_im": (16.1, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("im", 1)),
    "hazard_km2_im": (11.1, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("im", 2)),
    "hazard_km0_him": (22.4, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("him", 0)),
    "hazard_km0_him_ci_low": (19.9, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("him", 0, "ci_low")),
    "hazard_km0_him_ci_high": (25.3, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("him", 0, "ci_high")),
    "hazard_km1_him": (16.5, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("him", 1)),
    "hazard_km2_him": (12.0, 0.05, "Results §Dropout Localization (Dropout hazard along the run)", lambda: curve_km("him", 2)),
    "hazard_sens_min_ratio_im": (1.49, 0.005, "Results §Dropout Localization (Dropout hazard along the run) ('IRONMAN®: 1.49–1.65'; sensitivities of the cohort, period, at-risk set and start of the run; the strict rule and the stopped events have their own claims)",
                                 lambda: csv("dropout_hazard_contrasts.csv").query("race_type == 'im' and contrast == 'first_interval_vs_rest_between' and analysis not in ('main', 'strict', 't2_no_limit', 'including_stopped')")["ratio"].min()),
    "hazard_sens_max_ratio_im": (1.65, 0.005, "Results §Dropout Localization (Dropout hazard along the run)",
                                 lambda: csv("dropout_hazard_contrasts.csv").query("race_type == 'im' and contrast == 'first_interval_vs_rest_between' and analysis not in ('main', 'strict', 't2_no_limit', 'including_stopped')")["ratio"].max()),
    "hazard_sens_min_ratio_him": (1.41, 0.005, "Results §Dropout Localization (Dropout hazard along the run) ('IRONMAN® 70.3: 1.41–1.73')",
                                  lambda: csv("dropout_hazard_contrasts.csv").query("race_type == 'him' and contrast == 'first_interval_vs_rest_between' and analysis not in ('main', 'strict', 't2_no_limit', 'including_stopped')")["ratio"].min()),
    "hazard_sens_max_ratio_him": (1.73, 0.005, "Results §Dropout Localization (Dropout hazard along the run)",
                                  lambda: csv("dropout_hazard_contrasts.csv").query("race_type == 'him' and contrast == 'first_interval_vs_rest_between' and analysis not in ('main', 'strict', 't2_no_limit', 'including_stopped')")["ratio"].max()),

    # --- geography (Methods §Demographic and Geographic Variables) ---
    "unknown_sex_starters": (3304, CNT, "Methods §Demographic and Geographic Variables",
                             lambda: grab_section("descriptives.txt", "ALL DISTANCES:", r"^Unknown\s+(\d+)")),
    "other_age_code_im_n": (3382, CNT, "Methods §Demographic and Geographic Variables (age-group starters of known sex outside the 5-year bins, IRONMAN®)",
                            lambda: grab_section("descriptives.txt", "fits none of the eleven age bins", r"^  IRONMAN: ([\d,]+) \(")),
    "other_age_code_him_n": (750, CNT, "Methods §Demographic and Geographic Variables (age-group starters of known sex outside the 5-year bins, IRONMAN® 70.3)",
                             lambda: grab_section("descriptives.txt", "fits none of the eleven age bins", r"^  IRONMAN 70\.3: ([\d,]+) \(")),
    "country_field_coverage_pct": (99.9, PCT, "Methods §Demographic and Geographic Variables",
                                   lambda: grab("country_mapping.txt", r"Non-empty country field: [\d,]+ \(([\d.]+)%\)")),
    "ambiguous_codes_pct": (0.07, 0.005, "Methods §Demographic and Geographic Variables",
                            lambda: grab("country_mapping.txt", r"Ambiguous short codes[^:]*: [\d,]+ \(([\d.]+)%\)")),
    "continent_mapping_coverage_pct": (98.0, PCT, "Methods §Demographic and Geographic Variables",
                                       lambda: grab("country_mapping.txt", r"Starters mapped to a continent: [\d,]+ / [\d,]+ \(([\d.]+)%\)")),
    "country_variants_n": (369, CNT, "Methods §Demographic and Geographic Variables",
                           lambda: grab("country_mapping.txt", r"Total country-name variants in mapping: (\d+)")),
    "country_strings_distinct": (473, CNT, "Methods §Demographic and Geographic Variables",
                                 lambda: grab("country_mapping.txt", r"Distinct non-empty country strings: (\d+)")),
    "country_strings_mapped": (366, CNT, "Methods §Demographic and Geographic Variables",
                               lambda: grab("country_mapping.txt", r"Distinct strings covered by the mapping: (\d+)")),
    "event_series_mapped_n": (174, CNT, "Methods §Demographic and Geographic Variables ('174 series mapped by hand')",
                              lambda: grab("country_mapping.txt", r"Official series mapped by hand to their host country \(SLUG_TO_COUNTRY\): (\d+)")),
    "cc_series_mapped_n": (331, CNT, "Methods §Demographic and Geographic Variables (series of the secondary source's race list other than the World Championships, mapped by hand to their host country; not all have a race in the file)",
                           lambda: grab("country_mapping.txt", r"mapped by hand to their host country \(CC_SERIES_TO_COUNTRY\): (\d+)")),
    "host_countries_n": (69, CNT, "Methods §Demographic and Geographic Variables (host countries over the official index and the secondary source's race list; the analysed events have the same 69)",
                         lambda: grab("country_mapping.txt", r"^Host countries: (\d+)")),
    "race_location_coverage_pct": (100.0, PCT, "Methods §Demographic and Geographic Variables ('100% coverage')",
                                   lambda: grab("athlete_vs_venue.txt", r"Race location coverage: [\d,]+ / [\d,]+ = ([\d.]+)%")),

    # --- temporal (Results §Temporal Trends; Discussion; Figure 2) ---
    "annual_im_2002_pct": (13.7, PCT, "Results §Temporal Trends", lambda: annual("im", 2002, "inrace")),
    "annual_im_2003_pct": (14.7, PCT, "Results §Temporal Trends", lambda: annual("im", 2003, "inrace")),
    "annual_im_2024_pct": (8.1, PCT, "Results §Temporal Trends", lambda: annual("im", 2024, "inrace")),
    "annual_im_2025_pct": (7.7, PCT, "Results §Temporal Trends", lambda: annual("im", 2025, "inrace")),
    "annual_im_2019_pct": (10.0, PCT, "Results §Temporal Trends", lambda: annual("im", 2019, "inrace")),
    "annual_him_2019_pct": (4.8, PCT, "Results §Temporal Trends", lambda: annual("him", 2019, "inrace")),
    "annual_im_2020_pct": (6.7, PCT, "Results §Temporal Trends", lambda: annual("im", 2020, "inrace")),
    "annual_him_2020_pct": (4.8, PCT, "Results §Temporal Trends", lambda: annual("him", 2020, "inrace")),
    "annual_im_2021_pct": (11.0, PCT, "Results §Temporal Trends", lambda: annual("im", 2021, "inrace")),
    "annual_him_2021_pct": (4.6, PCT, "Results §Temporal Trends", lambda: annual("him", 2021, "inrace")),
    "annual_im_2026_pct": (7.9, PCT, "Results §Temporal Trends", lambda: annual("im", 2026, "inrace")),
    "annual_him_2026_pct": (3.8, PCT, "Results §Temporal Trends", lambda: annual("him", 2026, "inrace")),
    "annual_him_max_pct": (5.7, PCT, "Results §Temporal Trends (in-race DNF rate; 'between 3.2% (2007) and 5.7% (2017)')", lambda: annual_extreme("him", max, (2004, 2025), "inrace")),
    "annual_him_min_pct": (3.2, PCT, "Results §Temporal Trends", lambda: annual_extreme("him", min, (2004, 2025), "inrace")),
    "annual_him_2023_2025_min_pct": (4.1, PCT, "Results §Temporal Trends (in-race DNF rate; '4.1–4.5% in 2023–2025')", lambda: annual_extreme("him", min, (2023, 2025), "inrace")),
    "annual_him_2023_2025_max_pct": (4.5, PCT, "Results §Temporal Trends (in-race DNF rate; '4.1–4.5% in 2023–2025')", lambda: annual_extreme("him", max, (2023, 2025), "inrace")),
    "annual_im_2021_highest_since_2010": (11.0, PCT, "Results §Temporal Trends; Discussion §Temporal trends (in-race DNF rate; 'the highest IRONMAN® rate of any season from 2010 onward')", lambda: annual_extreme("im", max, (2010, 2026), "inrace")),
    "starts_im_2020": (4265, CNT, "Results §Temporal Trends", lambda: annual_count("im", 2020)),
    "starts_him_2020": (12_666, CNT, "Results §Temporal Trends", lambda: annual_count("him", 2020)),
    "trend_im_n_years": (23, CNT, "Methods §Statistical Analysis (DNF Rates and Temporal Trends) ('23 annual observations for IRONMAN®'); Results §Temporal Trends ('over the 23 seasons analysed')", lambda: trend("im", "n_years", "inrace")),
    "trend_him_n_years": (21, CNT, "Methods §Statistical Analysis (DNF Rates and Temporal Trends) ('21 for IRONMAN® 70.3'); Results §Temporal Trends ('over the 21 seasons analysed')", lambda: trend("him", "n_years", "inrace")),
    "trend_im_slope": (-0.16, 0.005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("im", "ols_slope")),
    "trend_im_slope_ci_low": (-0.30, 0.005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("im", "ols_ci_low")),
    "trend_im_slope_ci_high": (-0.03, 0.005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("im", "ols_ci_high")),
    "trend_im_p": (0.022, 0.0005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("im", "ols_p")),
    "trend_him_slope": (0.07, 0.005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("him", "ols_slope")),
    "trend_him_slope_ci_low": (0.02, 0.005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("him", "ols_ci_low")),
    "trend_him_slope_ci_high": (0.12, 0.005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("him", "ols_ci_high")),
    "trend_him_p": (0.014, 0.0005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("him", "ols_p")),
    "mk_him_p": (0.037, 0.0005, "Results §Temporal Trends (total DNF rate, secondary series)", lambda: trend("him", "mk_p")),
    # --- Figure 1 caption: starters per panel (age-group, finish-time DNFs removed, known sex) ---
    "fig1_n_im": (1_131_432, CNT, "Figure 1 caption", lambda: fig1_n("im")),
    "fig1_n_him": (1_991_230, CNT, "Figure 1 caption", lambda: fig1_n("him")),
    # --- analysed population (descriptives.txt section 0; excluded_race_years.csv) ---
    "excluded_race_years_n": (54, CNT, "Methods §Study Population (events excluded: duplicates, non-standard formats, races of another brand, stopped or wholesale-coded)", lambda: grab("descriptives.txt", r"^Excluded race-years: (\d+)")),
    "excluded_duplicates_n": (14, CNT, "Methods §Study Population (duplicated race-years)", lambda: grab("descriptives.txt", r"^Excluded race-years: \d+ \(duplicated copies (\d+)")),
    "excluded_formats_n": (29, CNT, "Methods §Study Population (non-standard-format editions)", lambda: grab("descriptives.txt", r"non-standard formats (\d+)")),
    "excluded_stopped_n": (6, CNT, "Methods §Study Population (stopped or wholesale-coded editions)", lambda: grab("descriptives.txt", r"stopped or wholesale-coded editions (\d+)\)")),
    "excluded_stopped_records_n": (8360, CNT, "Methods §Study Population (records of the stopped or wholesale-coded editions)", lambda: excluded("records", "stopped")),
    "stopped_next_highest_all_pct": (35.0, PCT, "Methods §Study Population (next-highest edition, both sources)",
                                     lambda: grab_section("descriptives.txt", "Distribution of the in-race DNF share of starters, all editions", r"next-highest edition ([\d.]+)%")),
    "excluded_5150_n": (10, CNT, "Methods §Study Population (Olympic-distance 5150 editions)", lambda: excluded("race_years", "olympic_5150")),
    "excluded_aquabike_n": (11, CNT, "Methods §Study Population (aquabike editions)", lambda: excluded("race_years", "aquabike")),
    "excluded_bike_run_n": (7, CNT, "Methods §Study Population (bike-run editions)", lambda: excluded("race_years", "bike_run")),
    "excluded_do_not_use_n": (1, CNT, "Methods §Study Population (edition flagged DO-NOT-USE)", lambda: excluded("race_years", "do_not_use")),
    "excluded_records_n": (46_174, CNT, "Methods §Study Population (records in the excluded race-years)", lambda: grab("descriptives.txt", r"^Excluded records: ([\d,]+)")),
    "excluded_dup_records_n": (22_457, CNT, "Methods §Study Population (records of the duplicated copies)", lambda: excluded("records", "duplicate")),
    "excluded_formats_records_n": (9603, CNT, "Methods §Study Population (records of the non-standard formats)", lambda: excluded("records") - excluded("records", "duplicate") - excluded("records", "non_ironman") - excluded("records", "stopped")),
    "duplicates_min_overlap_pct": (Bound(">=", 99.8), BOUND, "Methods §Study Population ('shared at least 99.8%': smallest share of a dropped copy's finishers shared with the kept copy; read from the 4-decimal display of descriptives.txt)",
        lambda: min(float(v) for v in re.findall(r"shared finisher keys [\d,]+ = ([\d.]+)% of the dropped copy", txt("descriptives.txt")))),
    "records_analysed_n": (3_586_149, CNT, "Abstract; Introduction (purpose); Methods §Study Population (records analysed after the exclusions)", lambda: grab("descriptives.txt", r"^  Records analysed: ([\d,]+)")),
    "race_years_name_year_type_n": (2180, CNT, "Methods §Dataset (race-years as event name, year and type in the file; Arizona 2008 holds two races)", lambda: grab("descriptives.txt", r"Race-years as \(event name, year, event type\) in the file: ([\d,]+)")),
    "race_years_analysed_n": (2127, CNT, "Methods §Study Population (events analysed)", lambda: grab("descriptives.txt", r"in the file: [\d,]+; excluded: [\d,]+; analysed: ([\d,]+)")),
    "shortened_editions_n": (11, CNT, "Methods §Study Population (shortened editions, kept)", lambda: grab("descriptives.txt", r"^Shortened editions: (\d+)")),

    # --- event-specific course positions (dropout_point.txt section 3; event_course.csv) ---
    "course_nominal_fallback_with_distance_n": (8, CNT, "Methods §Dropout Point Localization (events with finisher distances that fall back to the nominal positions)", lambda: float(csv("event_course.csv").basis.isin(["nominal_incoherent", "nominal_scale"]).sum())),
    "course_dnf_events_im": (250, CNT, "Methods §Dropout Point Localization (IRONMAN® events with in-race DNFs carrying a usable distance)", lambda: grab_section("dropout_point.txt", "(events with in-race DNFs carrying a distance)", r"IM: events ([\d,]+)")),
    "course_dnf_events_him": (690, CNT, "Methods §Dropout Point Localization (IRONMAN® 70.3 events with in-race DNFs carrying a usable distance)", lambda: grab_section("dropout_point.txt", "(events with in-race DNFs carrying a distance)", r"HIM: events ([\d,]+)")),
    "course_deviating_im": (248, CNT, "Methods §Dropout Point Localization (IRONMAN® events with usable DNF distances whose positions deviate from nominal)", lambda: grab_section("dropout_point.txt", "(events with in-race DNFs carrying a distance)", r"IM: events [\d,]+, with own positions [\d,]+; deviating from nominal in any position ([\d,]+)")),
    "course_deviating_him": (688, CNT, "Methods §Dropout Point Localization (IRONMAN® 70.3 events with usable DNF distances whose positions deviate from nominal)", lambda: grab_section("dropout_point.txt", "(events with in-race DNFs carrying a distance)", r"HIM: events [\d,]+, with own positions [\d,]+; deviating from nominal in any position ([\d,]+)")),
    "course_top_bike_end_im_km": (183.8, 0.00005, "Methods §Dropout Point Localization (most common IRONMAN® cycling completion)", lambda: course_top("im", "cycling completion")),
    "course_top_bike_end_im_events": (187, CNT, "Methods §Dropout Point Localization (events at the most common IRONMAN® cycling completion)", lambda: course_top("im", "cycling completion", "n")),
    "course_top_bike_end_him_km": (91.9, 0.00005, "Methods §Dropout Point Localization (most common IRONMAN® 70.3 cycling completion)", lambda: course_top("him", "cycling completion")),
    "course_top_bike_end_him_events": (359, CNT, "Methods §Dropout Point Localization (events at the most common IRONMAN® 70.3 cycling completion)", lambda: course_top("him", "cycling completion", "n")),
    "course_top_finish_im_km": (226, 0.00005, "Methods §Dropout Point Localization (most common IRONMAN® finish)", lambda: course_top("im", "finish")),
    "course_top_finish_him_km": (113, 0.00005, "Methods §Dropout Point Localization (most common IRONMAN® 70.3 finish)", lambda: course_top("him", "finish")),

    # --- distance classification under the T2 rule and cross-tabulation (dropout_point.txt sections 3-4) ---
    "dist_at_cc_im_n": (5701, CNT, "Results §Dropout Localization (IRONMAN® in-race DNFs at the cycling-completion distance)", lambda: grab_section("dropout_point.txt", "\nIM (n=", r"at the cycling-completion distance \(in 'bike'\): ([\d,]+)", after="SECTION 3: DROPOUT POINT")),
    "dist_at_cc_t2_im_n": (1672, CNT, "Results §Dropout Localization (of which with a T2 split within the duration limit)", lambda: grab_section("dropout_point.txt", "\nIM (n=", r"of which with a T2 split[^:]*: ([\d,]+)", after="SECTION 3: DROPOUT POINT")),
    "dist_at_cc_him_n": (10_381, CNT, "Results §Dropout Localization (IRONMAN® 70.3 in-race DNFs at the cycling-completion distance)", lambda: grab_section("dropout_point.txt", "\nHIM (n=", r"at the cycling-completion distance \(in 'bike'\): ([\d,]+)", after="SECTION 3: DROPOUT POINT")),
    "dist_at_cc_t2_him_n": (3218, CNT, "Results §Dropout Localization (of which with a T2 split within the duration limit)", lambda: grab_section("dropout_point.txt", "\nHIM (n=", r"of which with a T2 split[^:]*: ([\d,]+)", after="SECTION 3: DROPOUT POINT")),
    "dist_seg_im_run_t2rule_pct": (42.1, PCT, "Results §Dropout Localization (running share with the T2 rule, IRONMAN®)", lambda: seg_share("im", "run", "t2_rule")),
    "dist_seg_him_run_t2rule_pct": (31.9, PCT, "Results §Dropout Localization (running share with the T2 rule, IRONMAN® 70.3)", lambda: seg_share("him", "run", "t2_rule")),
    "t2_in_bike_im_pct": (13.4, PCT, "Results §Dropout Localization (Cross-tabulation: IRONMAN® 'T2' by splits in the cycling range)", lambda: grab_section("dropout_point.txt", "\nIM (n=", after="SECTION 4", pattern=r"T2 by splits, bike by distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "t2_in_bike_him_pct": (28.3, PCT, "Results §Dropout Localization (Cross-tabulation: IRONMAN® 70.3 'T2' by splits in the cycling range)", lambda: grab_section("dropout_point.txt", "HIM (n=", after="SECTION 4", pattern=r"T2 by splits, bike by distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "t2_bike_at_cc_im_pct": (99.6, PCT, "Results §Dropout Localization (of those, at the cycling-completion distance, IRONMAN®)", lambda: grab_section("dropout_point.txt", "\nIM (n=", after="SECTION 4", pattern=r"T2 by splits and bike by distance, at the cycling-completion distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "t2_bike_at_cc_him_pct": (99.5, PCT, "Results §Dropout Localization (of those, at the cycling-completion distance, IRONMAN® 70.3)", lambda: grab_section("dropout_point.txt", "HIM (n=", after="SECTION 4", pattern=r"T2 by splits and bike by distance, at the cycling-completion distance: [\d,]+ / [\d,]+ = ([\d.]+)%")),

    # --- run-segment DNFs with the T2 rule (dropout_point.txt section 5) ---
    "run_t2_km0_im": (1672, CNT, "Results §Dropout Localization (Distance-based method) (IRONMAN® DNFs at the cycling-completion distance with a T2 split within the limit)", lambda: grab_section("dropout_point.txt", "\nIM run-segment in-race", r"T2-completers at km 0: ([\d,]+)")),
    "run_t2_km0_him": (3218, CNT, "Results §Dropout Localization (Distance-based method) (IRONMAN® 70.3 DNFs at the cycling-completion distance with a T2 split within the limit)", lambda: grab_section("dropout_point.txt", "HIM run-segment in-race", r"T2-completers at km 0: ([\d,]+)")),
    "run_at_finish_im": (4, CNT, "Methods §Dropout Hazard Along the Run (IRONMAN® in-race DNFs at the finish distance)", lambda: grab_section("dropout_point.txt", "\nIM run-segment in-race", r"without a T2 split, ([\d,]+) at the finish distance")),
    "run_at_finish_him": (20, CNT, "Methods §Dropout Hazard Along the Run (IRONMAN® 70.3 in-race DNFs at the finish distance)", lambda: grab_section("dropout_point.txt", "HIM run-segment in-race", r"without a T2 split, ([\d,]+) at the finish distance")),
    "run_strict_n_im": (13_148, CNT, "Results §Dropout Localization (Dropout hazard along the run) (IRONMAN® run DNFs without the T2 rule)", lambda: grab_section("dropout_point.txt", "\nIM run-segment in-race", r"strict rule \(beyond the cycling completion only\): n = ([\d,]+)")),
    "run_strict_n_him": (10_575, CNT, "Results §Dropout Localization (Dropout hazard along the run) (IRONMAN® 70.3 run DNFs without the T2 rule)", lambda: grab_section("dropout_point.txt", "HIM run-segment in-race", r"strict rule \(beyond the cycling completion only\): n = ([\d,]+)")),

    # --- hazard, T2 rule and strict sensitivity (dropout_hazard.txt; dropout_hazard_contrasts.csv) ---
    "hazard_strict_ratio_im": (0.81, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Discussion §Early-run withdrawal; Table S3 (strict rule, between ratio, IRONMAN®)",
        lambda: hazard("im", "first_interval_vs_rest_between", "ratio", "strict")),
    "hazard_strict_ci_low_im": (0.64, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, between CI low, IRONMAN®)",
        lambda: hazard("im", "first_interval_vs_rest_between", "ci_low", "strict")),
    "hazard_strict_ci_high_im": (1.03, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, between CI high, IRONMAN®)",
        lambda: hazard("im", "first_interval_vs_rest_between", "ci_high", "strict")),
    "hazard_strict_events_with_start_mat_im": (39, CNT, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, events in which the start of the run was observed, IRONMAN®)",
        lambda: hazard("im", "first_interval_vs_rest_between", "n_events_with_start_mat", "strict")),
    "hazard_strict_within_ratio_im": (0.76, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, within ratio, IRONMAN®)",
        lambda: hazard("im", "first_interval_vs_rest_within", "ratio", "strict")),
    "hazard_strict_within_ci_low_im": (0.61, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, within CI low, IRONMAN®)",
        lambda: hazard("im", "first_interval_vs_rest_within", "ci_low", "strict")),
    "hazard_strict_within_ci_high_im": (0.95, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, within CI high, IRONMAN®)",
        lambda: hazard("im", "first_interval_vs_rest_within", "ci_high", "strict")),
    "hazard_strict_curve_ratio_im": (0.98, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, curve ratio, IRONMAN®)",
        lambda: hazard("im", "curve_km0-3_vs_km3-end_excl_terminal", "ratio", "strict")),
    "hazard_strict_curve_ci_low_im": (0.85, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, curve CI low, IRONMAN®)",
        lambda: hazard("im", "curve_km0-3_vs_km3-end_excl_terminal", "ci_low", "strict")),
    "hazard_strict_curve_ci_high_im": (1.11, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, curve CI high, IRONMAN®)",
        lambda: hazard("im", "curve_km0-3_vs_km3-end_excl_terminal", "ci_high", "strict")),
    "hazard_strict_ratio_him": (1.15, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Discussion §Early-run withdrawal; Table S3 (strict rule, between ratio, IRONMAN® 70.3)",
        lambda: hazard("him", "first_interval_vs_rest_between", "ratio", "strict")),
    "hazard_strict_ci_low_him": (0.77, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, between CI low, IRONMAN® 70.3)",
        lambda: hazard("him", "first_interval_vs_rest_between", "ci_low", "strict")),
    "hazard_strict_ci_high_him": (1.69, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, between CI high, IRONMAN® 70.3)",
        lambda: hazard("him", "first_interval_vs_rest_between", "ci_high", "strict")),
    "hazard_strict_events_with_start_mat_him": (25, CNT, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, events in which the start of the run was observed, IRONMAN® 70.3)",
        lambda: hazard("him", "first_interval_vs_rest_between", "n_events_with_start_mat", "strict")),
    "hazard_strict_within_ratio_him": (1.30, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, within ratio, IRONMAN® 70.3)",
        lambda: hazard("him", "first_interval_vs_rest_within", "ratio", "strict")),
    "hazard_strict_within_ci_low_him": (0.81, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, within CI low, IRONMAN® 70.3)",
        lambda: hazard("him", "first_interval_vs_rest_within", "ci_low", "strict")),
    "hazard_strict_within_ci_high_him": (2.07, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, within CI high, IRONMAN® 70.3)",
        lambda: hazard("him", "first_interval_vs_rest_within", "ci_high", "strict")),
    "hazard_strict_curve_ratio_him": (0.95, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, curve ratio, IRONMAN® 70.3)",
        lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ratio", "strict")),
    "hazard_strict_curve_ci_low_him": (0.71, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, curve CI low, IRONMAN® 70.3)",
        lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ci_low", "strict")),
    "hazard_strict_curve_ci_high_him": (1.26, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (strict rule, curve CI high, IRONMAN® 70.3)",
        lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ci_high", "strict")),
    "hazard_incl_stopped_delta_im": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('left both first-interval estimates unchanged': stopped events kept equal to main, ratio and 95% CI of both contrasts, IRONMAN®)",
        lambda: float(all(hazard("im", c, q, "including_stopped") == hazard("im", c, q) for c in ("first_interval_vs_rest_between", "first_interval_vs_rest_within") for q in ("ratio", "ci_low", "ci_high")))),

    # --- regression with standard errors clustered by event (prediction_*_or.csv, prediction_summary.csv) ---
    "reg_clusters_im": (668, CNT, "Methods §Logistic Regression (events = clusters, IRONMAN®)", lambda: pred("im", "n_clusters")),
    "reg_clusters_him": (1423, CNT, "Methods §Logistic Regression (events = clusters, IRONMAN® 70.3)", lambda: pred("him", "n_clusters")),
    "or_female_im_ci_low": (1.221, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, female)",
        lambda: pred_or("im", "female", "CI_low")),
    "or_female_im_ci_high": (1.312, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, female)",
        lambda: pred_or("im", "female", "CI_high")),
    "or_female_him_ci_low": (1.345, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, female)",
        lambda: pred_or("him", "female", "CI_low")),
    "or_female_him_ci_high": (1.431, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, female)",
        lambda: pred_or("him", "female", "CI_high")),
    "or_age35_im_ci_low": (1.081, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, age_35-39)",
        lambda: pred_or("im", "age_35-39", "CI_low")),
    "or_age35_im_ci_high": (1.141, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, age_35-39)",
        lambda: pred_or("im", "age_35-39", "CI_high")),
    "or_age35_him_ci_low": (1.010, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, age_35-39)",
        lambda: pred_or("him", "age_35-39", "CI_low")),
    "or_age35_him_ci_high": (1.064, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, age_35-39)",
        lambda: pred_or("him", "age_35-39", "CI_high")),
    "or_age70_im_ci_low": (8.006, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, age_70+)",
        lambda: pred_or("im", "age_70+", "CI_low")),
    "or_age70_im_ci_high": (9.544, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, age_70+)",
        lambda: pred_or("im", "age_70+", "CI_high")),
    "or_age70_him_ci_low": (4.855, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, age_70+)",
        lambda: pred_or("him", "age_70+", "CI_low")),
    "or_age70_him_ci_high": (5.761, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, age_70+)",
        lambda: pred_or("him", "age_70+", "CI_high")),
    "or_age18_im_ci_low": (1.040, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, age_18-24)",
        lambda: pred_or("im", "age_18-24", "CI_low")),
    "or_age18_im_ci_high": (1.159, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, age_18-24)",
        lambda: pred_or("im", "age_18-24", "CI_high")),
    "or_age18_him_ci_low": (1.002, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, age_18-24)",
        lambda: pred_or("him", "age_18-24", "CI_low")),
    "or_age18_him_ci_high": (1.100, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, age_18-24)",
        lambda: pred_or("him", "age_18-24", "CI_high")),
    "or_age25_im_ci_low": (0.935, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, age_25-29)",
        lambda: pred_or("im", "age_25-29", "CI_low")),
    "or_age25_im_ci_high": (1.003, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, age_25-29)",
        lambda: pred_or("im", "age_25-29", "CI_high")),
    "or_age25_him_ci_low": (0.929, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, age_25-29)",
        lambda: pred_or("him", "age_25-29", "CI_low")),
    "or_age25_him_ci_high": (0.990, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, age_25-29)",
        lambda: pred_or("him", "age_25-29", "CI_high")),
    "or_ath_asia_him_ci_low": (1.156, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, ath_Asia)",
        lambda: pred_or("him", "ath_Asia", "CI_low")),
    "or_ath_asia_him_ci_high": (1.467, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, ath_Asia)",
        lambda: pred_or("him", "ath_Asia", "CI_high")),
    "or_ath_oceania_him_ci_low": (0.834, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, ath_Oceania)",
        lambda: pred_or("him", "ath_Oceania", "CI_low")),
    "or_ath_oceania_him_ci_high": (1.120, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, ath_Oceania)",
        lambda: pred_or("him", "ath_Oceania", "CI_high")),
    "or_ath_africa_him_ci_low": (0.925, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, ath_Africa)",
        lambda: pred_or("him", "ath_Africa", "CI_low")),
    "or_ath_africa_him_ci_high": (1.421, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, ath_Africa)",
        lambda: pred_or("him", "ath_Africa", "CI_high")),
    "or_race_asia_im_ci_low": (1.328, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, race_Asia)",
        lambda: pred_or("im", "race_Asia", "CI_low")),
    "or_race_asia_im_ci_high": (2.036, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, race_Asia)",
        lambda: pred_or("im", "race_Asia", "CI_high")),
    "or_race_europe_im_ci_low": (1.445, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, race_Europe)",
        lambda: pred_or("im", "race_Europe", "CI_low")),
    "or_race_europe_im_ci_high": (1.917, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, race_Europe)",
        lambda: pred_or("im", "race_Europe", "CI_high")),
    "or_race_africa_im_ci_low": (1.237, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, race_Africa)",
        lambda: pred_or("im", "race_Africa", "CI_low")),
    "or_race_africa_im_ci_high": (1.942, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, race_Africa)",
        lambda: pred_or("im", "race_Africa", "CI_high")),
    "or_race_oceania_im_ci_low": (0.837, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, race_Oceania)",
        lambda: pred_or("im", "race_Oceania", "CI_low")),
    "or_race_oceania_im_ci_high": (1.228, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, race_Oceania)",
        lambda: pred_or("im", "race_Oceania", "CI_high")),
    "or_race_south_america_im_ci_low": (0.638, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, race_South_America)",
        lambda: pred_or("im", "race_South_America", "CI_low")),
    "or_race_south_america_im_ci_high": (1.498, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, race_South_America)",
        lambda: pred_or("im", "race_South_America", "CI_high")),
    "or_race_europe_him_ci_low": (1.082, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, race_Europe)",
        lambda: pred_or("him", "race_Europe", "CI_low")),
    "or_race_europe_him_ci_high": (1.469, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, race_Europe)",
        lambda: pred_or("him", "race_Europe", "CI_high")),
    "or_race_africa_him_ci_low": (1.008, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, race_Africa)",
        lambda: pred_or("him", "race_Africa", "CI_low")),
    "or_race_africa_him_ci_high": (2.008, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, race_Africa)",
        lambda: pred_or("him", "race_Africa", "CI_high")),
    "or_race_asia_him_ci_low": (1.067, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, race_Asia)",
        lambda: pred_or("him", "race_Asia", "CI_low")),
    "or_race_asia_him_ci_high": (1.454, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, race_Asia)",
        lambda: pred_or("him", "race_Asia", "CI_high")),
    "or_race_south_america_him_ci_low": (0.907, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, race_South_America)",
        lambda: pred_or("him", "race_South_America", "CI_low")),
    "or_race_south_america_him_ci_high": (1.332, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, race_South_America)",
        lambda: pred_or("him", "race_South_America", "CI_high")),
    "or_race_oceania_him_ci_low": (0.689, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, race_Oceania)",
        lambda: pred_or("him", "race_Oceania", "CI_low")),
    "or_race_oceania_him_ci_high": (1.023, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, race_Oceania)",
        lambda: pred_or("him", "race_Oceania", "CI_high")),
    "or_year_im_ci_low": (0.964, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN®, race_year)",
        lambda: pred_or("im", "race_year", "CI_low")),
    "or_year_im_ci_high": (0.984, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN®, race_year)",
        lambda: pred_or("im", "race_year", "CI_high")),
    "or_year_him_ci_low": (0.983, OR3, "Results §Risk Factor Identification (clustered 95% CI, low; IRONMAN® 70.3, race_year)",
        lambda: pred_or("him", "race_year", "CI_low")),
    "or_year_him_ci_high": (1.002, OR3, "Results §Risk Factor Identification (clustered 95% CI, high; IRONMAN® 70.3, race_year)",
        lambda: pred_or("him", "race_year", "CI_high")),

    # --- professionals, in-race DNFs only (pro_rank.txt, pro_curation.txt) ---
    "pro_inrace_dnf_n": (4697, CNT, "Results §Professional Athletes (IRONMAN® professional in-race DNFs)", lambda: grab("pro_rank.txt", r"finish-time DNFs \(recorded finish time, excluded\): [\d,]+; in-race DNFs: ([\d,]+)")),
    "pro_dist_rel_median_km": (0.0, KM, "Results §Professional Athletes (median position relative to the event's cycling completion)", lambda: grab("pro_curation.txt", r"Position relative to the event's cycling completion[^:]*: median (-?[\d.]+)")),
    "pro_dist_at_cc_pct": (14.6, PCT, "Results §Professional Athletes (share at the cycling-completion distance)", lambda: grab("pro_curation.txt", r"at it: [\d,]+ \(([\d.]+)%\)")),
    "pro_dist_before_cc_pct": (41.2, PCT, "Results §Professional Athletes (share before the cycling completion)", lambda: grab("pro_curation.txt", r"before the cycling completion: [\d,]+ \(([\d.]+)%\)")),
    "pro_dist_beyond_cc_pct": (44.1, PCT, "Results §Professional Athletes (share beyond the cycling completion)", lambda: grab("pro_curation.txt", r"beyond it: [\d,]+ \(([\d.]+)%\)")),

    # --- Declarations §Availability: rerun on the deposited file (results/deposit_rerun_audit.csv) ---
    "deposit_n_starters": (3_192_429, CNT, "Declarations §Availability of data and materials", lambda: deposit("n_starters")),

    # --- Results, Discussion, Abstract, cover letter and Data availability: ranges, verbal claims and shares ---
    "records_analysed_millions": (3.6, 0.05, "Discussion (opening paragraph); cover letter ('3.6 million ... race records')",
        lambda: grab("descriptives.txt", r"^  Records analysed: ([\d,]+)") / 1e6),
    "n_starters_millions": (3.2, 0.05, "Discussion §Strengths and limitations; Conclusions ('3.2 million starts')",
        lambda: (t1("im", "Total", "starters") + t1("him", "Total", "starters")) / 1e6),
    "size_vs_nikolaidis2023": (4, 0.5, "Discussion §Strengths and limitations ('about four times the 823,459-athlete IRONMAN® 70.3 analysis'; starters / 823,459)",
        lambda: (t1("im", "Total", "starters") + t1("him", "Total", "starters")) / 823_459),
    "fm_ratio_im_25_34_min": (1.1, 0.05, "Results §DNF Rates by Sex, Age, and Region ('approximately 1.1 in the 25–34 age range in IRONMAN®', lowest)",
        lambda: fm_ratio_extreme("im", "25-29", "30-34", min)),
    "fm_ratio_im_25_34_max": (1.1, 0.05, "Results §DNF Rates by Sex, Age, and Region ('approximately 1.1 in the 25–34 age range in IRONMAN®', highest)",
        lambda: fm_ratio_extreme("im", "25-29", "30-34", max)),
    "fm_ratio_him_25_34_min": (1.2, 0.05, "Results §DNF Rates by Sex, Age, and Region ('1.2–1.3 in IRONMAN® 70.3', 25–34)",
        lambda: fm_ratio_extreme("him", "25-29", "30-34", min)),
    "fm_ratio_him_25_34_max": (1.3, 0.05, "Results §DNF Rates by Sex, Age, and Region ('1.2–1.3 in IRONMAN® 70.3', 25–34)",
        lambda: fm_ratio_extreme("him", "25-29", "30-34", max)),
    "fm_ratio_im_45_70_min": (1.3, 0.05, "Results §DNF Rates by Sex, Age, and Region ('1.3–1.4 from the 45–49 age group onward in IRONMAN®')",
        lambda: fm_ratio_extreme("im", "45-49", "70+", min)),
    "fm_ratio_im_45_70_max": (1.4, 0.05, "Results §DNF Rates by Sex, Age, and Region ('1.3–1.4 from the 45–49 age group onward in IRONMAN®')",
        lambda: fm_ratio_extreme("im", "45-49", "70+", max)),
    "fm_ratio_him_35_69_min": (1.3, 0.05, "Results §DNF Rates by Sex, Age, and Region ('from the 35–39 to the 65–69 age group in IRONMAN® 70.3')",
        lambda: fm_ratio_extreme("him", "35-39", "65-69", min)),
    "fm_ratio_him_35_69_max": (1.4, 0.05, "Results §DNF Rates by Sex, Age, and Region ('from the 35–39 to the 65–69 age group in IRONMAN® 70.3')",
        lambda: fm_ratio_extreme("him", "35-39", "65-69", max)),
    "fm_ratio_him_70": (1.2, 0.05, "Results §DNF Rates by Sex, Age, and Region ('falling to 1.2 at 70+', IRONMAN® 70.3)",
        lambda: fm_ratios("him")["70+"]),
    "annual_im_2018_pct": (8.3, PCT, "Results §Temporal Trends (in-race DNF rate; 'against 8.3% in 2018')", lambda: annual("im", 2018, "inrace")),
    "checkpoints_p10_im": (7, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('10th–90th percentile 7–18')",
        lambda: checkpoint_stat("im", 0.1)),
    "checkpoints_p90_im": (18, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('10th–90th percentile 7–18')",
        lambda: checkpoint_stat("im", 0.9)),
    "checkpoints_p10_him": (4, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('4–10')",
        lambda: checkpoint_stat("him", 0.1)),
    "checkpoints_p90_him": (10, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('4–10')",
        lambda: checkpoint_stat("him", 0.9)),
    "first_checkpoint_p10_im_km": (1.4, KM, "Results §Dropout Localization (Dropout hazard along the run) ('10th–90th percentile 1.4–7.1 km and 1.4–5.5 km')",
        lambda: grab_section("dropout_hazard.txt", "IRONMAN\n", r"first run checkpoint \(km past the event's cycling completion\): median [\d.]+ \(P10 ([\d.]+)")),
    "first_checkpoint_p90_im_km": (7.1, KM, "Results §Dropout Localization (Dropout hazard along the run) ('10th–90th percentile 1.4–7.1 km and 1.4–5.5 km')",
        lambda: grab_section("dropout_hazard.txt", "IRONMAN\n", r"first run checkpoint \(km past the event's cycling completion\): median [\d.]+ \(P10 [\d.]+, P25 [\d.]+, P75 [\d.]+, P90 ([\d.]+)")),
    "first_checkpoint_p10_him_km": (1.4, KM, "Results §Dropout Localization (Dropout hazard along the run) ('1.4–5.5 km')",
        lambda: grab_section("dropout_hazard.txt", "IRONMAN 70.3", r"first run checkpoint \(km past the event's cycling completion\): median [\d.]+ \(P10 ([\d.]+)")),
    "first_checkpoint_p90_him_km": (5.5, KM, "Results §Dropout Localization (Dropout hazard along the run) ('1.4–5.5 km')",
        lambda: grab_section("dropout_hazard.txt", "IRONMAN 70.3", r"first run checkpoint \(km past the event's cycling completion\): median [\d.]+ \(P10 [\d.]+, P25 [\d.]+, P75 [\d.]+, P90 ([\d.]+)")),
    "hazard_curve_after3_min_im": (7, 0.5, "Results §Dropout Localization (Dropout hazard along the run) ('between about 7 and 21 per 10,000 athlete-km' after the third kilometre and before the final third; bins 3–27 km)",
        lambda: curve_after3("im", min, end=27)),
    "hazard_curve_after3_max_im": (21, 0.5, "Results §Dropout Localization (Dropout hazard along the run) ('between about 7 and 21 per 10,000 athlete-km' after the third kilometre and before the final third; bins 3–27 km)",
        lambda: curve_after3("im", max, end=27)),
    "hazard_curve_argmax_after3_im": (21, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('highest at 21–22 km'; bin start)",
        lambda: curve_argmax_after3("im")),
    "hazard_curve_final_third_min_im": (4, 0.5, "Results §Dropout Localization (Dropout hazard along the run) ('declined over the final third of the marathon to about 4'; bins from 28 km)",
        lambda: curve_after3("im", min, start=28)),
    "hazard_curve_after3_min_him": (7, 0.5, "Results §Dropout Localization (Dropout hazard along the run) ('the IRONMAN® 70.3 hazard fluctuated between about 7 and 16')",
        lambda: curve_after3("him", min)),
    "hazard_curve_after3_max_him": (16, 0.5, "Results §Dropout Localization (Dropout hazard along the run) ('the IRONMAN® 70.3 hazard fluctuated between about 7 and 16')",
        lambda: curve_after3("him", max)),
    "hazard_underpowered_n": (0, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('no contrast was flagged as underpowered')",
        lambda: float(csv("dropout_hazard_contrasts.csv")["underpowered"].astype(str).eq("True").sum())),
    "excluded_stopped_im_n": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('the one stopped IRONMAN® event')",
        lambda: excluded("race_years", "stopped", "im")),
    "excluded_stopped_im_secondary_n": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('came from the secondary source, which records no distance')",
        lambda: float(csv("excluded_race_years.csv").query("reason == 'stopped' and race_type == 'im' and source == 'coachcox'").shape[0])),
    "hazard_ratio_im_1dp": (1.5, 0.05, "Discussion (opening paragraph); Conclusions; cover letter ('1.5 times')",
        lambda: hazard("im", "first_interval_vs_rest_between", "ratio")),
    "hazard_ratio_him_1dp": (1.7, 0.05, "Discussion (opening paragraph); Conclusions; cover letter ('1.7 times')",
        lambda: hazard("him", "first_interval_vs_rest_between", "ratio")),
    "ft_im_at_or_above_cutoff_pct": (20.2, PCT, "Results §DNFs with a Recorded Finish Time ('20.2% and 42.8% at or above it')",
        lambda: grab_section("dropout_point.txt", "  IM: tallest Figure 4 bar", r"at or above the typical cut-off: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_him_at_or_above_cutoff_pct": (42.8, PCT, "Results §DNFs with a Recorded Finish Time ('20.2% and 42.8% at or above it')",
        lambda: grab_section("dropout_point.txt", "  HIM: tallest Figure 4 bar", r"at or above the typical cut-off: [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_fig4_mode_start_im_h": (17.00, 0.005, "Results §DNFs with a Recorded Finish Time ('the 15-minute bin from 17.00 hours'; tallest bar of Figure 4a)",
        lambda: grab("dropout_point.txt", r"^  IM: tallest Figure 4 bar ([\d.]+)-")),
    "ft_fig4_mode_share_im_pct": (15.8, PCT, "Results §DNFs with a Recorded Finish Time ('15.8% of IRONMAN® finish-time DNFs')",
        lambda: grab("dropout_point.txt", r"^  IM: tallest Figure 4 bar [\d.]+-[\d.]+ h \(15-min bins\): [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "ft_fig4_mode_start_him_h": (8.50, 0.005, "Results §DNFs with a Recorded Finish Time ('the 10-minute bin from 8.50 hours'; tallest bar of Figure 4b)",
        lambda: grab("dropout_point.txt", r"^  HIM: tallest Figure 4 bar ([\d.]+)-")),
    "ft_fig4_mode_share_him_pct": (16.8, PCT, "Results §DNFs with a Recorded Finish Time ('16.8% in IRONMAN® 70.3')",
        lambda: grab("dropout_point.txt", r"^  HIM: tallest Figure 4 bar [\d.]+-[\d.]+ h \(10-min bins\): [\d,]+ / [\d,]+ = ([\d.]+)%")),
    "inrace_with_distance_pct": (39.3, PCT, "Discussion §Strengths and limitations ('39.3% of in-race DNFs')",
        lambda: 100 * grab("dropout_point.txt", r"Distance completed by in-race DNFs with a distance value used \(after the course rule; km, official source, n=([\d,]+)\)")
                / grab("dropout_point.txt", r"^In-race DNFs: ([\d,]+) /")),
    "deposit_country_strings": (472, CNT, "Declarations §Availability of data and materials ('from 473 to 472')", lambda: deposit("country_strings_distinct")),
    "deposit_max_count_drop": (93, CNT, "Declarations §Availability of data and materials ('lowers the record counts by at most 93')", deposit_max_count_drop),
    "deposit_changed_estimates_n": (0, CNT, "Declarations §Availability of data and materials ('leaves every rate, ratio, hazard, and regression estimate reported here unchanged at the precision shown')",
        deposit_changed_estimates),

    # --- exclusions, course rule, T2 duration limit and hazard contrasts ---
    "excluded_non_ironman_n": (5, CNT, "Methods §Study Population (races of another brand)", lambda: excluded("race_years", "non_ironman")),
    "excluded_non_ironman_records_n": (5_754, CNT, "Methods §Study Population (records of the races of another brand)", lambda: excluded("records", "non_ironman")),
    "stopped_lowest_share_pct": (46.1, PCT, "Methods §Study Population ('the lowest of these events, at 46.1%')", stopped_lowest_share),
    "race_years_file_n": (2_181, CNT, "Methods §Dataset ('2,181 events')", lambda: grab("descriptives.txt", r"Race-years \(event_ids\) in the file: ([\d,]+)")),
    "race_years_with_starters_n": (2_126, CNT, "Methods §Study Population; Discussion §Strengths and limitations ('2,126 with at least one starter'; '3.2 million starts in 2,126 events')", lambda: grab("descriptives.txt", r"Analysed race-years with at least one starter: ([\d,]+)")),
    "no_swim_editions_n": (33, CNT, "Methods §Study Population ('33 official events held without a swim')", lambda: grab("descriptives.txt", r"Editions held without a swim[^:]*: (\d+)")),
    "dist_not_used_records_n": (1_176, CNT, "Methods §Dropout Point Localization ('1,176 DNF records' whose distances are not used)", lambda: grab("dropout_point.txt", r"DNF records whose distance is not used: ([\d,]+) in")),
    "t2_limit_im_min": (20.1, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('20.1 minutes')", lambda: t2_limit_min("im")),
    "t2_limit_him_min": (12.6, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('12.6 minutes')", lambda: t2_limit_min("him")),
    "t2_finishers_median_im_min": (6.6, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('6.6 and 4.3 minutes for the finishers')", lambda: t2_median("im", "finishers")),
    "t2_finishers_median_him_min": (4.3, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('6.6 and 4.3 minutes for the finishers')", lambda: t2_median("him", "finishers")),
    "at_t2_long_im_n": (333, CNT, "Methods §Dropout Hazard Along the Run ('IRONMAN®: 333 and 3,696'); Results §Dropout Localization (Distance-based method) ('333 a longer one')", lambda: grab("dropout_point.txt", r"^  IM: at_t2 [^\n]*?at_t2_long \(T2 split longer than the limit\) ([\d,]+)")),
    "at_t2_long_him_n": (902, CNT, "Methods §Dropout Hazard Along the Run ('IRONMAN® 70.3: 902 and 6,261'); Results §Dropout Localization (Distance-based method) ('902 a longer one')", lambda: grab("dropout_point.txt", r"^  HIM: at_t2 [^\n]*?at_t2_long \(T2 split longer than the limit\) ([\d,]+)")),
    "at_t2_long_median_im_min": (34.2, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('the DNFs above it had a median T2 split of 34.2 and 23.7 minutes')", lambda: t2_median("im", "at_t2_long")),
    "at_t2_long_median_him_min": (23.7, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('the DNFs above it had a median T2 split of 34.2 and 23.7 minutes')", lambda: t2_median("him", "at_t2_long")),
    "hazard_no_checkpoint_events_im": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('One IRONMAN® and five IRONMAN® 70.3 events')", lambda: hazard("im", "first_interval_vs_rest_between", "n_events_excluded_no_checkpoint")),
    "hazard_no_checkpoint_events_him": (5, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('One IRONMAN® and five IRONMAN® 70.3 events')", lambda: hazard("him", "first_interval_vs_rest_between", "n_events_excluded_no_checkpoint")),
    "hazard_no_checkpoint_all_2019": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('all held in 2019')", lambda: no_checkpoint_all_in("2019")),
    "hazard_incl_stopped_delta_him": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('left both first-interval estimates unchanged': stopped events kept equal to main, ratio and 95% CI of both contrasts, IRONMAN® 70.3)",
        lambda: float(all(hazard("him", c, q, "including_stopped") == hazard("him", c, q) for c in ("first_interval_vs_rest_between", "first_interval_vs_rest_within") for q in ("ratio", "ci_low", "ci_high")))),
    "hazard_incl_stopped_extra_cohort_him": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('the one stopped IRONMAN® 70.3 event of the cohort')",
        lambda: hazard("him", "first_interval_vs_rest_between", "n_events_cohort", "including_stopped") - hazard("him", "first_interval_vs_rest_between", "n_events_cohort")),
    "hazard_incl_stopped_extra_excluded_him": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('recorded no run checkpoint at or beyond 1 km')",
        lambda: hazard("him", "first_interval_vs_rest_between", "n_events_excluded_no_checkpoint", "including_stopped") - hazard("him", "first_interval_vs_rest_between", "n_events_excluded_no_checkpoint")),
    "hazard_incl_stopped_curve_ratio_him": (1.40, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3 (stopped events kept, curve ratio, IRONMAN® 70.3)", lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ratio", "including_stopped")),
    "hazard_incl_stopped_curve_ci_low_him": (1.20, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ci_low", "including_stopped")),
    "hazard_incl_stopped_curve_ci_high_him": (1.66, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "curve_km0-3_vs_km3-end_excl_terminal", "ci_high", "including_stopped")),
    "hazard_t2_no_limit_ratio_im": (1.78, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Discussion §Early-run withdrawal; Table S3 (T2 rule without the duration limit)", lambda: hazard("im", "first_interval_vs_rest_between", "ratio", "t2_no_limit")),
    "hazard_t2_no_limit_ci_low_im": (1.56, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("im", "first_interval_vs_rest_between", "ci_low", "t2_no_limit")),
    "hazard_t2_no_limit_ci_high_im": (2.06, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("im", "first_interval_vs_rest_between", "ci_high", "t2_no_limit")),
    "hazard_t2_no_limit_ratio_him": (2.37, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Discussion §Early-run withdrawal; Table S3 (T2 rule without the duration limit)", lambda: hazard("him", "first_interval_vs_rest_between", "ratio", "t2_no_limit")),
    "hazard_t2_no_limit_ci_low_him": (2.00, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "first_interval_vs_rest_between", "ci_low", "t2_no_limit")),
    "hazard_t2_no_limit_ci_high_him": (2.80, 0.005, "Results §Dropout Localization (Dropout hazard along the run); Table S3", lambda: hazard("him", "first_interval_vs_rest_between", "ci_high", "t2_no_limit")),
    # --- Results and Discussion: run checkpoints beyond km 0, seasonal extremes, venue shares, professional adjustment
    #     and other verbal claims ---
    "run_median_im_km": (16.3, KM, "Results §Dropout Localization (Dropout hazard along the run) (events with a run checkpoint beyond km 0)", lambda: run_restricted("im", "median")),
    "run_p25_im_km": (8.3, KM, "Results §Dropout Localization (Dropout hazard along the run) (events with a run checkpoint beyond km 0)", lambda: run_restricted("im", "p25")),
    "run_p75_im_km": (23.5, KM, "Results §Dropout Localization (Dropout hazard along the run) (events with a run checkpoint beyond km 0)", lambda: run_restricted("im", "p75")),
    "run_median_him_km": (7.3, KM, "Results §Dropout Localization (Dropout hazard along the run) (events with a run checkpoint beyond km 0)", lambda: run_restricted("him", "median")),
    "run_p25_him_km": (1.4, KM, "Results §Dropout Localization (Dropout hazard along the run) (events with a run checkpoint beyond km 0)", lambda: run_restricted("him", "p25")),
    "run_p75_him_km": (11.1, KM, "Results §Dropout Localization (Dropout hazard along the run) (events with a run checkpoint beyond km 0)", lambda: run_restricted("him", "p75")),
    "run_share_km0_im_pct": (10.5, PCT, "Results §Dropout Localization (Dropout hazard along the run); Discussion §Early-run withdrawal ('10.5%'; from the counts)", lambda: run_restricted("im", "km0_pct")),
    "run_share_km0_him_pct": (20.2, PCT, "Results §Dropout Localization (Dropout hazard along the run); Discussion §Early-run withdrawal ('20.2%'; from the counts)", lambda: run_restricted("him", "km0_pct")),
    "annual_him_max_year": (2017, CNT, "Results §Temporal Trends (in-race DNF rate; '5.7% (2017)')", lambda: annual_argextreme("him", max, (2004, 2025), "inrace")),
    "annual_him_min_year": (2007, CNT, "Results §Temporal Trends (in-race DNF rate; '3.2% (2007)')", lambda: annual_argextreme("him", min, (2004, 2025), "inrace")),
    "annual_im_inrace_min_pct": (6.7, PCT, "Results §Temporal Trends ('with the lowest of the 23 seasons analysed, 6.7%, in 2004'; seasons of the trend tests)", lambda: grab_section("temporal_trends.txt", "In-race DNF rate (primary):", r"IM: highest annual rate \d{4} [\d.]+%; lowest annual rate \d{4} ([\d.]+)%")),
    "annual_im_inrace_min_year": (2004, CNT, "Results §Temporal Trends ('in 2004')", lambda: grab_section("temporal_trends.txt", "In-race DNF rate (primary):", r"IM: highest annual rate \d{4} [\d.]+%; lowest annual rate (\d{4}) ")),
    "starters_im_2002_n": (6018, CNT, "Results §Temporal Trends ('6,018 IRONMAN® starters in 2002')",
        lambda: float(csv("temporal_trends_annual.csv").query("race_type == 'im' and race_year == 2002")["starters"].iloc[0])),
    "starters_im_from_2010_min_n": (Bound(">=", 40000), BOUND, "Results §Temporal Trends ('at least 40,000 in every season from 2010 except 2020')",
        lambda: float(csv("temporal_trends_annual.csv").query("race_type == 'im' and race_year >= 2010 and race_year != 2020")["starters"].min())),
    "annual_im_2003_without_top_event_pct": (11.5, PCT, "Results §Temporal Trends ('without IRONMAN® Coeur d'Alene 2003, … the 2003 rate would have been 11.5%'; in-race DNF rate)",
        lambda: grab_section("temporal_trends.txt", "In-race DNF rate (primary):", r"IM: highest annual rate 2003 [\d.]+%;[^\n]*?season rate without it: ([\d.]+)%")),
    "annual_im_2003_top_event_is_cda": (1, CNT, "Results §Temporal Trends (the IRONMAN® event with the highest in-race DNF share in 2003 is Coeur d'Alene 2003)",
        lambda: float(re.search(r"IM: highest annual rate 2003 [\d.]+%;[^\n]*?in 2003: Ironman Coeur d'Alene 2003 ", txt("temporal_trends.txt")) is not None)),
    "annual_im_max_since_2010_year": (2021, CNT, "Results §Temporal Trends ('2021 ... the highest IRONMAN® rate of any season from 2010 onward')", lambda: annual_argextreme("im", max, (2010, 2026), "inrace")),
    "annual_him_2015_2018_min_pct": (5.0, PCT, "Results §Temporal Trends (in-race DNF rate; '5.0–5.7% in 2015–2018')", lambda: annual_extreme("him", min, (2015, 2018), "inrace")),
    "annual_him_2015_2018_max_pct": (5.7, PCT, "Results §Temporal Trends (in-race DNF rate; '5.0–5.7% in 2015–2018')", lambda: annual_extreme("him", max, (2015, 2018), "inrace")),
    "annual_im_2019_rose": (1, CNT, "Results §Temporal Trends ('rose to 10.0% in 2019 (against 8.3% in 2018)': above 2018)", lambda: float(annual("im", 2019, "inrace") > annual("im", 2018, "inrace"))),
    "annual_him_2019_no_rise": (1, CNT, "Results §Temporal Trends ('did not rise in 2019': below 2018)", lambda: float(annual("him", 2019, "inrace") < annual("him", 2018, "inrace"))),
    "asia_im_share_asia_pct": (50.0, PCT, "Results §DNF Rates by Sex, Age, and Region ('50.0% of the starts of Asian athletes were at events held in Asia', IRONMAN®)", lambda: venue_share("im", "Asia", ["Asia"])),
    "asia_im_share_europe_pct": (21.5, PCT, "Results §DNF Rates by Sex, Age, and Region ('21.5% in Europe', IRONMAN®)", lambda: venue_share("im", "Asia", ["Europe"])),
    "asia_im_share_higher_odds_pct": (72.1, PCT, "Discussion §Regional variation ('72.1% of their IRONMAN® starts were at events held in Asia, Europe, or Africa')", lambda: venue_share("im", "Asia", ["Asia", "Europe", "Africa"])),
    "pro_adjust_drop_min_pp": (1.3, PCT, "Results §Professional Athletes ('lowered the four rates by 1.3–1.7 percentage points')", lambda: pro_adjust_drop(min)),
    "pro_adjust_drop_max_pp": (1.7, PCT, "Results §Professional Athletes ('lowered the four rates by 1.3–1.7 percentage points')", lambda: pro_adjust_drop(max)),
    "dist_rel_median_im_km": (0.0, KM, "Results §Dropout Localization (Distance-based method) ('the median position was the cycling completion itself')",
        lambda: grab_section("dropout_point.txt", "\n  IM (n=", r"position relative to the event's cycling completion \(km\): median (-?[\d.]+)", after="Distance completed by in-race DNFs")),
    "dist_rel_median_him_km": (-14.0, KM, "Results §Dropout Localization (Distance-based method) ('a median position 14.0 km before the cycling completion')",
        lambda: grab_section("dropout_point.txt", "\n  HIM (n=", r"position relative to the event's cycling completion \(km\): median (-?[\d.]+)", after="Distance completed by in-race DNFs")),
    "unknown_sex_pct": (0.1, PCT, "Methods §Demographic and Geographic Variables ('n = 3,304; 0.1%')",
        lambda: 100 * CLAIMS["unknown_sex_starters"][3]() / (t1("im", "Total", "starters") + t1("him", "Total", "starters"))),
    "dq_pct_of_records": (0.4, PCT, "Results §Sample Characteristics ('12,592 (0.4%) were DQ')",
        lambda: 100 * (t1("im", "Total", "dq") + t1("him", "Total", "dq")) / (t1("im", "Total", "n_records") + t1("him", "Total", "n_records"))),
    "pro_base_rate_about": (75, 0.5, "Discussion §Professional athletes ('a base rate of 75% of finishers outside the top 10')", lambda: CLAIMS["pro_outside_top10_fin_pct"][3]()),
    "sens_s2_dnf_n": (37_245, CNT, "Table S6 caption (Tables list)", lambda: float(csv("sensitivity_summary.csv").set_index("analysis").loc["S2", "n_dnf"])),
    "sens_s2_dnf_pct": (6.1, PCT, "Table S6 caption (Tables list)", lambda: float(csv("sensitivity_summary.csv").set_index("analysis").loc["S2", "dnf_rate_pct"])),
    "sens_s2_auc": (0.639, 0.0005, "Table S6 caption (Tables list)", lambda: float(csv("sensitivity_summary.csv").set_index("analysis").loc["S2", "auc"])),
    "age_steeper_from_45": (1, CNT, "Results §DNF Rates by Sex, Age, and Region ('progressively steeper increases from the 45–49 age group onward', all four series)", lambda: age_steeper("45-49")),
    "women_higher_every_age": (1, CNT, "Discussion §Sex differences in DNF risk ('Women had higher DNF rates than men in every age group'); Results §DNF Rates by Sex, Age, and Region and Figure 1 (female-to-male ratio above 1 in every age group, both event types)",
                               women_higher_every_age),
    "curve_rises_last_km_im": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('before rising in the last kilometres')", lambda: curve_dip_then_rise("im", 28, min_steps=2)),

    # --- trends of the in-race DNF rate, the primary series (temporal_trends_tests.csv; seasons 2020 and 2026 excluded) ---
    "trend_inrace_im_slope": (-0.18, 0.005, "Results §Temporal Trends (in-race DNF rate, primary series; IRONMAN® OLS slope, percentage points per year)", lambda: trend("im", "ols_slope", "inrace")),
    "trend_inrace_im_slope_ci_low": (-0.31, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® slope, 95% CI low)", lambda: trend("im", "ols_ci_low", "inrace")),
    "trend_inrace_im_slope_ci_high": (-0.04, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® slope, 95% CI high)", lambda: trend("im", "ols_ci_high", "inrace")),
    "trend_inrace_im_r2": (0.26, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® R²)", lambda: trend("im", "ols_r2", "inrace")),
    "trend_inrace_im_p": (0.014, 0.0005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® slope p)", lambda: trend("im", "ols_p", "inrace")),
    "mk_inrace_im_tau": (-0.28, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® Mann-Kendall tau)", lambda: trend("im", "mk_tau", "inrace")),
    "mk_inrace_im_sen": (-0.19, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® Sen's slope)", lambda: trend("im", "sen_slope", "inrace")),
    "mk_inrace_im_p": (0.064, 0.0005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® Mann-Kendall p)", lambda: trend("im", "mk_p", "inrace")),
    "trend_inrace_him_slope": (0.04, 0.005, "Results §Temporal Trends (in-race DNF rate, primary series; IRONMAN® 70.3 OLS slope, percentage points per year)", lambda: trend("him", "ols_slope", "inrace")),
    "trend_inrace_him_slope_ci_low": (-0.01, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® 70.3 slope, 95% CI low)", lambda: trend("him", "ols_ci_low", "inrace")),
    "trend_inrace_him_slope_ci_high": (0.08, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® 70.3 slope, 95% CI high)", lambda: trend("him", "ols_ci_high", "inrace")),
    "trend_inrace_him_r2": (0.13, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® 70.3 R²)", lambda: trend("him", "ols_r2", "inrace")),
    "trend_inrace_him_p": (0.113, 0.0005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® 70.3 slope p)", lambda: trend("him", "ols_p", "inrace")),
    "mk_inrace_him_tau": (0.20, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® 70.3 Mann-Kendall tau)", lambda: trend("him", "mk_tau", "inrace")),
    "mk_inrace_him_sen": (0.03, 0.005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® 70.3 Sen's slope)", lambda: trend("him", "sen_slope", "inrace")),
    "mk_inrace_him_p": (0.216, 0.0005, "Results §Temporal Trends (in-race DNF rate; IRONMAN® 70.3 Mann-Kendall p)", lambda: trend("him", "mk_p", "inrace")),

    # --- coverage of the distance field and km-0 share of run DNFs by season (dropout_point_by_season.csv; both event types) ---
    "dist_coverage_2016_pct": (5.2, PCT, "Methods §Dropout Point Localization (official-source in-race DNFs carrying a distance value as recorded, 2016)", lambda: season_value("all", 2016, "pct_with_recorded_distance")),
    "dist_coverage_2017_pct": (2.4, PCT, "Methods §Dropout Point Localization (official-source in-race DNFs carrying a distance value as recorded, 2017)", lambda: season_value("all", 2017, "pct_with_recorded_distance")),
    "dist_coverage_from_2018_above_99": (Bound(">", 99), BOUND, "Methods §Dropout Point Localization ('for more than 99% in every season from 2018'; lowest season share of official-source in-race DNFs carrying a distance value as recorded, both event types, every season from 2018)", lambda: season_extreme("pct_with_recorded_distance", min, (2018, 9999))),
    "run_km0_share_2019_pct": (30.3, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('in 2019 it was 30.3% of the run-segment DNFs of both event types'; share recorded at km 0, the transition exit)", lambda: season_value("all", 2019, "pct_run_dnf_km0")),
    "run_km0_share_other_min_pct": (11.8, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('against 11.8–22.7% in the other seasons'; lowest, both event types; every other season with run-segment DNFs)", lambda: km0_share_other(min)),
    "run_km0_share_other_max_pct": (22.7, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('against 11.8–22.7% in the other seasons'; highest, both event types; every other season with run-segment DNFs)", lambda: km0_share_other(max)),

    # --- seasons of the hazard cohorts (dropout_hazard_cohort_seasons.csv) ---
    "hazard_cohort_first_season_im": (2017, CNT, "Results §Dropout Localization (Dropout hazard along the run) (first season of the IRONMAN® cohort)", lambda: cohort_season("im", "main", "first_season")),
    "hazard_cohort_last_season_im": (2026, CNT, "Results §Dropout Localization (Dropout hazard along the run) (last season of the IRONMAN® cohort)", lambda: cohort_season("im", "main", "last_season")),
    "hazard_cohort_first_season_him": (2016, CNT, "Results §Dropout Localization (Dropout hazard along the run) (first season of the IRONMAN® 70.3 cohort)", lambda: cohort_season("him", "main", "first_season")),
    "hazard_cohort_last_season_him": (2026, CNT, "Results §Dropout Localization (Dropout hazard along the run) (last season of the IRONMAN® 70.3 cohort)", lambda: cohort_season("him", "main", "last_season")),
    "hazard_le2019_first_season_im": (2017, CNT, "Methods §Dropout Hazard Along the Run ('the seasons up to 2019, which began in 2016 or 2017'); Table S3 (first season of the cohort of the sensitivity analysis up to 2019, IRONMAN®)", lambda: cohort_season("im", "years_le_2019", "first_season")),
    "hazard_le2019_last_season_im": (2019, CNT, "Methods §Dropout Hazard Along the Run ('the seasons up to 2019'); Table S3 (last season of the cohort of the sensitivity analysis up to 2019, IRONMAN®)", lambda: cohort_season("im", "years_le_2019", "last_season")),
    "hazard_le2019_first_season_him": (2016, CNT, "Methods §Dropout Hazard Along the Run ('the seasons up to 2019, which began in 2016 or 2017'); Table S3 (first season of the cohort of the sensitivity analysis up to 2019, IRONMAN® 70.3)", lambda: cohort_season("him", "years_le_2019", "first_season")),
    "hazard_le2019_last_season_him": (2019, CNT, "Methods §Dropout Hazard Along the Run ('the seasons up to 2019'); Table S3 (last season of the cohort of the sensitivity analysis up to 2019, IRONMAN® 70.3)", lambda: cohort_season("him", "years_le_2019", "last_season")),

    # --- course positions, race location and finish-time DNF consistency ---
    "course_own_im": (386, CNT, "Methods §Dropout Point Localization (IRONMAN® events with their own course positions)", lambda: course_own("im")),
    "course_own_him": (916, CNT, "Methods §Dropout Point Localization (IRONMAN® 70.3 events with their own course positions)", lambda: course_own("him")),
    "asia_im_share_africa_pct": (0.6, PCT, "Results §DNF Rates by Sex, Age, and Region (share of the IRONMAN® starts of Asian athletes held in Africa)", lambda: venue_share("im", "Asia", ["Africa"])),
    "ft_splits_within_5s_above_99_9": (1, CNT, "Results §DNFs with a Recorded Finish Time ('8,185 of 8,189': agreement above 99.9%)",
        lambda: float(100 * CLAIMS["ft_splits_within_5s_n"][3]() / CLAIMS["ft_all_five_splits_n"][3]() > 99.9)),

    # --- Declarations §Availability: records omitted from the deposited file, by finish status (canonical minus deposit) ---
    "deposit_omitted_fin_n": (71, CNT, "Declarations §Availability of data and materials (omitted records: finishers)", lambda: deposit_omitted("fin")),
    "deposit_omitted_dnf_n": (6, CNT, "Declarations §Availability of data and materials (omitted records: DNFs)", lambda: deposit_omitted("dnf")),
    "deposit_omitted_dq_n": (1, CNT, "Declarations §Availability of data and materials (omitted records: DQs)", lambda: deposit_omitted("dq")),
    "deposit_omitted_dns_n": (15, CNT, "Declarations §Availability of data and materials (omitted records: DNSs)", lambda: deposit_omitted("dns")),
    "ft_share_him_pre2015_below_0_1": (Bound("<", 0.1), BOUND, "Results §Temporal Trends ('from below 0.1% of starters per season before 2015'; highest IRONMAN® 70.3 finish-time DNF share of starters in a season before 2015)",
        lambda: float(csv("temporal_trends_annual.csv").query("race_type == 'him' and race_year < 2015")["finish_time_rate_pct"].max())),
    "ft_share_him_from2015_min_pct": (0.2, PCT, "Results §Temporal Trends ('0.2–0.9% from 2015 onward'; IRONMAN® 70.3 finish-time DNFs)",
        lambda: float(csv("temporal_trends_annual.csv").query("race_type == 'him' and race_year >= 2015")["finish_time_rate_pct"].min())),
    "ft_share_him_from2015_max_pct": (0.9, PCT, "Results §Temporal Trends ('0.2–0.9% from 2015 onward'; IRONMAN® 70.3 finish-time DNFs)",
        lambda: float(csv("temporal_trends_annual.csv").query("race_type == 'him' and race_year >= 2015")["finish_time_rate_pct"].max())),
    "hazard_curve_argmax_after3_him": (11, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('highest at 11–12 km'; bin start, IRONMAN® 70.3)", lambda: curve_argmax_after3("him")),

    # --- seasons with at least 100 IRONMAN® male professional starts (pro_rank.txt section 4) ---
    "pro_annual_first_season": (2003, CNT, "Results §Professional Athletes ('(2003–2026)'; first season with at least 100 IRONMAN® male professional starts)",
                                lambda: grab("pro_rank.txt", r"Years with >= 100 MPRO starts: \d+ \((\d+)-\d+\)")),
    "pro_annual_last_season": (2026, CNT, "Results §Professional Athletes ('(2003–2026)'; last season with at least 100 IRONMAN® male professional starts)",
                               lambda: grab("pro_rank.txt", r"Years with >= 100 MPRO starts: \d+ \(\d+-(\d+)\)")),

    # --- trends of the finish-time DNF rate (temporal_trends_tests.csv; seasons 2020 and 2026 excluded) ---
    "trend_ft_him_slope": (0.031, 0.0005, "Results §Temporal Trends (finish-time DNF rate; IRONMAN® 70.3 OLS slope, percentage points per year)", lambda: trend("him", "ols_slope", "finish_time")),
    "trend_ft_him_slope_ci_low": (0.016, 0.0005, "Results §Temporal Trends (finish-time DNF rate; IRONMAN® 70.3 slope, 95% CI low)", lambda: trend("him", "ols_ci_low", "finish_time")),
    "trend_ft_him_slope_ci_high": (0.046, 0.0005, "Results §Temporal Trends (finish-time DNF rate; IRONMAN® 70.3 slope, 95% CI high)", lambda: trend("him", "ols_ci_high", "finish_time")),
    "trend_ft_him_p": (Bound("<", 0.001), BOUND, "Results §Temporal Trends (finish-time DNF rate; IRONMAN® 70.3 slope p)", lambda: trend("him", "ols_p", "finish_time")),
    "trend_ft_im_slope": (0.013, 0.0005, "Results §Temporal Trends (finish-time DNF rate; IRONMAN® OLS slope, percentage points per year)", lambda: trend("im", "ols_slope", "finish_time")),
    "trend_ft_im_slope_ci_low": (0.010, 0.0005, "Results §Temporal Trends (finish-time DNF rate; IRONMAN® slope, 95% CI low)", lambda: trend("im", "ols_ci_low", "finish_time")),
    "trend_ft_im_slope_ci_high": (0.017, 0.0005, "Results §Temporal Trends (finish-time DNF rate; IRONMAN® slope, 95% CI high)", lambda: trend("im", "ols_ci_high", "finish_time")),
    "trend_ft_im_p": (Bound("<", 0.001), BOUND, "Results §Temporal Trends (finish-time DNF rate; IRONMAN® slope p)", lambda: trend("im", "ols_p", "finish_time")),

    # --- in-race DNF rate within the seasons up to 2009 and over 2010-2025 (2020 excluded) and pooled by period
    #     (temporal_trends_tests.csv, temporal_trends_periods.csv) ---
    "trend_inrace_pre2010_im_slope": (0.02, 0.005, "Results §Temporal Trends ('no significant trend within the seasons before 2010'); Discussion §Temporal trends ('no significant trend within the seasons before or after it') (in-race DNF rate, seasons 2002–2009, IRONMAN® OLS slope)", lambda: trend("im", "ols_slope", "inrace", "2002-2009")),
    "trend_inrace_pre2010_im_slope_ci_low": (-0.98, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2002–2009; IRONMAN® slope, 95% CI low)", lambda: trend("im", "ols_ci_low", "inrace", "2002-2009")),
    "trend_inrace_pre2010_im_slope_ci_high": (1.03, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2002–2009; IRONMAN® slope, 95% CI high)", lambda: trend("im", "ols_ci_high", "inrace", "2002-2009")),
    "trend_inrace_pre2010_im_p": (0.959, 0.0005, "Results §Temporal Trends (in-race DNF rate, seasons 2002–2009; IRONMAN® slope p)", lambda: trend("im", "ols_p", "inrace", "2002-2009")),
    "trend_inrace_2010_im_slope": (0.09, 0.005, "Results §Temporal Trends; Discussion §Temporal trends ('no significant trend within the seasons before or after it'; in-race DNF rate, seasons 2010–2025, IRONMAN® OLS slope)", lambda: trend("im", "ols_slope", "inrace", "2010-2025")),
    "trend_inrace_2010_im_slope_ci_low": (-0.05, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2010–2025; IRONMAN® slope, 95% CI low)", lambda: trend("im", "ols_ci_low", "inrace", "2010-2025")),
    "trend_inrace_2010_im_slope_ci_high": (0.23, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2010–2025; IRONMAN® slope, 95% CI high)", lambda: trend("im", "ols_ci_high", "inrace", "2010-2025")),
    "trend_inrace_2010_im_p": (0.178, 0.0005, "Results §Temporal Trends; Discussion §Temporal trends (in-race DNF rate, seasons 2010–2025; IRONMAN® slope p)", lambda: trend("im", "ols_p", "inrace", "2010-2025")),
    "trend_inrace_pre2010_him_slope": (0.04, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2004–2009; IRONMAN® 70.3 OLS slope)", lambda: trend("him", "ols_slope", "inrace", "2004-2009")),
    "trend_inrace_pre2010_him_slope_ci_low": (-0.62, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2004–2009; IRONMAN® 70.3 slope, 95% CI low)", lambda: trend("him", "ols_ci_low", "inrace", "2004-2009")),
    "trend_inrace_pre2010_him_slope_ci_high": (0.70, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2004–2009; IRONMAN® 70.3 slope, 95% CI high)", lambda: trend("him", "ols_ci_high", "inrace", "2004-2009")),
    "trend_inrace_pre2010_him_p": (0.881, 0.0005, "Results §Temporal Trends (in-race DNF rate, seasons 2004–2009; IRONMAN® 70.3 slope p)", lambda: trend("him", "ols_p", "inrace", "2004-2009")),
    "trend_inrace_2010_him_slope": (0.00, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2010–2025; IRONMAN® 70.3 OLS slope)", lambda: trend("him", "ols_slope", "inrace", "2010-2025")),
    "trend_inrace_2010_him_slope_ci_low": (-0.06, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2010–2025; IRONMAN® 70.3 slope, 95% CI low)", lambda: trend("him", "ols_ci_low", "inrace", "2010-2025")),
    "trend_inrace_2010_him_slope_ci_high": (0.06, 0.005, "Results §Temporal Trends (in-race DNF rate, seasons 2010–2025; IRONMAN® 70.3 slope, 95% CI high)", lambda: trend("him", "ols_ci_high", "inrace", "2010-2025")),
    "trend_inrace_2010_him_p": (Bound(">", 0.9), BOUND, "Results §Temporal Trends ('p > 0.9'; in-race DNF rate, seasons 2010–2025, IRONMAN® 70.3 slope p: 0.987 here and 0.988 on the deposited file)", lambda: trend("him", "ols_p", "inrace", "2010-2025")),
    "period_inrace_im_2002_2009_pct": (12.4, PCT, "Results §Temporal Trends (pooled in-race DNF rate, IRONMAN® 2002–2009)", lambda: period_rate("im", "2002-2009")),
    "period_inrace_im_2010_2019_pct": (8.1, PCT, "Results §Temporal Trends (pooled in-race DNF rate, IRONMAN® 2010–2019)", lambda: period_rate("im", "2010-2019")),
    "period_inrace_im_2021_2025_pct": (9.3, PCT, "Results §Temporal Trends (pooled in-race DNF rate, IRONMAN® 2021–2025)", lambda: period_rate("im", "2021-2025")),
    "period_inrace_him_2004_2009_pct": (4.0, PCT, "Results §Temporal Trends (pooled in-race DNF rate, IRONMAN® 70.3 2004–2009)", lambda: period_rate("him", "2004-2009")),
    "period_inrace_him_2010_2019_pct": (4.9, PCT, "Results §Temporal Trends (pooled in-race DNF rate, IRONMAN® 70.3 2010–2019)", lambda: period_rate("him", "2010-2019")),
    "period_inrace_him_2021_2025_pct": (4.5, PCT, "Results §Temporal Trends (pooled in-race DNF rate, IRONMAN® 70.3 2021–2025)", lambda: period_rate("him", "2021-2025")),

    # --- official source of the starters and of the finish-time DNFs (temporal_trends_annual.csv; dropout_point.txt section 1) ---
    "official_share_starters_him_2015_pct": (51.7, PCT, "Results §Temporal Trends ('grew from 51.7% in 2015 to 91.5% in 2025'; the Discussion says it in words; official-source share of IRONMAN® 70.3 starters, 2015)",
                                             lambda: annual_count("him", 2015, "starters_official_pct")),
    "official_share_starters_him_2025_pct": (91.5, PCT, "Results §Temporal Trends ('grew from 51.7% in 2015 to 91.5% in 2025'; the Discussion says it in words; official-source share of IRONMAN® 70.3 starters, 2025)",
                                             lambda: annual_count("him", 2025, "starters_official_pct")),
    "ft_official_share_pct": (98.9, PCT, "Results §Temporal Trends; the Discussion says it in words ('almost exclusively from the official source (98.9% of all finish-time DNFs)'; share from the official source, both event types)",
                              lambda: grab("temporal_trends.txt", r"all seasons, both event types: [^\n]*finish-time DNFs from the official source [\d,]+ / [\d,]+ = ([\d.]+)%")),

    # --- T2 split of the DNFs at the cycling-completion distance within the duration limit (dropout_point_t2.csv) ---
    "at_t2_median_im_min": (8.7, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('against 8.7 and 5.8 minutes for those within it')", lambda: t2_median("im", "at_t2")),
    "at_t2_median_him_min": (5.8, PCT, "Results §Dropout Localization (Dropout hazard along the run) ('against 8.7 and 5.8 minutes for those within it')", lambda: t2_median("him", "at_t2")),

    # --- the IRONMAN® 70.3 hazard in the last kilometre plotted in Figure 3d (dropout_hazard_curve.csv) ---
    "hazard_last_plotted_km_him": (16, 0.5, "Results §Dropout Localization (Dropout hazard along the run) (IRONMAN® 70.3 hazard in the last plotted kilometre, about 16 per 10,000 athlete-km)", lambda: curve_last_plotted("him")),
    "hazard_km0_is_curve_max": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('highest at the start of the run'; 'The first kilometre carried the highest hazard': km-0 bin above every other plotted bin, both event types)",
        lambda: float(all(curve_km(rt, 0) > curve_after3(rt, max, start=1) for rt in ("im", "him")))),
    "hazard_last_plotted_bin_im": (41, CNT, "Results §Dropout Localization (Dropout hazard along the run) (structural guard of 'before rising in the last kilometres': start of the last plotted IRONMAN® bin, 41 km; the number itself is not printed)", lambda: curve_last_plotted("im", "start")),
    "sens_him_africa_asia_not_robust": (1, CNT, "Results §Risk Factor Identification ('In IRONMAN® 70.3, the higher odds at African venues kept their direction but their confidence intervals included 1 in Supplementary Tables S5, S7, and S8, as did those at Asian venues in Supplementary Table S5'); Discussion §Regional variation ('not robust'; Asian venues in S5)",
        lambda: venue_ci_pattern()),
    "curve_rises_last_km_him": (1, CNT, "Results §Dropout Localization (Dropout hazard along the run) ('rose again to about 16 in the last plotted kilometre')", lambda: curve_dip_then_rise("him", 12, min_steps=1)),
    "sens_him_age18_24_s5_ci_includes_1": (1, CNT, "Results §Risk Factor Identification ('the interval of the 18–24 group in IRONMAN® 70.3 included 1 in Supplementary Table S5')",
        lambda: float((lambda r: r.CI_low < 1 < r.CI_high)(csv("sensitivity_or.csv").query("analysis == 'S1' and variable == 'age_18-24'").loc[lambda d: d.model.str.startswith("HIM")].iloc[0]))),
    "sens_him_age18_24_s5_or": (1.027, OR3, "Results §Risk Factor Identification ('(1.027, 0.982–1.075)')", lambda: float(csv("sensitivity_or.csv").query("analysis == 'S1' and variable == 'age_18-24'").loc[lambda d: d.model.str.startswith("HIM")].iloc[0].OR)),
    "sens_him_age18_24_s5_ci_low": (0.982, OR3, "Results §Risk Factor Identification ('0.982–1.075')", lambda: float(csv("sensitivity_or.csv").query("analysis == 'S1' and variable == 'age_18-24'").loc[lambda d: d.model.str.startswith("HIM")].iloc[0].CI_low)),
    "sens_him_age18_24_s5_ci_high": (1.075, OR3, "Results §Risk Factor Identification ('0.982–1.075')", lambda: float(csv("sensitivity_or.csv").query("analysis == 'S1' and variable == 'age_18-24'").loc[lambda d: d.model.str.startswith("HIM")].iloc[0].CI_high)),
    "hazard_last_plotted_bin_him": (20, CNT, "Results §Dropout Localization (Dropout hazard along the run) (structural guard of 'in the last plotted kilometre': start of that IRONMAN® 70.3 bin, 20 km; the number itself is not printed)", lambda: curve_last_plotted("him", "start")),

    # --- share of the run-segment DNFs recorded at km 0 in 2019, by event type (dropout_point_by_season.csv) ---
    "run_km0_share_2019_im_pct": (18.2, PCT, "Results §Dropout Localization (Dropout hazard along the run) (share of the IRONMAN® run-segment DNFs recorded at km 0 in 2019)", lambda: season_value("im", 2019, "pct_run_dnf_km0")),
    "run_km0_share_2019_him_pct": (48.2, PCT, "Results §Dropout Localization (Dropout hazard along the run) (share of the IRONMAN® 70.3 run-segment DNFs recorded at km 0 in 2019)", lambda: season_value("him", 2019, "pct_run_dnf_km0")),

    # --- total DNF rate (IRONMAN® Mann-Kendall p), distance coverage before 2016 (dropout_point_by_season.csv), and
    #     the bound on the professional adjustment (pro_curation.txt) ---
    "mk_im_p_total": (0.081, 0.0005, "Results §Temporal Trends (total DNF rate, secondary series; IRONMAN® Mann-Kendall p)", lambda: trend("im", "mk_p", "total")),
    "dist_coverage_before_2016_max_n": (Bound("<=", 0.0), BOUND, "Methods §Dropout Point Localization ('for none of the in-race DNFs before 2016')",
        lambda: float(csv("dropout_point_by_season.csv").query("race_type == 'all' and race_year < 2016")["n_inrace_with_recorded_distance"].max())),
    "pro_adjust_drop_below_2pp": (Bound("<", 2), BOUND, "Discussion §Professional athletes ('lowered the rates by less than 2 percentage points'; read from the 4-decimal displays of pro_curation.txt)", lambda: pro_adjust_drop(max)),
}


STANDARD_DECIMALS = {0.0: 0, 0.05: 1, 0.005: 2, 0.0005: 3, 0.00005: 4}


def printed_matches(printed, actual: float, tol: float) -> bool:
    """For the standard tolerances (counts, 1-4 decimals) the printed value must equal the persisted value
    rounded half up at that precision, so a value sitting exactly on a rounding boundary admits one digit
    only; wider tolerances ('about', ranges stated in words) keep the absolute check; a printed Bound must hold
    for the persisted value at full precision."""
    if isinstance(printed, Bound):
        return printed.holds(actual)
    if not math.isfinite(float(actual)):
        return False
    for t, n in STANDARD_DECIMALS.items():
        if abs(tol - t) < 1e-12:
            q = Decimal(1).scaleb(-n)
            return (Decimal(repr(float(printed))).quantize(q, rounding=ROUND_HALF_UP)
                    == Decimal(repr(float(actual))).quantize(q, rounding=ROUND_HALF_UP))
    return abs(actual - printed) <= tol + 1e-9


def printed_cell(printed) -> object:
    """The printed value as written to the audit CSV: the bound as text ('< 0.001'), a number as float."""
    if printed is None:
        return None
    return str(printed) if isinstance(printed, Bound) else float(printed)


def main() -> None:
    rows = []
    n_fail = n_new = n_err = n_skip = 0
    print(f"{'claim':34s} {'printed':>12s} {'actual':>12s} {'delta':>10s}  status")
    for key, (printed, tol, where, fn) in CLAIMS.items():
        try:
            actual = fn()
        except SourceAbsent as exc:  # external source not present here: reported, not failed
            n_skip += 1
            print(f"{key:34s} {'':>12s} {'SKIP':>12s} {'':>10s}  {exc}")
            rows.append({"claim": key, "printed": printed_cell(printed), "actual": None, "delta": None,
                         "status": "SKIP", "location": where, "note": str(exc)})
            continue
        except Exception as exc:  # unreadable claim = not traceable
            n_err += 1
            print(f"{key:34s} {'':>12s} {'ERROR':>12s} {'':>10s}  {type(exc).__name__}: {exc}")
            rows.append({"claim": key, "printed": printed_cell(printed), "actual": None, "delta": None,
                         "status": "ERROR", "location": where, "note": str(exc)})
            continue
        if printed is None:
            n_new += 1
            print(f"{key:34s} {'—':>12s} {actual:>12,.4g} {'':>10s}  NEW")
            rows.append({"claim": key, "printed": None, "actual": actual, "delta": None,
                         "status": "NEW", "location": where, "note": ""})
            continue
        delta = actual - (printed.value if isinstance(printed, Bound) else printed)
        ok = printed_matches(printed, actual, tol)
        n_fail += not ok
        shown = str(printed) if isinstance(printed, Bound) else f"{printed:,.4g}"
        print(f"{key:34s} {shown:>12s} {actual:>12,.4g} {delta:>+10.4g}  {'ok' if ok else 'FAIL'}")
        rows.append({"claim": key, "printed": printed_cell(printed), "actual": float(actual),
                     "delta": float(delta), "status": "ok" if ok else "FAIL",
                     "location": where, "note": "bound" if isinstance(printed, Bound) else ""})

    df = pd.DataFrame(rows)
    # Header: the dataset of the audited results (read from their headers; this audit reads no dataset, so the
    # '# Dataset:' line of reproducibility_header, which names the configured input, is replaced by it) and the code
    # fingerprints of this audit (read back when this CSV serves as the deposit rerun audit) and of the analysis code
    # that produced the audited results.
    results_code = results_analysis_code()
    current_code = code_sha256(ANALYSIS_CODE_EXCLUDE)
    dataset = results_dataset()
    base = [f"{DATASET_KEY}{dataset}; the dataset of the audited results, read from their headers"
            if line.startswith(DATASET_KEY) else line
            for line in reproducibility_header("Manuscript number audit — printed claims vs persisted results").splitlines()]
    header = ("\n".join(base) + "\n"
              + f"{AUDIT_CODE_KEY} sha256 {code_sha256(AUDIT_CODE_EXCLUDE)} (notebooks/*.py except "
                f"{', '.join(AUDIT_CODE_EXCLUDE)})\n"
              + f"{RESULTS_CODE_KEY} {('sha256 ' + results_code) if re.fullmatch(r'[0-9a-f]{64}', results_code) else results_code}\n")
    OUT_CSV.write_text(header + df.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="\n")
    stale = results_code != current_code or dataset.startswith(("mixed", "absent"))

    n_ok = int((df.status == "ok").sum())
    print(f"\n{'=' * 60}")
    print(f"{len(df)} claims: {n_ok} ok, {n_fail} FAIL, {n_new} NEW (no printed value yet), "
          f"{n_skip} SKIP (external source absent), {n_err} ERROR{' + 1 ERROR of stale results' if stale else ''}; "
          f"dataset of the audited results: {dataset}")
    if n_skip:
        print("SKIP (not checked here, external source absent): " + ", ".join(df.loc[df.status == "SKIP", "claim"]))
    if n_fail or n_err:
        print("A FAIL means the manuscript and the persisted results disagree: rewrite the prose "
              "or regenerate the artefact, then re-register the printed value here.")
    if stale:
        print(f"ERROR: the audited results were not all produced by the current analysis code on one input file "
              f"(code {results_code[:40]}, now {current_code[:12]}; dataset {dataset[:60]}): stale or mixed results, "
              f"rerun the pipeline (notebooks/README.md)")
    print(f"written to {OUT_CSV}")
    sys.exit(1 if (n_fail or n_err or stale) else 0)


if __name__ == "__main__":
    main()
