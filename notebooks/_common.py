"""
Shared data loading and utilities for the Ironman DNF paper.
All scripts import from this module to ensure a single data pipeline.

Analysed population (load_dataset)
----------------------------------
Fifty-four race-years of the merged file are excluded when the file is loaded, so every script,
and a rerun on the deposited de-identified file, analyses the same population:
  * 14 duplicated race-years: a second copy of a race that is already in the file
    (DUPLICATE_RACE_YEARS; the copy kept is the more complete one, rule documented there);
  * 29 editions of non-standard formats: Olympic-distance "5150" races recorded as IRONMAN® events,
    aquabike races (swim and cycling only), bike-run races (no swim) and one edition that the source
    itself flags "DO-NOT-USE" (NONSTANDARD_FORMAT_EVENTS, identified by the event-name rules in
    NONSTANDARD_FORMAT_RULES; load_dataset fails if the rules and the list disagree);
  * 5 races of another brand recorded as IRONMAN® events in the data: the Challenge Family races of
    the secondary source (NON_IRONMAN_EVENTS, identified by the event-name rules in NON_IRONMAN_BRANDS;
    load_dataset fails if the rules and the list disagree);
  * 6 stopped or wholesale-coded editions: editions (both sources) whose in-race DNFs are at least
    40% of their starters (STOPPED_EDITIONS; load_dataset fails unless the rule, evaluated on the
    data, flags exactly these editions).
Editions held over a shortened course (e.g. by weather) are kept; shortened_editions() lists them.
descriptives.py persists the counts by reason, the overlap of the duplicated copies, the format
editions, the non-IRONMAN races, the distribution behind the stopped-edition threshold and the
shortened editions (descriptives.txt, section "Excluded records").

Course positions (event_course_table)
-------------------------------------
total_distance_km is the cumulative course distance of the LAST timing mat an athlete crossed.
Each official-source event gets its own swim-exit, cycling-completion and finish positions: the
modal swim distance, swim + cycling distance and total distance of its finishers. Every boundary is
compared with the same tolerance, equality at two decimals (km_past). The nominal positions
(3.86 / 184.11 / 226.31 km and 1.93 / 92.05 / 113.15 km) are the fallback for events whose
finishers carry no distance or whose modal distances do not form a usable course (rules in
event_course_table). When the finishers' distances exist but do not form a usable course (basis
nominal_incoherent or nominal_scale, UNUSABLE_DISTANCE_BASES), the DNFs' distances of that event
cannot be placed on any course either: attach_course sets them to missing, so every distance-based
analysis treats those records as records without a distance.

T2 duration limit (t2_limits, run_position)
-------------------------------------------
A DNF recorded exactly at the cycling-completion distance with a T2 split crossed the timing mat at
the transition exit. It enters the run at km 0 (the T2 rule) only when its T2 split is at
most the 99th percentile of the T2 splits of the official-source finishers of its event type
(T2_LIMIT_QUANTILE, computed on the loaded data by t2_limits); with a longer T2 split it withdrew in
the transition ('at_t2_long'), like a DNF at that distance without a T2 split ('at_no_t2').
"""

import hashlib
import os
import re
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from importlib import metadata
from pathlib import Path

import pandas as pd
import numpy as np

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
# Overridable for reproduction checks (e.g. a rerun on the deposited de-identified
# file into a scratch directory, without touching the canonical results/).
DATA_PATH = Path(os.environ.get("IRONMAN_DNF_DATA_PATH",
                                PROJECT_ROOT.parent.parent / "_data_raw" / "ironman_merged.csv")).resolve()
RESULTS_DIR = Path(os.environ.get("IRONMAN_DNF_RESULTS_DIR", Path(__file__).parent / "results")).resolve()
FIGURES_DIR = Path(os.environ.get("IRONMAN_DNF_FIGURES_DIR", PROJECT_ROOT / "figures")).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = Path(os.environ.get("IRONMAN_DNF_TABLE_DIR", PROJECT_ROOT / "manuscript" / "tables")).resolve()
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# === Constants ===
SEED = 42

# Dataset reference numbers — version 4.0.0 of the companion dataset (merged file,
# collected 15 October 2026; concept DOI 10.5281/zenodo.19284673), counted over the whole
# file, before the exclusions below. Version 3.0.0, the one described in the data descriptor
# (Sci 2026, 8(9), 265), had 2,706,922 / 2,239,335 / 176,416 / 271,615 / 9,214 and carried
# 66 race-years from both sources (127,172 records); v4 removes them and extends the coverage.
DATASET_VERSION = "4.0.0"
TOTAL_RECORDS = 3_632_323
N_FINISHERS = 3_003_826
N_DNF = 216_130
N_DNS = 389_227
N_DQ = 12_798

# Nominal segment boundaries (km): the fallback course positions (event_course_table)
SWIM_END_KM_IM = 3.86
BIKE_END_KM_IM = 3.86 + 180.25  # ~184.11
FINISH_KM_IM = 3.86 + 180.25 + 42.20  # ~226.31

SWIM_END_KM_HIM = 1.93
BIKE_END_KM_HIM = 1.93 + 90.12  # ~92.05
FINISH_KM_HIM = 1.93 + 90.12 + 21.10  # ~113.15

NOMINAL_COURSE = {
    "im": {"swim_end": SWIM_END_KM_IM, "bike_end": BIKE_END_KM_IM, "finish": FINISH_KM_IM},
    "him": {"swim_end": SWIM_END_KM_HIM, "bike_end": BIKE_END_KM_HIM, "finish": FINISH_KM_HIM},
}
NOMINAL_RUN_KM = {"im": 42.20, "him": 21.10}

# Figure palette: neutral greys everywhere; colour only on the central contrast of a figure,
# i.e. men × women (Figure 1) and IRONMAN® × IRONMAN® 70.3 when both share one axes (Figure 5).
COLOR_CONTRAST_A = "#203F8C"   # Nova O2 dark blue
COLOR_CONTRAST_B = "#C45E1A"   # dark orange (the base orange #F27729 fails contrast on white)
COLOR_MALE = COLOR_CONTRAST_A
COLOR_FEMALE = COLOR_CONTRAST_B
COLOR_IM = COLOR_CONTRAST_A
COLOR_HIM = COLOR_CONTRAST_B
COLOR_NEUTRAL = "#3F3F46"      # Gray 600: main marks
COLOR_LIGHT = "#A1A1AA"        # Gray 400 light: secondary stacked component
COLOR_REFERENCE = "#71717A"    # Gray 500: reference lines and bands
COLOR_SHADE = "#EDEDEF"        # Gray 100: shaded periods
COLOR_AXIS = "#000000"

FONTS_DIR = Path(__file__).parent / "_fonts"   # Liberation Sans TTFs; not distributed (notebooks/_fonts/ is in the paper's .gitignore)
FIG_WIDTH_IN = 180 / 25.4      # Scientific Reports double-column width (180 mm)
FIG_HALF_WIDTH_IN = 89 / 25.4  # single-column width (89 mm)
FIG_DPI = 300
FIG_FONT_SIZE = 8              # pt at final size: every text of every figure, panel letters included (in bold)


def apply_figure_style() -> None:
    """Journal style, after the Scientific Reports figure guide: one sans-serif typeface (Helvetica or Arial; Liberation
    Sans has the metrics of Arial) in one size for all the text at final size, FIG_FONT_SIZE (8 pt), panel letters
    included (in bold; generate_figures.label_panels), no line thinner than 1 pt, no titles inside the figure. When the
    Liberation Sans files are in `notebooks/_fonts/` (fonts-liberation 2.1.5, SIL OFL 1.1; not distributed), they are
    registered and Liberation Sans is put first, so a system-wide Helvetica or Arial cannot replace it and the figures
    are drawn in the same face on every platform; without them the face is the first available of Helvetica, Arial,
    Liberation Sans (installed system-wide) and DejaVu Sans, which matplotlib ships. The legend of Figure 3c-d sits
    outside the axes, so a wider face cannot move it onto the data (generate_figures.legend_clear_of_axes)."""
    import matplotlib as mpl
    from matplotlib import font_manager

    registered = set()   # family names of the TTF files registered from notebooks/_fonts/
    if FONTS_DIR.exists():
        for f in sorted(FONTS_DIR.glob("*.ttf")):
            font_manager.fontManager.addfont(str(f))
            registered.add(font_manager.get_font(str(f)).family_name)
    sans = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    if "Liberation Sans" in registered:   # the registered face first, whatever is installed system-wide
        sans = ["Liberation Sans"] + [s for s in sans if s != "Liberation Sans"]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        # one size for all the text: tick labels, axis and colour-bar labels, legends, annotations (generate_figures)
        "font.size": FIG_FONT_SIZE, "axes.labelsize": FIG_FONT_SIZE, "axes.titlesize": FIG_FONT_SIZE,
        "xtick.labelsize": FIG_FONT_SIZE, "ytick.labelsize": FIG_FONT_SIZE, "legend.fontsize": FIG_FONT_SIZE,
        "axes.spines.top": False, "axes.spines.right": False,
        # every line at least 1 pt: spines, ticks, patch, hatch and marker edges (the figures set the others)
        "axes.linewidth": 1.0, "lines.linewidth": 1.4,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
        "patch.linewidth": 1.0, "hatch.linewidth": 1.0, "lines.markeredgewidth": 1.0,
        "figure.dpi": 100, "savefig.dpi": FIG_DPI, "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })

# Optimized dtypes for memory efficiency (as recommended in the Usage Notes of the companion data descriptor)
DTYPES = {
    "finish_status": "category",
    "race_type": "category",
    "source": "category",
    "age_group": "category",
    "country_iso2": "category",
    "country": "category",
    "event_name": "category",
}

INT_COLS = ["swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec", "overall_sec"]


# =====================================================================================
# Analysed population: race-years excluded at load time
# =====================================================================================
# 1. Duplicated race-years. Pairs of event_ids that hold the same race: in each pair at least
#    99.8% of the finishers of the smaller copy share bib, overall, swim and cycling times with
#    the other copy (descriptives.txt, "Excluded records", recomputed at every run by
#    duplicate_overlap()). Thirteen pairs are two copies in the secondary source (the aggregator
#    lists the race under two names); one is an official-source subset (the "Kukui Division" of the
#    2019 World Championship) of the secondary source's complete copy of that race.
#    The copy kept is the more complete one, by these rules in this order:
#      rule 1  a copy that is a subset of the other is dropped;
#      rule 2  a copy whose event type contradicts its finish times is dropped (full-distance
#              finish times filed under an IRONMAN 70.3 name);
#      rule 3  otherwise the copy with more records is kept (more athletes, i.e. more DNF and DNS
#              statuses), then the one with more finishers, then the one with more recorded
#              split times (swim, T1, cycling, T2 and run splits > 0);
#      rule 4  between copies that tie on all of these, the copy named as the series is named in
#              its other editions in the file is kept.
#    Consequence of rule 3: the kept copies of Ironman 70.3 Vineman 2014-2016 and Ironman 70.3
#    Buenos Aires 2017-2019 carry no transition (T1, T2) splits, which the dropped copies have.
DUPLICATE_RACE_YEARS: dict[str, tuple[str, str]] = {
    # dropped event_id: (kept event_id, "dropped copy = kept copy; rule")
    "7b8fe9f3-87d2-ea11-a812-000d3a5a1cf8": ("502", "2019 IRONMAN World Championship Kukui Division (official source, 320 records) "
                                                    "= subset of Ironman World Championship 2019 (secondary source, 2,434 records); rule 1"),
    "2029": ("415", "Ironman 70.3 Mar del Plata 2017 (median finish 12.0 h) = Ironman Argentina 2017 (full distance); rule 2"),
    "2030": ("465", "Ironman 70.3 Mar del Plata 2018 (median finish 10.7 h) = Ironman Argentina 2018; rule 2"),
    "2031": ("531", "Ironman 70.3 Mar del Plata 2019 (median finish 11.6 h) = Ironman Argentina 2019; rule 2"),
    "871": ("1351", "Ironman 70.3 Santa Rosa 2014 (2,218 records) = Ironman 70.3 Vineman 2014 (2,565); rule 3, records"),
    "872": ("1384", "Ironman 70.3 Santa Rosa 2015 (2,218 records) = Ironman 70.3 Vineman 2015 (2,575); rule 3, records"),
    "873": ("1409", "Ironman 70.3 Santa Rosa 2016 (1,893 records) = Ironman 70.3 Vineman 2016 (1,996); rule 3, records"),
    "759": ("1975", "Ironman 70.3 South American Championship 2017 (2,235 records) = Ironman 70.3 Buenos Aires 2017 (2,241); rule 3, records"),
    "760": ("1976", "Ironman 70.3 South American Championship 2018 (2,524 records) = Ironman 70.3 Buenos Aires 2018 (2,527); rule 3, records"),
    "642": ("1977", "Ironman 70.3 South American Championship 2019 (1,331 records, 1,119 finishers) = Ironman 70.3 Buenos Aires 2019 "
                    "(1,331 records, 1,121 finishers); rule 3, finishers"),
    "1974": ("758", "Ironman 70.3 Buenos Aires 2016 (4,629 split times) = Ironman 70.3 South American Championship 2016 "
                    "(same records and finishers, 7,825 split times); rule 3, split times"),
    "1231": ("534", "Ironman 70.3 Cancun 2012 (7,217 split times) = Ironman 70.3 Cozumel 2012 (same records and finishers, "
                    "7,220 split times); rule 3, split times"),
    "1411": ("1036", "Ironman 70.3 Alagoas 2017 = Ironman 70.3 Maceio 2017 (tie on records, finishers and split times; "
                     "'Maceio' names the 2018, 2019 and 2022 editions); rule 4"),
    "564": ("2117", "Ironman 70.3 Virginia 2019 = Ironman 70.3 Virginia's Blue Ridge 2019 (tie on records, finishers and split "
                    "times; 'Virginia's Blue Ridge' names the 2021 edition); rule 4"),
}

# 2. Non-standard formats. Identified by these event-name rules (case-insensitive) and listed
#    explicitly below; load_dataset() checks that the two agree for every event in the file.
NONSTANDARD_FORMAT_RULES: dict[str, tuple[str, str]] = {
    # reason: (regular expression on event_name, description)
    "olympic_5150": (r"\b5150\b", "Olympic-distance 5150 race recorded as an IRONMAN® event (1.5 km swim, 40 km cycling, 10 km run)"),
    "aquabike": (r"aq(?:ua|au)bike", "aquabike race (swim and cycling, no run; its athletes are recorded as DNF)"),   # 'Aqaubike': the secondary source's spelling
    "bike_run": (r"bike-run", "bike-run race (cycling and run, no swim)"),
    "do_not_use": (r"do-not-use", "edition that the source flags 'DO-NOT-USE'"),
}
NONSTANDARD_FORMAT_EVENTS: dict[str, str] = {
    # event_id: reason                                   # event name (event type in the file)
    "ce6bef41-c899-4aec-9b2e-92f35db3fc87": "olympic_5150",  # 2015 5150 Geelong (im)
    "e175fd50-f144-e411-93f8-005056951bf1": "olympic_5150",  # 2015 5150 Kraichgau (im)
    "8e3e1c74-1457-e511-9409-005056951bf1": "olympic_5150",  # 2016 5150 Kraichgau (im)
    "66c19e29-1648-e511-9409-005056951bf1": "olympic_5150",  # 2016 5150 Mont-Tremblant (im)
    "3424bad7-9ad3-e511-940c-005056951bf1": "olympic_5150",  # 2016 5150 Penghu (im)
    "ddd390da-b585-e611-9410-005056951bf1": "olympic_5150",  # 2017 5150 Kraichgau (im)
    "d6539912-73a1-e711-9419-005056951bf1": "olympic_5150",  # 2018 5150 Kraichgau (im)
    "dc2db09c-6899-e711-9419-005056951bf1": "olympic_5150",  # 2018 5150 Mont-Tremblant (im)
    "45c117aa-5de3-e811-a965-000d3a3749bc": "olympic_5150",  # 2019 5150 Kraichgau (im)
    "cac04a7a-48c6-e811-a96a-000d3a35f53a": "olympic_5150",  # 2019 5150 Mont-Tremblant (im)
    "02d85d68-0898-4743-8de6-ea333bdb3cd2": "aquabike",      # 2023 Aquabike Geelong (im)
    "a08ff39f-2626-4df7-ba26-26ec15401535": "aquabike",      # 2024 Aquabike Geelong (im)
    "59780d02-4153-401a-8a52-c2cd9fe9b935": "aquabike",      # 2025 Aquabike 70.3 Cairns (him)
    "577fb4e0-382f-4ecd-878d-04d58915a8f0": "aquabike",      # 2025 Aquabike 70.3 Geelong (him)
    "148c922d-3496-4743-bdf6-75bb74acbbda": "aquabike",      # 2025 Aquabike 70.3 Melbourne (him)
    "dba86a4d-1e9c-4f07-99a4-30379a22a9f6": "aquabike",      # 2025 Aquabike 70.3 Western Australia (him)
    "886835f5-5d6d-4268-8358-4efefcab72c9": "aquabike",      # 2025 Aquabike 70.3 Western Sydney (him)
    "d9fcadcb-3757-4bd4-9e0e-0074269d3424": "aquabike",      # 2026 Aquabike 70.3 Geelong (him)
    "f648b82c-67a0-4c71-87f5-46d18c30e88e": "aquabike",      # 2026 Aquabike 70.3 New Zealand (him)
    "11de0919-8daa-4dcf-9e4f-b9ccbdc53ed5": "aquabike",      # 2026 Aquabike 70.3 Western Sydney (him)
    "1431": "aquabike",                                      # Ironman 70.3 Atlantic City Aqaubike 2018 (him, secondary source)
    "4fa307d2-f2d8-47c3-a8c5-132728e12c99": "bike_run",      # 2021 IRONMAN 70.3 Cascais Portugal Bike-Run (him)
    "4038f7f7-2d68-48cb-af26-05653a1fc228": "bike_run",      # 2021 IRONMAN 70.3 Emilia Romagna Bike-Run (him)
    "a3adc4a3-34c0-4842-a7b0-175200f5b2b5": "bike_run",      # 2021 IRONMAN 70.3 Jonkoping Bike-Run (him)
    "105c2e2a-62d3-4694-bee0-e91576b77c99": "bike_run",      # 2021 IRONMAN 70.3 Les Sables d'Olonne Bike-Run (him)
    "3294b9f7-f930-4c8e-a6a9-7410b0bfc5d1": "bike_run",      # 2021 IRONMAN 70.3 Zell am See Kaprun Bike-Run (him)
    "f2d3dd3c-606c-4c4d-bffd-c10d4d73f4a4": "bike_run",      # 2021 IRONMAN Emilia Romagna Bike-Run (im)
    "469d87df-cb50-485e-9404-4a1512831ed6": "bike_run",      # 2021 IRONMAN Portugal-Cascais Bike-Run (im)
    "e774b571-7a1b-4e6e-afb7-6721fa001b0e": "do_not_use",    # 2023 IRONMAN Kalmar-Sweden DO-NOT-USE (see notes) (im)
}

# 3. Races of another brand recorded as IRONMAN® events in the data. The secondary source files five Challenge
#    Family races (Challenge Copenhagen 2010-2012, Challenge Weymouth 2014-2015) as full-distance
#    IRONMAN races; they are not IRONMAN events. Identified by these event-name rules (case-insensitive;
#    besides Challenge, the rules cover other long-distance triathlon brands, none of which occurs in
#    either source) and listed explicitly below; load_dataset() checks that the two agree for every
#    event in the file.
NON_IRONMAN_BRANDS: dict[str, str] = {
    # brand: regular expression on event_name
    "Challenge Family": r"\bchallenge\b",
    "Rev3": r"\brev ?3\b",
    "HITS": r"\bhits\b",
    "Xterra": r"\bx-?terra\b",
    "Ultraman": r"\bultraman\b",
    "Escape": r"\bescape\b",
    "Oceanman": r"\boceanman\b",
}
NON_IRONMAN_EVENTS: dict[str, str] = {
    # event_id: brand          # event name (event type in the file, source)
    "210": "Challenge Family",   # Challenge Copenhagen 2010 (im, secondary source)
    "211": "Challenge Family",   # Challenge Copenhagen 2011 (im, secondary source)
    "212": "Challenge Family",   # Challenge Copenhagen 2012 (im, secondary source)
    "356": "Challenge Family",   # Challenge Weymouth 2014 (im, secondary source)
    "357": "Challenge Family",   # Challenge Weymouth 2015 (im, secondary source)
}

# 4. Stopped or wholesale-coded editions. Editions of either
#    source whose in-race DNFs (DNFs without a recorded finish time) are at least STOPPED_SHARE (40%) of
#    their starters: the field was stopped, or recorded wholesale as DNF, rather than athletes
#    withdrawing one by one. The threshold lies in the gap between the 99.5th percentile of the
#    editions and the lowest listed edition. The rule is evaluated on the editions that remain after
#    exclusions 1-3; load_dataset() fails unless it flags exactly the editions listed.
#    descriptives.txt ("Excluded records") persists the distribution of the share across editions
#    (next-highest edition below the threshold, 99.5th percentile), on both sources and on each source.
STOPPED_SHARE = 0.40
STOPPED_EDITIONS: dict[str, str] = {
    # event_id: event name — what the record shows (starters = FIN + DNF + DQ; split = split time > 0)
    "e116d0b1-4eae-e111-80ae-005056956277": "2011 IRONMAN 70.3 Mallorca: Triathlon — every starter (1,631) an in-race DNF "
                                            "(no finisher): 1,628 with a swim split and 1,628 with a T1 split (1,625 with "
                                            "both), 1,535 with a cycling split and 1,396 with a T2 split; no run split, "
                                            "finish time or distance",
    "f4616994-3055-e211-b7a2-005056956277": "2013 IRONMAN 70.3 Switzerland: Triathlon — every starter (1,188) an in-race DNF "
                                            "(no finisher): every one with a swim split, 1,084 also with a T1 split, 1 with a "
                                            "cycling split; no T2 or run split, finish time or distance",
    "24dc0e64-dc91-e811-941f-005056951bf1": "2019 IRONMAN 70.3 Texas — 82.8% of starters in-race DNFs (1,850 of 2,234), "
                                            "1,581 of them at the cycling-completion distance (1,577 with a T2 split) and "
                                            "none beyond it",
    "152": "Ironman Japan 2007 (secondary source) — 52.4% of starters in-race DNFs (413 of 788): 345 with a swim split, "
           "327 with a cycling split, none with a run split (the source records no transition split for this race)",
    "1348": "Ironman 70.3 Superfrog 2014 (secondary source) — every starter (4) an in-race DNF, each with a run split but "
            "no finish time",
    "1126": "Ironman 70.3 Brazil 2008 (secondary source) — 46.1% of starters in-race DNFs (271 of 588), every one with a "
            "swim split, 263 (97%) with a cycling split, none with a run split (the source records no transition split "
            "for this race)",
}

EXCLUSION_LABELS = {
    "duplicate": "duplicated race-year (second copy of a race already in the file)",
    **{k: v[1] for k, v in NONSTANDARD_FORMAT_RULES.items()},
    "non_ironman": "race of another brand recorded as an IRONMAN® event in the data ("
                   + ", ".join(sorted(set(NON_IRONMAN_EVENTS.values()))) + ")",
    "stopped": f"stopped or wholesale-coded edition (in-race DNF ≥ {STOPPED_SHARE:.0%} of starters)",
}
EXCLUDED_EVENTS: dict[str, str] = {**{e: "duplicate" for e in DUPLICATE_RACE_YEARS}, **NONSTANDARD_FORMAT_EVENTS,
                                   **{e: "non_ironman" for e in NON_IRONMAN_EVENTS},
                                   **{e: "stopped" for e in STOPPED_EDITIONS}}
assert len(EXCLUDED_EVENTS) == (len(DUPLICATE_RACE_YEARS) + len(NONSTANDARD_FORMAT_EVENTS) + len(NON_IRONMAN_EVENTS)
                                + len(STOPPED_EDITIONS)), "event listed twice"
assert not set(v[0] for v in DUPLICATE_RACE_YEARS.values()) & set(EXCLUDED_EVENTS), "a kept copy is also excluded"
RULE_COLUMNS = ["event_id", "event_name", "finish_status", "overall_sec", "source"]   # read by load_dataset to apply the exclusions


def _name_rule_match(event_name: str, patterns: dict[str, str]) -> str | None:
    """Key of the first pattern (case-insensitive regular expression) that matches the event name, else None."""
    for key, pattern in patterns.items():
        if re.search(pattern, str(event_name), flags=re.I):
            return key
    return None


def format_rule_reason(event_name: str) -> str | None:
    """Reason of the first NONSTANDARD_FORMAT_RULES pattern that matches the event name, else None."""
    return _name_rule_match(event_name, {k: v[0] for k, v in NONSTANDARD_FORMAT_RULES.items()})


def brand_rule_match(event_name: str) -> str | None:
    """Brand of the first NON_IRONMAN_BRANDS pattern that matches the event name, else None."""
    return _name_rule_match(event_name, NON_IRONMAN_BRANDS)


def _check_name_rules(events: pd.DataFrame, rule, listed_all: dict[str, str], what: str) -> None:
    """Fail when an event-name rule and its explicit list disagree for the events of the file
    (events: one row per event, columns event_id and event_name)."""
    by_rule = {str(e): rule(n) for e, n in zip(events["event_id"], events["event_name"])}
    by_rule = {e: r for e, r in by_rule.items() if r is not None}
    present = set(events["event_id"].astype(str))
    listed = {e: r for e, r in listed_all.items() if e in present}
    if by_rule != listed:
        missing = {e: r for e, r in by_rule.items() if listed.get(e) != r}
        extra = {e: r for e, r in listed.items() if by_rule.get(e) != r}
        raise AssertionError(f"{what}: event-name rules and the explicit list disagree "
                             f"(matched by the rules, not listed: {missing}; listed, not matched: {extra})")


def check_format_rules(events: pd.DataFrame) -> None:
    """Fail when the event-name rules and the explicit list of non-standard formats disagree."""
    _check_name_rules(events, format_rule_reason, NONSTANDARD_FORMAT_EVENTS, "non-standard formats (NONSTANDARD_FORMAT_EVENTS)")


def check_brand_rules(events: pd.DataFrame) -> None:
    """Fail when the brand rules and the explicit list of non-IRONMAN races disagree."""
    _check_name_rules(events, brand_rule_match, NON_IRONMAN_EVENTS, "non-IRONMAN races (NON_IRONMAN_EVENTS)")


def edition_inrace_share(df: pd.DataFrame) -> pd.DataFrame:
    """Per edition (event_id, as str): source, event name, starters (FIN + DNF + DQ), in-race DNFs
    (DNFs without a recorded finish time) and their share of the starters (%)."""
    fs = df["finish_status"].astype("string")
    st = df[fs.isin(["FIN", "DNF", "DQ"]).fillna(False).to_numpy()]
    inrace = ((st["finish_status"].astype("string") == "DNF").fillna(False)
              & ~(st["overall_sec"].astype("Float64").fillna(0) > 0)).to_numpy()
    g = pd.DataFrame({"event_id": st["event_id"].astype("string").to_numpy(), "inrace": inrace,
                      "source": st["source"].astype(str).to_numpy(),
                      "event_name": st["event_name"].astype(str).to_numpy() if "event_name" in st.columns else ""})
    t = g.groupby("event_id").agg(source=("source", "first"), event_name=("event_name", "first"),
                                  starters=("inrace", "size"), inrace_dnf=("inrace", "sum"))
    t["inrace_share_pct"] = t["inrace_dnf"] / t["starters"] * 100
    return t


def check_stopped_rule(df: pd.DataFrame) -> None:
    """Fail unless the stopped-edition rule (editions of either source with in-race DNFs >= STOPPED_SHARE
    of their starters), evaluated on df (after exclusions 1-3), flags exactly STOPPED_EDITIONS (those present)."""
    t = edition_inrace_share(df)
    flagged = set(t.index[t["inrace_dnf"] / t["starters"] >= STOPPED_SHARE])   # ratio, not the % (0.4 * 100 is not 40 in floating point)
    listed = set(STOPPED_EDITIONS) & set(t.index)
    if flagged != listed:
        raise AssertionError(f"stopped editions: the rule flags {sorted(flagged - listed)} not listed and does not flag "
                             f"listed {sorted(listed - flagged)} (STOPPED_EDITIONS, in-race DNF >= {STOPPED_SHARE:.0%} of starters)")


def exclude_race_years(df: pd.DataFrame, keep_stopped: bool = False) -> pd.DataFrame:
    """Drop the records of the excluded race-years (EXCLUDED_EVENTS). Records without an event_id
    (the official records without a finish status) are kept; they leave the analyses as records
    without a finish status. Requires event_id; checks the format and brand rules when event_name is
    present and the stopped-edition rule when finish_status, overall_sec and source are present.
    keep_stopped=True keeps the stopped or wholesale-coded editions (hazard sensitivity analysis)."""
    if "event_id" not in df.columns:
        raise ValueError("event_id column required to apply the exclusions")
    ids = df["event_id"].astype("string")
    if "event_name" in df.columns:
        ev = pd.DataFrame({"event_id": ids, "event_name": df["event_name"].astype("string")}).dropna().drop_duplicates("event_id")
        check_format_rules(ev)
        check_brand_rules(ev)
    first = [e for e, r in EXCLUDED_EVENTS.items() if r != "stopped"]
    df = df[~ids.isin(first).fillna(False).to_numpy()]
    if {"finish_status", "overall_sec", "source"} <= set(df.columns):
        check_stopped_rule(df)
    if not keep_stopped:
        df = df[~df["event_id"].astype("string").isin(list(STOPPED_EDITIONS)).fillna(False).to_numpy()]
    return df


def load_dataset(columns: list[str] | None = None, exclude: bool = True, keep_stopped: bool = False) -> pd.DataFrame:
    """Load the merged Ironman dataset with optimized dtypes.

    exclude=True (default): the excluded race-years (EXCLUDED_EVENTS) are dropped, so every script
    analyses the same population; the columns the exclusion rules read (RULE_COLUMNS) are read for
    that and dropped again when they were not requested. keep_stopped=True keeps the stopped or
    wholesale-coded editions (dropout_hazard.py sensitivity analysis). exclude=False returns the
    whole file (descriptives.py counts the exclusions from it)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"dataset file not found: {DATA_PATH} (the merged file ironman_merged.csv, dataset v{DATASET_VERSION}, or its "
            f"de-identified deposit ironman_dataset_deidentified.csv, Zenodo DOI 10.5281/zenodo.23421027; set "
            f"IRONMAN_DNF_DATA_PATH to its location)")
    cols = None if columns is None else list(dict.fromkeys(list(columns) + (RULE_COLUMNS if exclude else [])))
    df = pd.read_csv(DATA_PATH, usecols=cols, dtype=DTYPES, low_memory=False)
    # Cast int columns to nullable Int32 where present
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
    if exclude:
        df = exclude_race_years(df, keep_stopped=keep_stopped)
        if columns is not None:
            df = df[list(dict.fromkeys(columns))]
    return df


def load_dnf(columns: list[str] | None = None) -> pd.DataFrame:
    """Load only DNF records."""
    if columns and "finish_status" not in columns:
        columns = ["finish_status"] + columns
    df = load_dataset(columns)
    return df[df["finish_status"] == "DNF"].copy()


def duplicate_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Overlap and completeness of the two copies of each duplicated race-year (DUPLICATE_RACE_YEARS).

    df: the whole file (load_dataset(..., exclude=False)) with event_id, event_name, finish_status,
    bib, overall_sec and the five split columns. Finisher key = (bib, overall, swim, cycling times)
    of records with an overall time > 0 and a bib; shared keys are reported as a share of each copy.
    """
    out = []
    ids = df["event_id"].astype("string")
    for dropped, (kept, note) in DUPLICATE_RACE_YEARS.items():
        row = {"dropped_event_id": dropped, "kept_event_id": kept, "rule": note.rsplit("; ", 1)[-1]}
        keys = {}
        for tag, eid in (("dropped", dropped), ("kept", kept)):
            x = df[(ids == eid).fillna(False).to_numpy()]
            fs = x["finish_status"].astype("string")
            ov = x["overall_sec"].fillna(0)
            f = x[(ov > 0).to_numpy() & x["bib"].notna().to_numpy()]
            k = set(f["bib"].astype(str) + "|" + f["overall_sec"].astype(str) + "|" + f["swim_sec"].astype(str)
                    + "|" + f["bike_sec"].astype(str))
            keys[tag] = k
            row.update({
                f"{tag}_name": str(x["event_name"].iloc[0]) if len(x) else "",
                f"{tag}_records": len(x),
                f"{tag}_finishers": int((fs == "FIN").sum()),
                f"{tag}_dnf": int((fs == "DNF").sum()),
                f"{tag}_split_times": int(sum((x[c].fillna(0) > 0).sum() for c in INT_COLS if c != "overall_sec")),
                f"{tag}_finisher_keys": len(k),
            })
        shared = len(keys["dropped"] & keys["kept"])
        row["shared_finisher_keys"] = shared
        row["shared_pct_of_dropped"] = 100 * shared / len(keys["dropped"]) if keys["dropped"] else float("nan")
        row["shared_pct_of_kept"] = 100 * shared / len(keys["kept"]) if keys["kept"] else float("nan")
        row["shared_pct_of_smaller"] = 100 * shared / min(len(keys["dropped"]), len(keys["kept"])) if keys["dropped"] and keys["kept"] else float("nan")
        out.append(row)
    return pd.DataFrame(out)


# =====================================================================================
# Course positions: event-specific swim exit, cycling completion and finish
# =====================================================================================
POSITION_DECIMALS = 2          # common tolerance: two positions are equal when they agree at 2 decimals
COURSE_COLUMNS = ["event_id", "source", "race_type", "finish_status", "overall_sec",
                  "swim_distance_km", "bike_distance_km", "run_distance_km", "total_distance_km"]
COURSE_SCALE_MIN = 0.90        # a modal total below 90% of nominal ...
COURSE_TIME_GUARD = 0.95       # ... is a shortened course only if the finishers were faster than 95% of typical
COURSE_RUN_MAX = 1.25          # a modal run distance above 125% of the nominal run is not a usable run length
SHORTENED_DISTANCE = 0.90      # shortened edition: modal total below 90% of nominal (own course) ...
SHORTENED_TIME = 0.75          # ... or median finisher time below 75% of the typical median of its event type
SHORTENED_NAME = r"\bpartial\b"  # ... or an event name that declares a partial course (e.g. "(Partial Bike)")
NO_MAT_LABEL = "no timing mat crossed"   # distance 0: the field's default value (compatible with withdrawal during the swim)
# Course bases whose finishers carry distances that do not form a usable course: the recorded distances
# of every record of these events are not used (attach_course sets them to missing)
UNUSABLE_DISTANCE_BASES = ("nominal_incoherent", "nominal_scale")
T2_LIMIT_QUANTILE = 0.99       # T2 duration limit: 99th percentile of the official-source finishers' T2 splits (t2_limits)

COURSE_BASIS_LABELS = {
    "own": "own positions (modal finisher distances)",
    "own_nominal_run": "own swim exit and cycling completion; nominal run length (modal run distance > 125% of nominal)",
    "nominal_no_distance": "nominal positions: no finisher with a total distance",
    "nominal_incoherent": "nominal positions: modal total distance <= 0 or not equal to modal swim + cycling + run distances",
    "nominal_scale": "nominal positions: modal total < 90% of nominal with typical finish times (distances recorded on another scale)",
}


def km_past(distance, boundary) -> pd.Series:
    """Position of a recorded distance relative to a course position, rounded to POSITION_DECIMALS:
    0 = at the position (equal at two decimals), < 0 before it, > 0 beyond it. The single
    comparison rule used at every boundary (swim exit, cycling completion, finish)."""
    d = pd.Series(distance, copy=False).astype("float64")
    b = boundary.to_numpy(dtype="float64") if isinstance(boundary, pd.Series) else boundary
    return (d - b).round(POSITION_DECIMALS)


def _event_mode(keys: pd.Series, values: pd.Series) -> pd.Series:
    """Modal value per event (values rounded to 4 decimals; ties: the smallest value)."""
    d = pd.DataFrame({"k": keys.astype("string").to_numpy(), "v": values.astype("float64").round(4).to_numpy()}).dropna()
    if d.empty:
        return pd.Series(dtype="float64")
    c = d.groupby(["k", "v"]).size().reset_index(name="n").sort_values(["k", "n", "v"], ascending=[True, False, True])
    return c.drop_duplicates("k").set_index("k")["v"]


def event_finish_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Per event (both sources): event type, finishers and median overall time of its finishers
    (hours), with time_ratio = event median / typical median of its event type, the typical median
    being the median of the event medians of that event type."""
    fs = df["finish_status"].astype("string")
    ov = df["overall_sec"].astype("Float64").fillna(0)
    fin = df[((fs == "FIN").fillna(False) & (ov > 0)).to_numpy()]
    g = fin.groupby(fin["event_id"].astype("string"), observed=True)
    ev = pd.DataFrame({"race_type": g["race_type"].first().astype(str), "source": g["source"].first().astype(str),
                       "n_finishers": g.size(), "median_finish_h": g["overall_sec"].median().astype("float64") / 3600})
    typical = ev.groupby("race_type")["median_finish_h"].median()
    ev["typical_median_h"] = ev["race_type"].map(typical)
    ev["time_ratio"] = ev["median_finish_h"] / ev["typical_median_h"]
    return ev


def event_course_table(df: pd.DataFrame) -> pd.DataFrame:
    """Course positions of each official-source event with finishers (index: event_id as str).

    df: all records as loaded (both sources: the typical finish time of each event type is taken
    over all events), with COURSE_COLUMNS. From the event's finishers (FIN):
      swim      = modal swim_distance_km                  -> swim exit
      swim_bike = modal swim_distance_km + bike_distance_km -> cycling completion
      run       = modal run_distance_km
      total     = modal total_distance_km                 -> finish
    total_distance_km (the field that DNFs carry) is on the scale of the segment fields when
    total = swim + bike + run. Basis of the positions (COURSE_BASIS_LABELS):
      own                 total present, > 0 and equal to swim_bike + run at 2 decimals,
                          0 <= swim <= swim_bike < total, and either total >= 90% of nominal or the
                          finishers' median time < 95% of the typical median of the event type
                          (a shortened course rather than distances on another scale);
                          run <= 125% of the nominal run;
      own_nominal_run     as own, but run > 125% of the nominal run: finish = swim_bike + nominal run;
      nominal_no_distance no finisher with a total distance;
      nominal_incoherent  total <= 0 or total != swim_bike + run (or inconsistent order);
      nominal_scale       total < 90% of nominal while the finish times are typical.
    run_len = finish - bike_end. deviates_from_nominal: any of the three positions differs from
    the nominal one at 2 decimals.
    """
    missing = sorted(set(COURSE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"event_course_table: columns missing: {missing}")
    times = event_finish_medians(df)
    src = df["source"].astype("string")
    fs = df["finish_status"].astype("string")
    fin = df[((src == "official") & (fs == "FIN")).fillna(False).to_numpy()]
    eid = fin["event_id"].astype("string")
    t = pd.DataFrame({"race_type": fin.groupby(eid, observed=True)["race_type"].first().astype(str)})
    t["swim"] = _event_mode(eid, fin["swim_distance_km"])
    t["swim_bike"] = _event_mode(eid, fin["swim_distance_km"].astype("float64") + fin["bike_distance_km"].astype("float64"))
    t["run"] = _event_mode(eid, fin["run_distance_km"])
    t["total"] = _event_mode(eid, fin["total_distance_km"])
    t = t.join(times[["n_finishers", "median_finish_h", "time_ratio"]])
    nom = pd.DataFrame({k: t["race_type"].map(lambda r, k=k: NOMINAL_COURSE[r][k]) for k in ("swim_end", "bike_end", "finish")})
    nom_run = t["race_type"].map(NOMINAL_RUN_KM)
    has_total = t["total"].notna()
    coherent = (has_total & (t["total"] > 0) & ((t["total"] - t["swim_bike"] - t["run"]).round(POSITION_DECIMALS) == 0)
                & (t["swim"] >= 0) & (t["swim_bike"] >= t["swim"]) & (t["total"] > t["swim_bike"])).fillna(False)
    scale_bad = coherent & (t["total"] < COURSE_SCALE_MIN * nom["finish"]) & ~(t["time_ratio"] < COURSE_TIME_GUARD)
    run_bad = coherent & ~scale_bad & (t["run"] > COURSE_RUN_MAX * nom_run)
    t["basis"] = np.select([~has_total, ~coherent, scale_bad, run_bad],
                           ["nominal_no_distance", "nominal_incoherent", "nominal_scale", "own_nominal_run"], "own")
    own = t["basis"].isin(["own", "own_nominal_run"])
    t["swim_end"] = np.where(own, t["swim"], nom["swim_end"])
    t["bike_end"] = np.where(own, t["swim_bike"], nom["bike_end"])
    t["finish"] = np.where(t["basis"] == "own", t["total"],
                           np.where(t["basis"] == "own_nominal_run", t["swim_bike"] + nom_run, nom["finish"]))
    t["run_len"] = t["finish"] - t["bike_end"]
    t["deviates_from_nominal"] = ((t["swim_end"] - nom["swim_end"]).round(POSITION_DECIMALS).ne(0)
                                  | (t["bike_end"] - nom["bike_end"]).round(POSITION_DECIMALS).ne(0)
                                  | (t["finish"] - nom["finish"]).round(POSITION_DECIMALS).ne(0))
    t.index.name = "event_id"
    return t


def attach_course(df: pd.DataFrame, course: pd.DataFrame) -> pd.DataFrame:
    """Add each record's course positions (swim_end, bike_end, finish, run_len, course_basis) from
    event_course_table(); records of events without a row (secondary source, or no finishers) get
    the nominal positions of their event type (course_basis 'nominal_no_course').

    Distances not used: the records of events whose basis is in UNUSABLE_DISTANCE_BASES (finishers'
    distances that do not form a usable course: nominal_incoherent, nominal_scale) get
    total_distance_km set to missing, and distance_not_used marks those that carried a value. Every
    distance-based analysis reads total_distance_km after this function (directly, or through
    infer_dropout_from_distance and run_position), so it treats them as records without a distance.
    Idempotent."""
    c = course.reindex(df["event_id"].astype("string").to_numpy())
    rt = df["race_type"].astype(str).to_numpy()
    for k in ("swim_end", "bike_end", "finish"):
        nom = np.array([NOMINAL_COURSE[r][k] for r in rt], dtype="float64")
        df[k] = np.where(c[k].notna().to_numpy(), c[k].to_numpy(dtype="float64", na_value=np.nan), nom)
    df["run_len"] = df["finish"] - df["bike_end"]
    df["course_basis"] = c["basis"].fillna("nominal_no_course").to_numpy()
    if "total_distance_km" in df.columns:
        unusable = df["course_basis"].isin(UNUSABLE_DISTANCE_BASES).to_numpy()
        before = df["distance_not_used"].to_numpy(dtype=bool) if "distance_not_used" in df.columns else np.zeros(len(df), dtype=bool)
        df["distance_not_used"] = unusable & (df["total_distance_km"].notna().to_numpy() | before)
        df["total_distance_km"] = df["total_distance_km"].astype("float64").mask(unusable)
    return df


def infer_dropout_from_distance(df: pd.DataFrame, course: pd.DataFrame) -> pd.DataFrame:
    """Segment of the last timing mat crossed, from total_distance_km (Method 1 — official source only).

    Classes, with every position compared by km_past (equality at two decimals) against the
    event's own course positions (event_course_table; nominal fallback):
      'no timing mat crossed'  distance 0 (the field's default; compatible with withdrawal in the swim)
      'swim'   0 < distance, at or before the swim exit
      'bike'   beyond the swim exit, at or before the cycling completion
      'run'    beyond the cycling completion
      'unknown'  no distance value (none recorded, or not used: UNUSABLE_DISTANCE_BASES, attach_course)
    T1 and T2 are not separate classes: transitions do not accumulate distance (the finishers'
    total distance is the sum of the swim, cycling and run distances). A DNF at the cycling-completion
    distance is 'bike' here; with a T2 split within the duration limit it enters the run at km 0 under
    the T2 rule of the hazard (run_position), which this classification does not apply.
    """
    if "total_distance_km" not in df.columns:
        raise ValueError("total_distance_km column required")
    if "bike_end" not in df.columns:
        df = attach_course(df, course)
    d = df["total_distance_km"].astype("float64")
    cls = np.select(
        [d.isna().to_numpy(), (d <= 0).to_numpy(), (km_past(d, df["swim_end"]) <= 0).to_numpy(),
         (km_past(d, df["bike_end"]) <= 0).to_numpy()],
        ["unknown", NO_MAT_LABEL, "swim", "bike"], "run")
    df["dropout_distance"] = pd.Categorical(cls, categories=[NO_MAT_LABEL, "swim", "bike", "run", "unknown"])
    return df


RUN_CLASS_LABELS = {
    "before": "before the cycling completion (no mat crossed, swim or cycling segment)",
    "at_t2": "at the cycling-completion distance with a T2 split within the duration limit (crossed the timing mat at the transition exit: enters the run at km 0)",
    "at_t2_long": "at the cycling-completion distance with a T2 split longer than the limit (withdrew in the transition; not modelled in the main analysis)",
    "at_no_t2": "at the cycling-completion distance without a T2 split (withdrew in the transition)",
    "run": "beyond the cycling completion and before the finish distance",
    "at_finish": "at the finish distance (crossed the finish mat, no finish time)",
    "beyond": "beyond the finish distance",
}
# Run DNFs under each rule (run classes counted as run-segment DNFs)
RUN_RULES: dict[str, tuple[str, ...]] = {
    "t2_rule": ("at_t2", "run"),                       # main analysis (T2 rule)
    "strict": ("run",),                                # sensitivity: no DNF at the cycling-completion distance enters the run
    "t2_no_limit": ("at_t2", "at_t2_long", "run"),     # sensitivity: T2 rule without the duration limit
}


def t2_limits(df: pd.DataFrame) -> dict[str, float]:
    """T2 duration limit (seconds) per event type: the T2_LIMIT_QUANTILE (99th) percentile of t2_sec
    (linear interpolation) among the official-source finishers of df with t2_sec > 0. df: the loaded
    data, with source, finish_status, race_type and t2_sec."""
    src = df["source"].astype("string")
    fs = df["finish_status"].astype("string")
    t2 = df["t2_sec"].astype("Float64")
    m = ((src == "official") & (fs == "FIN") & (t2.fillna(0) > 0)).fillna(False).to_numpy()
    x = pd.DataFrame({"race_type": df["race_type"].astype(str).to_numpy()[m], "t2": t2.to_numpy(dtype="float64", na_value=np.nan)[m]})
    lim = {rt: float(g["t2"].quantile(T2_LIMIT_QUANTILE)) for rt, g in x.groupby("race_type")}
    missing = {"im", "him"} - set(lim)
    if missing:
        raise ValueError(f"t2_limits: no official-source finisher with a T2 split for {sorted(missing)}")
    return lim


def run_position(df: pd.DataFrame, course: pd.DataFrame, t2_limit: dict[str, float]) -> pd.DataFrame:
    """Running position of in-race DNFs (the hazard's and dropout_point.py section 5's run-DNF
    definition). t2_limit: T2 duration limit per event type in seconds (t2_limits).

    km_run = km_past(total_distance_km, event's cycling completion) and run_class (RUN_CLASS_LABELS):
    'before' km_run < 0; 'at_t2' km_run = 0 with 0 < t2_sec <= the limit of its event type;
    'at_t2_long' km_run = 0 with t2_sec above the limit; 'at_no_t2' km_run = 0 without a T2 split;
    'run' km_run > 0 and before the finish (km_past(distance, finish) < 0); 'at_finish' equal to the
    finish; 'beyond' beyond it; 'no_distance' without a distance value (none recorded, or not used:
    attach_course). Run DNFs (RUN_RULES): main rule (T2 rule) = 'at_t2' (at km 0) + 'run'; strict
    rule = 'run' only; T2 rule without the duration limit = 'at_t2' + 'at_t2_long' + 'run'. Records
    at the finish distance are not run DNFs: their interval would have zero width (they crossed the
    last mat of the course).
    """
    missing = {"im", "him"} - set(t2_limit)
    if missing:
        raise ValueError(f"run_position: T2 limit missing for {sorted(missing)}")
    if "bike_end" not in df.columns:
        df = attach_course(df, course)
    d = df["total_distance_km"].astype("float64")
    k = km_past(d, df["bike_end"]).to_numpy()
    to_finish = km_past(d, df["finish"]).to_numpy()
    t2 = df["t2_sec"].astype("Float64").fillna(0).to_numpy(dtype="float64")
    lim = df["race_type"].astype(str).map(t2_limit).to_numpy(dtype="float64")
    has_t2 = t2 > 0
    df["km_run"] = k
    df["run_class"] = np.select(
        [np.isnan(k), k < 0, (k == 0) & has_t2 & (t2 <= lim), (k == 0) & has_t2, k == 0, to_finish < 0, to_finish == 0],
        ["no_distance", "before", "at_t2", "at_t2_long", "at_no_t2", "run", "at_finish"], "beyond")
    df.loc[df["run_class"].isin(["at_t2", "at_t2_long", "at_no_t2"]), "km_run"] = 0.0
    return df


def shortened_editions(df: pd.DataFrame, course: pd.DataFrame) -> pd.DataFrame:
    """Editions held over a shortened course (kept in every analysis; listed for the Methods).

    Flagged when (distance) the official-source event has its own course (basis 'own') with a modal
    finisher total distance below SHORTENED_DISTANCE (90%) of nominal, or (time) the median overall
    time of its finishers is below SHORTENED_TIME (75%) of the typical median of its event type
    (both sources; catches editions whose distance fields keep the nominal values and
    secondary-source editions, which carry no distance), or (name) its name declares a partial
    course (SHORTENED_NAME). Editions held without a swim are not counted here (descriptives.txt
    reports them apart)."""
    ev = event_finish_medians(df)
    names = df[["event_id", "event_name"]].dropna().drop_duplicates("event_id")
    ev = ev.join(names.assign(event_id=names["event_id"].astype("string")).set_index("event_id")["event_name"].astype(str))
    ev = ev.join(course[["basis", "total"]])
    nom = ev["race_type"].map(lambda r: NOMINAL_COURSE[r]["finish"])
    ev["distance_share"] = ev["total"] / nom
    by_dist = (ev["basis"] == "own") & (ev["distance_share"] < SHORTENED_DISTANCE)
    by_time = ev["time_ratio"] < SHORTENED_TIME
    by_name = ev["event_name"].str.contains(SHORTENED_NAME, case=False, regex=True)
    ev["criterion"] = [" + ".join(c for c, f in (("distance", a), ("time", b), ("name", n)) if f)
                       for a, b, n in zip(by_dist, by_time, by_name)]
    out = ev[by_dist | by_time | by_name].copy()
    out.index.name = "event_id"
    return out.sort_values(["race_type", "event_name"])


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


# Code fingerprints (sha256 over notebooks/*.py, file name and content, in file-name order).
# ANALYSIS_CODE_EXCLUDE: the scripts that write no result to RESULTS_DIR (the figures, the tables, and the number
# audit, which also holds the printed values of the manuscript); the analysis-code fingerprint therefore changes
# only with the code that produces the results, and every results file records it (reproducibility_header).
NOTEBOOKS_DIR = Path(__file__).resolve().parent
ANALYSIS_CODE_EXCLUDE = ("generate_figures.py", "generate_tables.py", "manuscript_number_audit.py")
AUDIT_CODE_EXCLUDE = ("generate_figures.py",)   # fingerprint written by manuscript_number_audit.py into its CSV


def code_sha256(exclude: tuple[str, ...]) -> str:
    """sha256 over the notebooks/*.py files except those named in `exclude`, in file-name order (each file
    contributes its name, a NUL byte, its content and a NUL byte)."""
    h = hashlib.sha256()
    for p in sorted(NOTEBOOKS_DIR.glob("*.py")):
        if p.name not in exclude:
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def reproducibility_header(script_name: str) -> str:
    """Generate standard reproducibility header for results files. The reference counts are constants of the
    merged dataset file (DATASET_VERSION), printed as such on any input (e.g. the deposited file); the counts of
    the file actually read are in descriptives.txt. The analysis-code line fingerprints the scripts that produce
    the results (code_sha256(ANALYSIS_CODE_EXCLUDE))."""
    return (
        f"# {script_name}\n"
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}, "
        f"statsmodels {_pkg_version('statsmodels')}, scipy {_pkg_version('scipy')}, "
        f"scikit-learn {_pkg_version('scikit-learn')}, pymannkendall {_pkg_version('pymannkendall')}\n"
        f"# Dataset: {DATA_PATH.name} (version {DATASET_VERSION})\n"
        f"# Seed: {SEED}\n"
        f"# Reference counts of the merged dataset file (ironman_merged.csv, v{DATASET_VERSION}, whole file; constants, "
        f"not counted from the file read): N = {TOTAL_RECORDS:,}, DNF = {N_DNF:,}\n"
        f"# Analysis code: sha256 {code_sha256(ANALYSIS_CODE_EXCLUDE)} (notebooks/*.py except "
        f"{', '.join(ANALYSIS_CODE_EXCLUDE)})\n"
        f"# Analysed population: {len(DUPLICATE_RACE_YEARS)} duplicated race-years, {len(NONSTANDARD_FORMAT_EVENTS)} "
        f"non-standard-format editions, {len(NON_IRONMAN_EVENTS)} non-IRONMAN races and {len(STOPPED_EDITIONS)} stopped or "
        f"wholesale-coded editions excluded at load (_common.load_dataset; descriptives.txt, Excluded records)\n"
    )


def write_csv(df: pd.DataFrame, name: str, title: str, float_format: str | None = None) -> Path:
    """Persist a results table as CSV under RESULTS_DIR with the reproducibility
    header as '#' comment lines (read back with pd.read_csv(..., comment='#'), as every
    reader in this pipeline does). The analysis scripts write all their results CSVs here;
    manuscript_number_audit.py writes its own CSV with the same header.
    float_format is passed to DataFrame.to_csv (None keeps the full float repr).
    generate_tables.py and manuscript_number_audit.py read only results files (these
    CSVs and the txt outputs) — no number is typed by hand there."""
    path = RESULTS_DIR / name
    header = reproducibility_header(title).rstrip("\n") + "\n"
    path.write_text(header + df.to_csv(index=False, float_format=float_format, lineterminator="\n"), encoding="utf-8",
                    newline="\n")
    print(f"Saved to {path}")
    return path


# === Rounding for printed values ===
def fmt_half_up(x, nd: int, thousands: bool = False) -> str:
    """x rounded half up to nd decimals, as text. The value is taken from its shortest repr, i.e.
    the full-precision number persisted upstream (CSV at %.10g or full repr), never from a value
    already rounded to the printed precision. Missing values print as an em dash."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return "—"
    d = x if isinstance(x, Decimal) else Decimal(repr(float(x)))
    q = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)   # no "-0.000"
    return f"{q:,.{nd}f}" if thousands else f"{q:.{nd}f}"


def pct_half_up(num, den, nd: int = 1) -> str:
    """100 * num / den from integer counts, computed exactly and rounded half up to nd decimals."""
    return fmt_half_up(Decimal(int(num)) * 100 / Decimal(int(den)), nd)


def infer_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Infer sex from the age_group field (convention of the companion data descriptor)."""
    if "age_group" not in df.columns:
        raise ValueError("age_group column required for gender inference")
    ag = df["age_group"].astype(str)
    df["gender"] = np.where(
        ag.str.startswith("F") | (ag == "FPRO"), "Female",
        np.where(ag.str.startswith("M") | (ag == "MPRO"), "Male", "Unknown")
    )
    df["gender"] = df["gender"].astype("category")
    return df


def infer_last_completed_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Identify the last race segment with a recorded split time (Method 2).

    IMPORTANT — Nomenclature:

    This method identifies the LAST COMPLETED SEGMENT, not the dropout
    segment. The distinction is critical:

    - If an athlete has bike_sec > 0 but t2_sec == 0, we know the athlete
      recorded a bike split. We do NOT know whether:
      (a) they completed the bike and their race ended during T2,
      (b) their race ended during the bike (bike_sec is a partial time), or
      (c) they completed T2 but the system didn't record the T2 split.

    - Therefore, the output column is named "last_completed_segment"
      (not "dropout_segment") to avoid implying where the race
      actually ended.

    - The last completed segment is inferred from the last available
      split time greater than zero; the race may have ended during the
      identified segment or during the transition to the next segment.

    Finish-time DNFs (DNFs with a recorded overall finish time,
    overall_sec > 0) are classified separately as "finish_time": the
    source records a complete finish time for them while flagging them
    DNF, so no segment is inferred. In-race DNFs (no recorded finish
    time) receive the segment labels below.

    Zero-valued splits are treated as missing because the official API
    fills split fields with 0 (not null) for DNF athletes. Among
    official-source in-race DNFs, most have swim_sec > 0 and about half
    bike_sec > 0, but almost none run_sec > 0 (shares in dropout_point.txt
    section 1): zeros in later segments are placeholders, not times.
    """
    if "overall_sec" in df.columns:
        has_finish = df["overall_sec"].notna() & (df["overall_sec"] > 0)
    else:
        has_finish = pd.Series(False, index=df.index)

    def _split_present(col: str) -> pd.Series:
        return df[col].notna() & (df[col] > 0) if col in df.columns else pd.Series(False, index=df.index)

    conditions = [
        has_finish,
        (~has_finish) & _split_present("run_sec"),
        (~has_finish) & _split_present("t2_sec"),
        (~has_finish) & _split_present("bike_sec"),
        (~has_finish) & _split_present("t1_sec"),
        (~has_finish) & _split_present("swim_sec"),
    ]
    choices = ["finish_time", "run", "T2", "bike", "T1", "swim"]
    df["last_completed_segment"] = np.select(conditions, choices, default="unknown")
    df["last_completed_segment"] = df["last_completed_segment"].astype("category")
    return df



# === Continental mapping ===
# Built from the distinct values of the country field of ironman_merged.csv (the number of distinct
# non-empty strings among starters is persisted in country_mapping.txt).
# Strategy: map country text field → continent for ALL records (official + CoachCox).
# The country_iso2 field is only populated for official-source records;
# the country text field is non-empty for 99.9% of the starters and 99.6% of all analysed records
# (any finish status), both computed in country_mapping.txt.
#
# Mapping procedure (for reproducibility):
# 1. Extracted all unique country values with value_counts()
# 2. Mapped 369 name variants by hand (see country_mapping.txt for the strings covered; 98.0% of starters
#    mapped to a continent; country_mapping.txt)
# 3. Variants (e.g., Deutschland/Germany, Schweiz/Switzerland) mapped to same continent
# 4. Ambiguous short codes (fewer than three characters, e.g. D, F, E, I, B, A, AN) left
#    unmapped: assigning them would rest on unverifiable assumptions (this includes
#    "us" and "US", 3 starters in total)
# 5. "Unknown", empty, and garbage entries → "Unknown"
# 6. Remaining rare entries (<100 records each) mapped where identifiable

CONTINENT_MAP = {
    # North America
    "United States": "North America", "UNITED STATES": "North America",
    "U.S.": "North America",
    "USA (United States)": "North America", "Vereinigte Staaten": "North America",
    "Canada": "North America", "Kanada": "North America",
    "Mexico": "North America", "Puerto Rico": "North America",
    "Costa Rica": "North America", "Guatemala": "North America",
    "Panama": "North America", "Honduras": "North America",
    "El Salvador": "North America", "Nicaragua": "North America",
    "Dominican Republic": "North America", "Dominica": "North America",
    "Jamaica": "North America", "Belize": "North America",
    "Barbados": "North America", "Bahamas": "North America",
    "Trinidad and Tobago": "North America", "Haiti": "North America",
    "Cuba": "North America", "Bermuda": "North America",
    "Cayman Islands": "North America", "CAYMAN ISLANDS": "North America",
    "Aruba": "North America", "Curaçao": "North America", "Curacao": "North America",
    "U.S. Virgin Islands": "North America", "United States Virgin Islands": "North America",
    "British Virgin Islands": "North America", "Virgin Islands": "North America",
    "Guam": "North America", "American Samoa": "North America",
    "Northern Mariana Islands": "North America", "Saipan": "North America",
    "Grenada": "North America", "Saint Kitts & Nevis": "North America",
    "Saint Kitts and Nevis": "North America", "Saint Lucia": "North America",
    "Antigua and Barbuda": "North America", "Anguilla": "North America",
    "Montserrat": "North America", "Turks and Caicos": "North America",
    "Turks and Caicos Islands": "North America",
    "Sint Maarten": "North America", "Saint Martin": "North America",
    "Saint Barthelemy": "North America", "Saint Barthlemy": "North America",
    "Saint Barthélemy": "North America",
    "Caribbean Netherlands": "North America",
    "Bonaire, Sint Eustatius and Saba": "North America",
    "Netherlands Antilles": "North America", "Guadeloupe": "North America",
    "GUADELOUPE": "North America", "Gaudeloupe": "North America",
    "Martinique": "North America", "St. Vincent & Grenadines": "North America",
    "Saint Vincent and the Grenadines": "North America",
    "United States Minor Outlying Islands": "North America",
    "St Lucia": "North America",

    # South America
    "Brazil": "South America", "Brasilien": "South America",
    "Argentina": "South America", "Argentinia": "South America",
    "Chile": "South America", "Colombia": "South America",
    "Ecuador": "South America", "Peru": "South America",
    "Venezuela": "South America", "Uruguay": "South America",
    "Paraguay": "South America", "Bolivia": "South America",
    "Suriname": "South America", "Guyana": "South America",
    "French Guiana": "South America",

    # Europe
    "United Kingdom": "Europe", "U.K.": "Europe",
    "GBR (Great Britain)": "Europe", "Northern Ireland": "Europe",
    "Germany": "Europe", "Deutschland": "Europe",
    "GER (Germany)": "Europe", "Germ": "Europe",
    "France": "Europe", "Frankreich": "Europe",
    "FRA (France)": "Europe", "Frances": "Europe",
    "Italy": "Europe", "Italien": "Europe",
    "Spain": "Europe", "Spanien": "Europe",
    "Sweden": "Europe", "Schweden": "Europe", "SWE (Sweden)": "Europe",
    "Switzerland": "Europe", "Schweiz": "Europe",
    "Denmark": "Europe", "DEN (Denmark)": "Europe",
    "Belgium": "Europe", "Belgien": "Europe", "BEL (Belgium)": "Europe",
    "Poland": "Europe", "Polen": "Europe",
    "Netherlands": "Europe", "Niederlande": "Europe", "Netherland": "Europe",
    "The Netherlands": "Europe", "NED (Netherlands)": "Europe",
    "Austria": "Europe", "AUT (Austria)": "Europe",
    "Ireland": "Europe", "Ireland (Republic of)": "Europe",
    "Ireland  Republic of": "Europe", "Irland": "Europe",
    "Norway": "Europe", "Norwegen": "Europe",
    "Finland": "Europe", "Finnland": "Europe",
    "Portugal": "Europe",
    "Estonia": "Europe", "Latvia": "Europe", "Lithuania": "Europe",
    "Czech Republic": "Europe", "Czechia": "Europe",
    "Tschechische Republik": "Europe", "The Czech Republic": "Europe",
    "Slovakia": "Europe", "Slovak republic": "Europe", "Slovak Republic": "Europe",
    "Slowakei": "Europe",
    "Hungary": "Europe",
    "Romania": "Europe", "Bulgaria": "Europe",
    "Croatia": "Europe", "Slovenia": "Europe",
    "Serbia": "Europe", "Serbia and Montenegro": "Europe",
    "Montenegro": "Europe", "Bosnia and Herzegovina": "Europe",
    "Albania": "Europe", "North Macedonia": "Europe",
    "Republic of North Macedonia": "Europe", "Macedonia": "Europe",
    "Greece": "Europe", "Griechenland": "Europe",
    "Iceland": "Europe", "ICE": "Europe",
    "Malta": "Europe", "MALTA": "Europe",
    "Cyprus": "Europe", "CYPRUS": "Europe",
    "Luxembourg": "Europe", "Luxemburg": "Europe",
    "Monaco": "Europe", "Andorra": "Europe",
    "Liechtenstein": "Europe", "San Marino": "Europe",
    "Gibraltar": "Europe", "Faroe Islands": "Europe",
    "Isle of Man": "Europe", "Isle Of Man": "Europe",
    "Jersey": "Europe", "Guernsey": "Europe",
    "Aland Islands": "Europe", "Åland Islands": "Europe",
    "Greenland": "Europe", "GREENLAND": "Europe",
    "Svalbard and Jan Mayen": "Europe",
    "Russia": "Europe", "Russian Federation": "Europe",
    "RUS (Russia)": "Europe", "Russian": "Europe",
    "Ukraine": "Europe", "Belarus": "Europe", "Moldova": "Europe",
    "Georgia": "Europe", "Armenia": "Europe", "Azerbaijan": "Europe",
    "Kosovo": "Europe", "Holy See": "Europe",
    "Holy See (Vatican City State)": "Europe",
    "Falkland Islands": "Europe", "Falkland Islands (Malvinas)": "Europe",
    "NIR": "Europe", "ENG": "Europe", "SCO": "Europe",

    # Asia
    "Japan": "Asia", "Jpana": "Asia", "Japn": "Asia",
    "South Korea": "Asia", "Korea": "Asia", "korea": "Asia",
    "South KOREA": "Asia", "KOREA": "Asia", "South korea": "Asia",
    "Korea republic of": "Asia", "korea republic of": "Asia",
    "Republic of (South) Korea": "Asia", "The Republic of KOREA": "Asia",
    "The Repulic of KOREA": "Asia", "Rebublic of Korea": "Asia",
    "China": "Asia", "Hong Kong": "Asia", "Hongkong": "Asia",
    "Taiwan": "Asia", "R.O.C Taiwan": "Asia", "R.O.C": "Asia",
    "Taiwan (R.O.C.)": "Asia", "Taiwan, Province of China": "Asia",
    "India": "Asia", "Philippines": "Asia", "Philippinen": "Asia",
    "Singapore": "Asia", "Singapur": "Asia",
    "Vietnam": "Asia", "Viet Nam": "Asia",
    "Malaysia": "Asia", "Thailand": "Asia",
    "Indonesia": "Asia", "Cambodia": "Asia",
    "Bahrain": "Asia", "BAHRAIN": "Asia",
    "United Arab Emirates": "Asia", "Vereinigte Arabische Emirate": "Asia",
    "Israel": "Asia", "Turkey": "Asia",
    "Saudi Arabia": "Asia", "Qatar": "Asia",
    "Kuwait": "Asia", "Oman": "Asia",
    "Lebanon": "Asia", "LEB": "Asia",
    "Jordan": "Asia", "Iraq": "Asia", "Iran": "Asia", "IRAN": "Asia",
    "Pakistan": "Asia", "Bangladesh": "Asia",
    "Sri Lanka": "Asia", "Nepal": "Asia", "NEP": "Asia",
    "Macao": "Asia", "Macau": "Asia",
    "Kazakhstan": "Asia", "Kyrgyzstan": "Asia",
    "Uzbekistan": "Asia", "Tajikistan": "Asia", "Turkmenistan": "Asia",
    "Mongolia": "Asia", "Myanmar": "Asia", "MYA": "Asia",
    "Brunei": "Asia", "Brunei Darussalam": "Asia", "BRU": "Asia",
    "Laos": "Asia", "Lao People's Democratic Republic": "Asia",
    "Maldives": "Asia", "MALDIVES": "Asia",
    "Afghanistan": "Asia", "Bhutan": "Asia",
    "North Korea": "Asia", "Syria": "Asia",
    "Syrian Arab Republic": "Asia", "Yemen": "Asia",
    "Palestine": "Asia", "State of Palestine": "Asia",
    "Palestinian Territory": "Asia", "Palestinian Territory, Occupied": "Asia",
    "Timor-Leste": "Asia",

    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Fiji": "Oceania", "Papua New Guinea": "Oceania",
    "Samoa": "Oceania", "Tonga": "Oceania",
    "Vanuatu": "Oceania", "Solomon Islands": "Oceania",
    "New Caledonia": "Oceania", "NewCaledonia": "Oceania",
    "French Polynesia": "Oceania", "Tahiti": "Oceania",
    "Cook Islands": "Oceania", "Niue": "Oceania",
    "Micronesia": "Oceania",
    "Palau": "Oceania", "Kiribati": "Oceania",
    "Marshall Islands": "Oceania", "Nauru": "Oceania",
    "Tuvalu": "Oceania", "Norfolk Island": "Oceania",
    "Christmas Island": "Oceania", "Cocos Islands": "Oceania",
    "Cocos (Keeling) Islands": "Oceania",
    "Pitcairn": "Oceania", "Tokelau": "Oceania",

    # Africa
    "South Africa": "Africa", "SAF": "Africa",
    "Morocco": "Africa", "MOR": "Africa",
    "Egypt": "Africa", "Tunisia": "Africa",
    "Algeria": "Africa", "Libya": "Africa",
    "Libyan Arab Jamahiriya": "Africa",
    "Kenya": "Africa", "Nigeria": "Africa",
    "Ghana": "Africa", "Cameroon": "Africa",
    "Senegal": "Africa", "Côte d'Ivoire": "Africa",
    "Cote D'Ivoire": "Africa",
    "Ethiopia": "Africa", "Tanzania": "Africa",
    "Tanzania, United Republic of": "Africa",
    "Uganda": "Africa", "Rwanda": "Africa",
    "Zimbabwe": "Africa", "Zambia": "Africa",
    "Mozambique": "Africa", "Namibia": "Africa",
    "Botswana": "Africa", "Eswatini": "Africa", "Swaziland": "Africa",
    "Mauritius": "Africa", "Madagascar": "Africa",
    "Seychelles": "Africa", "Réunion": "Africa",
    "Reunion": "Africa", "RÃ©union": "Africa",
    "Mayotte": "Africa", "Djibouti": "Africa",
    "Eritrea": "Africa", "Somalia": "Africa",
    "Sudan": "Africa", "South Sudan": "Africa",
    "Angola": "Africa", "DR Congo": "Africa",
    "Democratic Republic of the Congo": "Africa", "Congo": "Africa",
    "Gabon": "Africa", "Equatorial Guinea": "Africa",
    "Central African Republic": "Africa",
    "Sierra Leone": "Africa", "Liberia": "Africa",
    "Mali": "Africa", "Niger": "Africa",
    "Burkina Faso": "Africa", "Togo": "Africa",
    "Benin": "Africa", "Guinea": "Africa",
    "Guinea-Bissau": "Africa", "Gambia": "Africa", "GAM": "Africa",
    "Cabo Verde": "Africa", "Cape Verde": "Africa",
    "Comoros": "Africa", "Lesotho": "Africa",
    "Malawi": "Africa", "Burundi": "Africa",
    "Western Sahara": "Africa", "Chad": "Africa",
    "Mauritania": "Africa", "Sao Tome & Principe": "Africa",
    "Saint Helena": "Africa",
    "Saint Helena, Ascension and Tristan da Cunha": "Africa",
    "TAN": "Africa",
}


def assign_continent(df: pd.DataFrame) -> pd.DataFrame:
    """Assign continent from the country text field.

    Uses country (text) as primary source because it is non-empty for 99.9%
    of the starters and 99.6% of all analysed records, both sources pooled
    (official and CoachCox; country_mapping.txt). The country_iso2 field is
    only populated for official-source records.

    Mapping covers 369 country-name variants (98.0% of starters mapped; country_mapping.txt).
    Unmapped entries (rare variants, garbage, single-letter codes not identified,
    "Unknown") are assigned "Unknown".
    """
    df["continent"] = df["country"].astype(str).str.strip().map(CONTINENT_MAP).fillna("Unknown")
    df["continent"] = df["continent"].astype("category")
    return df


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df):,} records (excluded race-years removed)")
    print(f"Finish status:\n{df['finish_status'].value_counts()}")
