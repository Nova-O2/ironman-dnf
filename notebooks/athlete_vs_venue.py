"""
Exploratory: Athlete origin × Race location cross-tabulation.
Separates demographic effect (who the athlete is) from environmental effect
(where the race takes place).
Produces: results/athlete_vs_venue.txt, athlete_vs_venue_crosstab.csv (Table S1),
          athlete_vs_venue_home_away.csv (Table S2), athlete_vs_venue_share_by_type.csv (share of each athlete
          continent's starts held on each race continent, by event type), country_mapping.txt
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    CONTINENT_MAP, load_dataset, assign_continent,
    reproducibility_header, write_csv, RESULTS_DIR,
)
from _race_locations import (
    assign_race_continent, build_race_location_table, load_cc_metadata, load_subevents,
    region_field_continent_map, series_name, SLUG_TO_COUNTRY, CC_SERIES_TO_COUNTRY,
)


def persist_country_mapping_stats(df: pd.DataFrame | None = None, records_country: pd.Series | None = None):
    """Persist counts describing the CONTINENT_MAP coverage.

    Claims supported (Methods §Demographic and Geographic Variables): number of
    country-name variants in the mapping; share of starters with a non-empty
    country field (df: the starters), and the same share among all analysed records
    (records_country: the country field of every record, whatever its finish status);
    share of starters whose country value is an ambiguous short code (< 3 characters
    and not covered by the mapping, so left unmapped by design; the lookup is the one
    assign_continent applies: stripped string, exact case); number of distinct country
    strings in the data; share of starters mapped; number of event series (official
    and secondary source) mapped by hand to their host country.

    Output: results/country_mapping.txt
    """
    n_variants = len(CONTINENT_MAP)
    by_continent = {}
    for variant, continent in CONTINENT_MAP.items():
        by_continent[continent] = by_continent.get(continent, 0) + 1
    continents = sorted(by_continent.keys())

    out = []
    out.append(reproducibility_header("Country → Continent Mapping — Variant Count"))
    out.append("Source: _common.CONTINENT_MAP (single source of truth for")
    out.append("athlete-origin continental assignment across all analyses).")
    out.append("")
    out.append("=" * 60)
    out.append("VARIANT COUNT")
    out.append("=" * 60)
    out.append(f"\nTotal country-name variants in mapping: {n_variants}")
    out.append(f"Number of distinct continents covered:  {len(continents)}")
    out.append("\nVariants per continent:")
    for cont in continents:
        out.append(f"  {cont:<18} {by_continent[cont]:>4}")

    if df is not None:
        out.append("\n" + "=" * 60)
        out.append("COUNTRY FIELD IN THE DATA (starters: FIN + DNF + DQ)")
        out.append("=" * 60)
        country = df["country"].astype(str).str.strip()
        nonempty = country.notna() & (country != "") & (country.str.lower() != "nan")
        mapped = country.map(CONTINENT_MAP).notna()
        short_all = nonempty & (country.str.len() < 3)
        short = short_all & ~mapped           # short strings the mapping leaves unmapped
        short_mapped = country[short_all & mapped].value_counts()
        out.append(f"\nStarters: {len(df):,}")
        out.append(f"Non-empty country field: {int(nonempty.sum()):,} ({nonempty.mean()*100:.4f}%)")
        if records_country is not None:
            rc = records_country.astype(str).str.strip()
            rc_nonempty = rc.notna() & (rc != "") & (rc.str.lower() != "nan")
            out.append(f"  among all analysed records (any finish status, DNS and records without a status included): "
                       f"{int(rc_nonempty.sum()):,} / {len(rc):,} ({rc_nonempty.mean()*100:.4f}%)")
        out.append(f"Ambiguous short codes (< 3 characters, excluded from the mapping): "
                   f"{int(short.sum()):,} ({short.mean()*100:.4f}%)")
        out.append("Short strings covered by the mapping (not counted above): "
                   + (", ".join(f"'{k}' ({v:,})" for k, v in short_mapped.items()) or "none"))
        out.append(f"Distinct non-empty country strings: {country[nonempty].nunique():,}")
        out.append(f"Distinct strings covered by the mapping: {country[mapped].nunique():,}")
        out.append(f"Starters mapped to a continent: {int(mapped.sum()):,} / {len(df):,} ({mapped.mean()*100:.4f}%)")

    out.append("\n" + "=" * 60)
    out.append("RACE LOCATION — EVENT SERIES MAPPED TO THEIR HOST COUNTRY")
    out.append("=" * 60)
    out.append("\nRule: race -> host country -> continent by CONTINENT_MAP (the athlete-origin mapping).")
    n_slugs = load_subevents()["parent_slug"].dropna().nunique()
    table = build_race_location_table()
    cc = table[table["source"] == "coachcox"]
    out.append(f"Event series (parent slugs) in the official index: {n_slugs}")
    out.append(f"Official series mapped by hand to their host country (SLUG_TO_COUNTRY): {len(SLUG_TO_COUNTRY)}")
    out.append("Official World Championship series (2): host by year, from the event name")
    n_cc_series = load_cc_metadata()["race_name"].map(series_name).nunique()
    out.append(f"Secondary-source series in race_metadata.csv: {n_cc_series}; mapped by hand to their "
               f"host country (CC_SERIES_TO_COUNTRY): {len(CC_SERIES_TO_COUNTRY)}; World Championship series "
               f"mapped by year: {cc.loc[cc['is_world_championship'], 'series'].nunique()}")
    out.append("  (one secondary-source series name, 'Ironman 70.3 San Juan', covers two venues: its races up to 2016")
    out.append("   are mapped as 'Ironman 70.3 Puerto Rico')")
    out.append(f"Host countries: {table['country'].nunique()}")

    output_text = "\n".join(out)
    (RESULTS_DIR / "country_mapping.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(f"Saved to {RESULTS_DIR / 'country_mapping.txt'}")
    return n_variants


def main():
    cols = ["finish_status", "race_type", "country", "event_id", "event_name", "source"]
    print("Loading dataset...")
    df = load_dataset(cols)
    records_country = df["country"]   # every analysed record, whatever its finish status (country_mapping.txt)
    df = df[df["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()
    df["is_dnf"] = (df["finish_status"] == "DNF").astype(int)

    # Map athlete origin (country text → continent)
    df = assign_continent(df)
    df = df.rename(columns={"continent": "athlete_continent"})

    # Map race location (host country of each race from structured metadata: official series
    # slug, secondary-source series, World Championship host by year -> CONTINENT_MAP)
    df = assign_race_continent(df)
    df = df.rename(columns={"race_continent": "event_continent"})

    out = []
    out.append(reproducibility_header("Athlete Origin × Race Location"))

    # --- Coverage ---
    out.append("=" * 60)
    out.append("SECTION 1: MAPPING COVERAGE")
    out.append("=" * 60)

    athlete_known = (df["athlete_continent"] != "Unknown").sum()
    event_known = (df["event_continent"] != "Unknown").sum()
    out.append(f"\nAthlete origin coverage: {athlete_known:,} / {len(df):,} = {athlete_known/len(df)*100:.4f}%")
    out.append(f"Race location coverage: {event_known:,} / {len(df):,} = {event_known/len(df)*100:.4f}%")
    out.append("World Championship events are mapped to their host continent for each year.")
    out.append(f"\nNote: athlete origin from 'country' field (registration nationality).")
    out.append("Race location from structured event metadata (_race_locations.py): host country of each race,")
    out.append("assigned to a continent by the athlete-origin mapping (CONTINENT_MAP).")

    # Correction of the secondary-source race location: the continent of the source's region field against the
    # host-country rule, among starters of secondary-source races (the region field is read only for this count).
    cc = df[df["source"].astype(str) == "coachcox"]
    region_cont = cc["event_id"].astype(str).map(region_field_continent_map())
    moved = cc[region_cont != cc["event_continent"].astype(str)]
    out.append(f"\nSecondary-source starters whose race continent under the host-country rule differs from the")
    out.append(f"continent of the source's region field: {len(moved):,} of {len(cc):,} (DNF {int(moved['is_dnf'].sum()):,})")
    if len(moved):
        g = moved.assign(region_continent=region_cont[moved.index]).groupby(
            ["region_continent", "event_continent"], observed=True)["is_dnf"].agg(starters="size", DNF="sum")
        out.append("  region field -> host-country rule:")
        out.append("  " + g.to_string().replace("\n", "\n  "))
    unk = df[df["event_continent"] == "Unknown"]
    out.append(f"Starters without a race location: {len(unk):,}")
    if len(unk):
        out.append("  top unmapped events:")
        for name, n in unk["event_name"].astype(str).value_counts().head(10).items():
            out.append(f"    {n:>7,}  {name}")

    # --- DNF rate by event continent ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 2: DNF RATE BY RACE LOCATION")
    out.append("=" * 60)

    event_mapped = df[df["event_continent"] != "Unknown"]
    g = event_mapped.groupby(["event_continent", "race_type"], observed=True)["is_dnf"].agg(N_starts="count", N_DNF="sum", DNF_rate="mean")
    g["DNF_rate"] = (g["DNF_rate"] * 100).round(4)
    out.append("\n" + g.sort_values("N_starts", ascending=False).to_string())

    # --- Cross-tabulation ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 3: CROSS-TABULATION (ATHLETE ORIGIN × RACE LOCATION)")
    out.append("=" * 60)
    out.append("\nDNF rate (%) — rows = athlete origin, cols = race location.")
    out.append("Only combinations with N >= 1,000 starts shown.")

    both = df[
        (df["athlete_continent"] != "Unknown") &
        (df["event_continent"] != "Unknown")
    ]
    out.append(f"Records with both mapped: {len(both):,}")

    ct = both.groupby(["athlete_continent", "event_continent"], observed=True).agg(
        N=("is_dnf", "count"),
        DNF=("is_dnf", "sum"),
    ).reset_index()
    ct["DNF_rate"] = (ct["DNF"] / ct["N"] * 100).round(4)
    write_csv(ct.rename(columns={"N": "N_starts", "DNF": "N_DNF", "DNF_rate": "DNF_rate_pct"}),
              "athlete_vs_venue_crosstab.csv",
              "Athlete origin × race location — all combinations (Table S1 reads N >= 1,000)")

    # Pivot — DNF rate
    sig = ct[ct["N"] >= 1000]
    pivot_rate = sig.pivot_table(index="athlete_continent", columns="event_continent", values="DNF_rate")
    out.append(f"\nDNF rate (%):")
    out.append(pivot_rate.round(1).to_string())

    # Pivot — N
    pivot_n = sig.pivot_table(index="athlete_continent", columns="event_continent", values="N")
    out.append(f"\nN starts:")
    out.append(pivot_n.to_string())

    # --- Home vs away ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 4: HOME vs AWAY EFFECT")
    out.append("=" * 60)
    out.append("\n'Home' = athlete continent matches race continent.")

    both_mapped = both.copy()
    both_mapped["is_home"] = both_mapped["athlete_continent"].astype(str) == both_mapped["event_continent"].astype(str)

    home_away = both_mapped.groupby(["athlete_continent", "is_home"], observed=True)["is_dnf"].agg(N_starts="count", N_DNF="sum", DNF_rate="mean")
    home_away["DNF_rate"] = (home_away["DNF_rate"] * 100).round(4)
    out.append("\n" + home_away.to_string())

    # Overall home vs away
    ha_overall = both_mapped.groupby("is_home")["is_dnf"].agg(N_starts="count", N_DNF="sum", DNF_rate="mean")
    ha_overall["DNF_rate"] = (ha_overall["DNF_rate"] * 100).round(4)
    ha_overall.index = ["Away", "Home"]
    out.append(f"\nOverall:")
    out.append(ha_overall.to_string())
    ha_rows = home_away.reset_index().rename(columns={"DNF_rate": "DNF_rate_pct"})
    ha_rows["is_home"] = ha_rows["is_home"].map({True: "Home", False: "Away"})
    ha_all = ha_overall.reset_index().rename(columns={"index": "is_home", "DNF_rate": "DNF_rate_pct"})
    ha_all.insert(0, "athlete_continent", "Overall")
    write_csv(pd.concat([ha_rows, ha_all], ignore_index=True), "athlete_vs_venue_home_away.csv",
              "Home vs away DNF rate by athlete continent (Table S2)")

    # --- Where the athletes of each continent race, by event type ---
    out.append("\n" + "=" * 60)
    out.append("SECTION 5: RACE LOCATION OF EACH ATHLETE CONTINENT'S STARTS, BY EVENT TYPE")
    out.append("=" * 60)
    out.append("\nShare (%) of the starts of athletes from each continent (rows) held on each race continent")
    out.append("(columns); each row sums to 100.")
    share = both.groupby(["race_type", "athlete_continent", "event_continent"], observed=True).size().rename("N_starts").reset_index()
    share["share_pct"] = 100 * share["N_starts"] / share.groupby(["race_type", "athlete_continent"], observed=True)["N_starts"].transform("sum")
    for rt, label in (("im", "IRONMAN"), ("him", "IRONMAN 70.3")):
        piv = share[share["race_type"] == rt].pivot_table(index="athlete_continent", columns="event_continent",
                                                          values="share_pct", fill_value=0.0, observed=True)
        out.append(f"\n{label}:")
        out.append(piv.round(2).to_string())
    write_csv(share, "athlete_vs_venue_share_by_type.csv",
              "Race location of each athlete continent's starts, by event type (share of the continent's starts, %)")

    # --- Save ---
    output_text = "\n".join(out)
    (RESULTS_DIR / "athlete_vs_venue.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(f"Saved to {RESULTS_DIR / 'athlete_vs_venue.txt'}")
    print(output_text)
    return df, records_country


if __name__ == "__main__":
    df, records_country = main()
    print("\n" + "#" * 60)
    print("# Persisting country-variant mapping count ...")
    print("#" * 60 + "\n")
    persist_country_mapping_stats(df, records_country)
