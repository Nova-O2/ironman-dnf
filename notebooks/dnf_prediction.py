"""
RQ4: Logistic regression for DNF risk factor identification.
Two separate models: IM and HIM (never pooled).
Outcome: in-race DNF (DNF without a recorded finish time) vs finisher or DQ.
Note: AUC is reported for context but ORs are the primary output.

Population: starters (FIN + DNF + DQ), excluding
- professionals (MPRO, FPRO; analysed separately in pro_rank_analysis.py)
- finish-time DNFs (DNFs with a recorded overall finish time, overall_sec > 0)
- unknown sex (count in descriptives.txt section 2) and age-group codes
  outside the age bins
- unknown continent of origin and unmapped race location

Variables: sex, age bin (5-year, 70+ pooled), athlete continent, race continent, race_year

Standard errors are clustered by event (event_id, one cluster per race-year): athletes of the
same race share its conditions, so the naive (independent-observations) standard errors are too
small. The odds ratios are unchanged; CIs and p-values come from the clustered covariance.
ORs, CIs and p are persisted at full precision (%.10g) in the CSVs.

Sensitivities (sensitivity_analyses.py):
- S1: finish-time DNFs included in the outcome
- S2: combined IM + HIM model with event type, 20% simple random sample
- S3: excluding the 2020-2021 seasons
- S4: including professionals with an is_pro indicator

Produces: results/prediction_im.txt, prediction_him.txt, prediction_im_or.csv (Table 2),
          prediction_him_or.csv (Table 3), prediction_summary.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_dataset, infer_gender, assign_continent, write_csv,
    reproducibility_header, RESULTS_DIR,
)

CSV_FLOAT = "%.10g"
from _race_locations import assign_race_continent


def prepare_data(df: pd.DataFrame, race_type: str) -> pd.DataFrame:
    """Prepare data for logistic regression.

    Filters to starters (FIN + DNF + DQ), excludes PROs, finish-time DNFs,
    unknown sex, unknown continent of origin and unmapped race location.
    Adds derived variables.
    """
    # Starters only
    sub = df[df["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()

    # Filter distance
    sub = sub[sub["race_type"] == race_type]

    # Exclude PROs
    sub = sub[~sub["age_group"].astype(str).isin(["MPRO", "FPRO"])]

    # Exclude finish-time DNFs (overall_sec > 0 among DNFs): the outcome is the in-race DNF
    is_finish_time = (sub["finish_status"] == "DNF") & sub["overall_sec"].notna() & (sub["overall_sec"] > 0)
    sub = sub[~is_finish_time]

    # Infer gender, exclude Unknown
    sub = infer_gender(sub)
    sub = sub[sub["gender"].isin(["Male", "Female"])]

    # Outcome
    sub["dnf"] = (sub["finish_status"] == "DNF").astype(int)

    # Age bins (5-year, from age_group)
    # Consolidate 70+ into single bin (N too small for stable estimation)
    from descriptives import map_age_bin
    sub["age_bin"] = sub["age_group"].astype(str).apply(map_age_bin)
    sub = sub[~sub["age_bin"].isin(["Other", "PRO"])]

    # Merge all 70+ bins
    sub["age_bin"] = sub["age_bin"].apply(
        lambda x: "70+" if x in ("70-74", "75-79", "80-84", "80+", "85-89", "85+", "90+", "70+") else x
    )

    # Continent (athlete origin)
    sub = assign_continent(sub)
    sub = sub.rename(columns={"continent": "athlete_continent"})
    sub = sub[sub["athlete_continent"] != "Unknown"]

    # Race continent
    sub = assign_race_continent(sub)
    sub = sub[sub["race_continent"] != "Unknown"]

    # is_home removed from model — confounded by compositional differences
    # (North Americans dominate "home", Europeans dominate "away").
    # Reported descriptively in supplementary tables (athlete × venue analysis).

    return sub


def run_logistic(df: pd.DataFrame, race_type: str) -> tuple[list[str], pd.DataFrame, dict]:
    """Run logistic regression; return output lines, the odds-ratio table and a summary."""
    import statsmodels.api as sm
    from statsmodels.tools import add_constant

    out = []
    out.append(reproducibility_header(f"RQ4 — Logistic Regression ({race_type.upper()})"))

    sub = prepare_data(df, race_type)

    out.append(f"Sample: {len(sub):,} starters ({race_type.upper()}, age-group only)")
    out.append(f"In-race DNFs (outcome): {sub['dnf'].sum():,} ({sub['dnf'].mean()*100:.4f}%)")
    out.append("Excluded: PROs, finish-time DNFs, unknown sex, unknown continent of origin, unmapped race location")

    # --- Dummy encoding ---
    # Reference categories: Male, 30-34, North America (athlete), North America (race)
    ref_age = "30-34"
    ref_athlete_cont = "North America"
    ref_race_cont = "North America"

    out.append(f"\nReference categories: sex=Male, age={ref_age}, "
               f"athlete_continent={ref_athlete_cont}, race_continent={ref_race_cont}")

    # Gender dummy
    sub["female"] = (sub["gender"] == "Female").astype(int)

    # Age dummies
    age_bins = sorted([b for b in sub["age_bin"].unique() if b != ref_age])
    for ab in age_bins:
        sub[f"age_{ab}"] = (sub["age_bin"] == ab).astype(int)

    # Athlete continent dummies
    athlete_conts = sorted([c for c in sub["athlete_continent"].unique() if c != ref_athlete_cont])
    for c in athlete_conts:
        col = f"ath_{c.replace(' ', '_')}"
        sub[col] = (sub["athlete_continent"] == c).astype(int)

    # Race continent dummies
    race_conts = sorted([c for c in sub["race_continent"].unique() if c != ref_race_cont])
    for c in race_conts:
        col = f"race_{c.replace(' ', '_')}"
        sub[col] = (sub["race_continent"] == c).astype(int)

    # Feature matrix
    feature_cols = (
        ["female"]
        + [f"age_{ab}" for ab in age_bins]
        + [f"ath_{c.replace(' ', '_')}" for c in athlete_conts]
        + [f"race_{c.replace(' ', '_')}" for c in race_conts]
        + ["race_year"]
    )

    X = sub[feature_cols].astype(float)
    X = add_constant(X)
    y = sub["dnf"]
    groups = pd.factorize(sub["event_id"].astype(str))[0]   # one cluster per event (race-year)
    n_clusters = int(groups.max()) + 1

    out.append(f"Features: {len(feature_cols)}")
    out.append(f"Observations: {len(X):,}")
    out.append(f"Standard errors: clustered by event (event_id), {n_clusters:,} clusters; "
               f"CIs and p-values from the clustered covariance (cov_type='cluster')")

    # --- Fit ---
    out.append("\nFitting logistic regression...")
    model = sm.Logit(y, X)
    result = model.fit(disp=0, maxiter=100, cov_type="cluster", cov_kwds={"groups": groups})

    out.append(f"Converged: {result.mle_retvals['converged']}")
    out.append(f"Log-likelihood: {result.llf:,.1f}")
    out.append(f"Pseudo R²: {result.prsquared:.6f}")

    # --- Odds Ratios ---
    out.append("\n" + "=" * 70)
    out.append(f"ODDS RATIOS (95% CI; standard errors clustered by event, {n_clusters:,} clusters)")
    out.append("=" * 70)

    or_df = pd.DataFrame({
        "OR": np.exp(result.params),
        "CI_low": np.exp(result.conf_int()[0]),
        "CI_high": np.exp(result.conf_int()[1]),
        "p": result.pvalues,
        "SE_log_odds": result.bse,
    })
    or_df["sig"] = or_df["p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    or_df["CI_includes_1"] = (or_df["CI_low"] <= 1) & (or_df["CI_high"] >= 1)
    out.append(or_df.to_string(float_format=lambda v: f"{v:.6g}"))
    out.append("(full precision in the CSV)")

    # --- Model Performance ---
    out.append("\n" + "=" * 70)
    out.append("MODEL PERFORMANCE")
    out.append("=" * 70)

    y_pred_prob = result.predict(X)
    auc = roc_auc_score(y, y_pred_prob)
    out.append(f"AUC-ROC: {auc:.6f}")

    # Threshold = prevalence
    prevalence = y.mean()
    y_pred = (y_pred_prob >= prevalence).astype(int)
    out.append(f"Threshold (prevalence): {prevalence:.4f}")
    out.append(f"\nClassification report:")
    out.append(classification_report(y, y_pred, target_names=["Finisher", "DNF"]))

    summary = {"race_type": race_type, "n": len(X), "n_dnf": int(y.sum()),
               "dnf_rate_pct": float(y.mean()) * 100, "auc": float(auc),
               "converged": bool(result.mle_retvals["converged"]), "n_features": len(feature_cols),
               "pseudo_r2": float(result.prsquared), "cov_type": "cluster (event_id)", "n_clusters": n_clusters}

    # --- Save predicted probabilities summary ---
    out.append("=" * 70)
    out.append("PREDICTED PROBABILITY DISTRIBUTION")
    out.append("=" * 70)
    for outcome, label in [(0, "Finishers"), (1, "DNFs")]:
        probs = y_pred_prob[y == outcome]
        out.append(f"\n{label} (n={len(probs):,}):")
        out.append(f"  P5={probs.quantile(.05):.4f}, P25={probs.quantile(.25):.4f}, "
                   f"P50={probs.quantile(.50):.4f}, P75={probs.quantile(.75):.4f}, "
                   f"P95={probs.quantile(.95):.4f}")

    or_out = or_df.drop(columns=["sig"]).reset_index().rename(columns={"index": "variable"})
    or_out.insert(0, "race_type", race_type)
    return out, or_out, summary


def main():
    cols = [
        "finish_status", "race_type", "age_group", "country",
        "event_id", "event_name", "source",
        "overall_sec", "race_year",
    ]
    print("Loading dataset...")
    df = load_dataset(cols)

    summaries = []
    for rt in ["im", "him"]:
        print(f"\n{'='*40}")
        print(f"Running {rt.upper()} model...")
        print(f"{'='*40}")

        out, or_out, summary = run_logistic(df, rt)
        output_text = "\n".join(out)

        output_path = RESULTS_DIR / f"prediction_{rt}.txt"
        output_path.write_text(output_text, encoding="utf-8", newline="\n")
        print(f"Saved to {output_path}")
        print(output_text)
        write_csv(or_out, f"prediction_{rt}_or.csv",
                  f"RQ4 — odds ratios ({rt.upper()} model; Table {'2' if rt == 'im' else '3'}); SEs clustered by event",
                  float_format=CSV_FLOAT)
        summaries.append(summary)
    write_csv(pd.DataFrame(summaries), "prediction_summary.csv", "RQ4 — model summaries (n, DNF, AUC, clusters)",
              float_format=CSV_FLOAT)


if __name__ == "__main__":
    main()
