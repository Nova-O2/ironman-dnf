"""
Sensitivity analyses for the logistic regression (RQ4). Four analyses:
  S1: finish-time DNFs (DNFs with a recorded finish time) included in the outcome
  S2: combined IM + HIM model with event type (is_him), 20% simple random sample
  S3: excluding the 2020-2021 seasons
  S4: including professionals with an is_pro indicator

Standard errors are clustered by event (event_id), as in dnf_prediction.py (also in the combined
model S2); ORs, CIs and p are persisted at full precision (%.10g).

Produces: results/sensitivity_s1.txt through sensitivity_s4.txt,
          sensitivity_or.csv and sensitivity_summary.csv (Tables S5-S8)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_dataset, infer_gender, assign_continent, fmt_half_up,
    reproducibility_header, write_csv, RESULTS_DIR, SEED,
)

CSV_FLOAT = "%.10g"
from _race_locations import assign_race_continent
from descriptives import map_age_bin

OR_ROWS: list[dict] = []      # every fitted coefficient, all analyses (sensitivity_or.csv)
SUMMARY_ROWS: list[dict] = [] # one row per model (sensitivity_summary.csv)


def prepare_base(df: pd.DataFrame, race_type: str | None = None,
                 include_finish_time: bool = False,
                 include_pro: bool = False,
                 exclude_years: list[int] | None = None) -> pd.DataFrame:
    """Flexible data preparation for sensitivity analyses."""
    sub = df[df["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()

    if race_type:
        sub = sub[sub["race_type"] == race_type]

    if not include_pro:
        sub = sub[~sub["age_group"].astype(str).isin(["MPRO", "FPRO"])]

    if not include_finish_time:
        is_ft = (sub["finish_status"] == "DNF") & sub["overall_sec"].notna() & (sub["overall_sec"] > 0)
        sub = sub[~is_ft]

    if exclude_years:
        sub = sub[~sub["race_year"].isin(exclude_years)]

    sub = infer_gender(sub)
    sub = sub[sub["gender"].isin(["Male", "Female"])]
    sub["dnf"] = (sub["finish_status"] == "DNF").astype(int)

    sub["age_bin"] = sub["age_group"].astype(str).apply(map_age_bin)
    sub = sub[sub["age_bin"] != "Other"]
    # If including PROs, keep them but map to a generic age bin (they don't have age ranges)
    if include_pro:
        sub.loc[sub["age_bin"] == "PRO", "age_bin"] = "30-34"  # assign to reference group; is_pro flag captures the effect
    else:
        sub = sub[sub["age_bin"] != "PRO"]
    sub["age_bin"] = sub["age_bin"].apply(
        lambda x: "70+" if x in ("70-74", "75-79", "80-84", "80+", "85-89", "85+", "90+", "70+") else x
    )

    sub = assign_continent(sub)
    sub = sub.rename(columns={"continent": "athlete_continent"})
    sub = sub[sub["athlete_continent"] != "Unknown"]

    sub = assign_race_continent(sub)
    sub = sub[sub["race_continent"] != "Unknown"]

    return sub


def fit_model(sub: pd.DataFrame, label: str, include_race_type: bool = False,
              include_is_pro: bool = False) -> list[str]:
    """Fit logistic regression and return output lines."""
    out = []
    out.append(f"\n{'='*70}")
    out.append(f"MODEL: {label}")
    out.append(f"{'='*70}")
    out.append(f"N = {len(sub):,}, DNF = {sub['dnf'].sum():,} ({sub['dnf'].mean()*100:.4f}%)")

    ref_age = "30-34"
    ref_cont = "North America"

    sub["female"] = (sub["gender"] == "Female").astype(int)

    age_bins = sorted([b for b in sub["age_bin"].unique() if b != ref_age])
    for ab in age_bins:
        sub[f"age_{ab}"] = (sub["age_bin"] == ab).astype(int)

    athlete_conts = sorted([c for c in sub["athlete_continent"].unique() if c != ref_cont])
    for c in athlete_conts:
        sub[f"ath_{c.replace(' ', '_')}"] = (sub["athlete_continent"] == c).astype(int)

    race_conts = sorted([c for c in sub["race_continent"].unique() if c != ref_cont])
    for c in race_conts:
        sub[f"race_{c.replace(' ', '_')}"] = (sub["race_continent"] == c).astype(int)

    feature_cols = (
        ["female"]
        + [f"age_{ab}" for ab in age_bins]
        + [f"ath_{c.replace(' ', '_')}" for c in athlete_conts]
        + [f"race_{c.replace(' ', '_')}" for c in race_conts]
        + ["race_year"]
    )

    if include_race_type:
        sub["is_him"] = (sub["race_type"] == "him").astype(int)
        feature_cols.append("is_him")

    if include_is_pro:
        sub["is_pro"] = sub["age_group"].astype(str).isin(["MPRO", "FPRO"]).astype(int)
        feature_cols.append("is_pro")

    X = add_constant(sub[feature_cols].astype(float))
    y = sub["dnf"]
    groups = pd.factorize(sub["event_id"].astype(str))[0]   # one cluster per event (race-year)
    n_clusters = int(groups.max()) + 1

    model = sm.Logit(y, X)
    result = model.fit(disp=0, maxiter=100, cov_type="cluster", cov_kwds={"groups": groups})

    auc = roc_auc_score(y, result.predict(X))
    out.append(f"Converged: {result.mle_retvals['converged']}")
    out.append(f"AUC-ROC: {auc:.6f}")
    out.append(f"Standard errors: clustered by event (event_id), {n_clusters:,} clusters")
    SUMMARY_ROWS.append({"analysis": CURRENT_ANALYSIS, "model": label, "n": len(sub),
                         "n_dnf": int(sub["dnf"].sum()), "dnf_rate_pct": float(sub["dnf"].mean()) * 100,
                         "auc": float(auc), "converged": bool(result.mle_retvals["converged"]),
                         "cov_type": "cluster (event_id)", "n_clusters": n_clusters})

    or_df = pd.DataFrame({
        "OR": np.exp(result.params),
        "CI_low": np.exp(result.conf_int()[0]),
        "CI_high": np.exp(result.conf_int()[1]),
        "p": result.pvalues,
    })
    or_df["sig"] = or_df["p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    for var, r in or_df.iterrows():
        OR_ROWS.append({"analysis": CURRENT_ANALYSIS, "model": label, "variable": var,
                        "OR": r["OR"], "CI_low": r["CI_low"], "CI_high": r["CI_high"], "p": r["p"],
                        "n_clusters": n_clusters})

    # Show only key variables (not all age dummies)
    key_vars = ["female"]
    if include_race_type:
        key_vars.append("is_him")
    if include_is_pro:
        key_vars.append("is_pro")
    key_vars += [c for c in or_df.index if c.startswith("ath_") or c.startswith("race_")]
    key_vars += ["age_70+", "race_year"]
    # Deduplicate (race_ continent vars + race_year could overlap)
    key_vars = list(dict.fromkeys(key_vars))

    out.append("\nKey odds ratios:")
    for v in key_vars:
        if v in or_df.index:
            r = or_df.loc[v]
            out.append(f"  {v:25s} OR={fmt_half_up(r['OR'], 3)} ({fmt_half_up(r['CI_low'], 3)}–{fmt_half_up(r['CI_high'], 3)}) "
                       f"p={fmt_half_up(r['p'], 4)} {r['sig']}")

    return out


CURRENT_ANALYSIS = ""


def main():
    global CURRENT_ANALYSIS
    cols = [
        "finish_status", "race_type", "age_group", "country",
        "event_id", "event_name", "source",
        "overall_sec", "race_year",
    ]
    print("Loading dataset...")
    df = load_dataset(cols)

    # === S1: finish-time DNFs included in the outcome ===
    CURRENT_ANALYSIS = "S1"
    print("\n=== S1: finish-time DNFs included in the outcome ===")
    out = [reproducibility_header("Sensitivity S1: finish-time DNFs included in the outcome")]
    for rt in ["im", "him"]:
        sub = prepare_base(df, race_type=rt, include_finish_time=True)
        out += fit_model(sub, f"{rt.upper()} — with finish-time DNFs")
    output_text = "\n".join(out)
    (RESULTS_DIR / "sensitivity_s1.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(output_text)

    # === S2: Combined IM+HIM ===
    CURRENT_ANALYSIS = "S2"
    print("\n=== S2: Combined model ===")
    out = [reproducibility_header("Sensitivity S2: Combined IM+HIM with race_type")]
    sub = prepare_base(df, race_type=None)
    # 20% simple random sample for computational feasibility (full population and sample sizes printed below)
    sample = sub.sample(frac=0.2, random_state=SEED).copy()
    out.append(f"NOTE: 20% random sample (seed={SEED}) for computational feasibility.")
    out.append(f"Full population: {len(sub):,}. Sample: {len(sample):,}.")
    out += fit_model(sample, "Combined IM+HIM (20% sample)", include_race_type=True)
    output_text = "\n".join(out)
    (RESULTS_DIR / "sensitivity_s2.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(output_text)

    # === S3: Exclude 2020-2021 ===
    CURRENT_ANALYSIS = "S3"
    print("\n=== S3: Exclude 2020-2021 ===")
    out = [reproducibility_header("Sensitivity S3: Exclude 2020-2021")]
    for rt in ["im", "him"]:
        sub = prepare_base(df, race_type=rt, exclude_years=[2020, 2021])
        out += fit_model(sub, f"{rt.upper()} — excluding 2020-2021")
    output_text = "\n".join(out)
    (RESULTS_DIR / "sensitivity_s3.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(output_text)

    # === S4: Include PROs ===
    CURRENT_ANALYSIS = "S4"
    print("\n=== S4: Include PROs ===")
    out = [reproducibility_header("Sensitivity S4: Include PROs with is_pro")]
    for rt in ["im", "him"]:
        sub = prepare_base(df, race_type=rt, include_pro=True)
        out += fit_model(sub, f"{rt.upper()} — with PROs", include_is_pro=True)
    output_text = "\n".join(out)
    (RESULTS_DIR / "sensitivity_s4.txt").write_text(output_text, encoding="utf-8", newline="\n")
    print(output_text)

    write_csv(pd.DataFrame(OR_ROWS), "sensitivity_or.csv",
              "Sensitivity analyses S1-S4 — odds ratios (Tables S5-S8); SEs clustered by event", float_format=CSV_FLOAT)
    write_csv(pd.DataFrame(SUMMARY_ROWS), "sensitivity_summary.csv", "Sensitivity analyses S1-S4 — model summaries",
              float_format=CSV_FLOAT)
    print("\nAll sensitivity analyses complete.")


if __name__ == "__main__":
    main()
