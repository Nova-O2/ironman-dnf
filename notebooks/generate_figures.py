"""
Publication-ready figures, sized for Scientific Reports (180 mm double column; Figure 5 at 89 mm single column), each
saved under one base name in two files (save_figure): TIFF 300 DPI (RGB, LZW) and a vector PDF (fonts embedded as
TrueType, pdf.fonttype 42; written without a creation date, so a rerun gives the same bytes).
Style, after the Scientific Reports figure guide: one sans-serif face in one size for all the text, 8 pt
(_common.FIG_FONT_SIZE: tick, axis and colour-bar labels, legends, annotations and heatmap values; Liberation Sans,
registered from notebooks/_fonts/ and put first when present, else the first available of Helvetica, Arial,
Liberation Sans and DejaVu Sans; _common.apply_figure_style), panels labelled lowercase bold (a, b, c) at that size,
no line thinner than 1 pt, neutral greys with colour only on the central contrast, no titles inside the figure; event
type and n of each panel are given in the captions. The shared legend of Figure 3c-d sits below the bottom row,
outside the axes, so its placement does not depend on the face (checked by legend_clear_of_axes).

Figures (numbered by first citation in the manuscript):
1. DNF rate by age group × sex × distance (a, b)
2. DNF rate by year (full vs half distance), in-race and finish-time DNFs stacked (a, b)
3. Dropout localization: agreement of the two localization methods, cross-tabulation heatmaps (a, b), and
   the interval-censored dropout hazard per km along the run, main analysis (c, d)
4. Overall time of finish-time DNFs (DNFs with a recorded finish time) vs the typical cut-off (a, b)
5. ROC curves (full + half distance)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_dataset, infer_gender, infer_last_completed_segment,
    infer_dropout_from_distance, event_course_table, COURSE_COLUMNS,
    COLOR_IM, COLOR_HIM, COLOR_MALE, COLOR_FEMALE,
    COLOR_NEUTRAL, COLOR_LIGHT, COLOR_REFERENCE, COLOR_SHADE, COLOR_AXIS,
    FIG_WIDTH_IN, FIG_HALF_WIDTH_IN, FIG_DPI, FIG_FONT_SIZE, apply_figure_style,
    FIGURES_DIR, RESULTS_DIR,
)
from descriptives import map_age_bin

# Output directory
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from PIL import Image as PILImage
import io

apply_figure_style()

def label_panels(axes) -> None:
    """Lowercase bold panel letters (a, b, …) at the top-left of each axes, in order, at the size of all the other
    text of the figure (FIG_FONT_SIZE)."""
    for ax, label in zip(axes, "abcdefgh"):
        ax.text(-0.13, 1.03, label, transform=ax.transAxes, fontsize=FIG_FONT_SIZE,
                fontweight="bold", va="bottom", ha="left")


def save_figure(fig, path):
    """Save figure as TIFF RGB 300 DPI with LZW compression, at the size set in plt.subplots
    (no bbox cropping, so the printed size equals the journal column width), and as a vector PDF
    of the same size next to it (same base name; fonts embedded as TrueType, pdf.fonttype 42).
    The PDF is written without a creation date, the only run-dependent entry of its metadata,
    so a rerun gives the same bytes.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor="white")
    buf.seek(0)
    img = PILImage.open(buf).convert("RGB")
    img.save(str(path), format="TIFF", compression="tiff_lzw", dpi=(FIG_DPI, FIG_DPI))
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  → {path} ({size_mb:.1f} MB)")
    if size_mb > 10:
        print(f"    WARNING: exceeds 10 MB journal limit!")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, format="pdf", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    print(f"  → {pdf} ({pdf.stat().st_size / 1024 / 1024:.1f} MB)")
    plt.close(fig)


def load_starters():
    """Load and prepare starters dataset (shared across figures), with the event course positions
    (_common.event_course_table, computed on all records before the starter filter)."""
    cols = list(dict.fromkeys([
        "finish_status", "race_type", "age_group", "country",
        "event_id", "event_name", "source",
        "swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec", "overall_sec",
        "total_distance_km", "race_year",
    ] + COURSE_COLUMNS))
    df = load_dataset(cols)
    course = event_course_table(df)
    df = df[df["finish_status"].isin(["FIN", "DNF", "DQ"])].copy()
    df = infer_gender(df)
    df["is_dnf"] = (df["finish_status"] == "DNF").astype(int)
    return df, course


def figure2_temporal(df):
    """Figure 2: DNF rate by year, IM vs HIM, in-race and finish-time DNFs stacked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.1), sharey=True)
    ymax = 0.0

    for ax, rt, label in [(ax1, "im", "IRONMAN®"), (ax2, "him", "IRONMAN® 70.3")]:
        sub = df[df["race_type"] == rt]

        # In-race DNF = no recorded finish time; finish-time DNF = recorded finish time (overall_sec > 0)
        has_time = (sub["finish_status"] == "DNF") & sub["overall_sec"].notna() & (sub["overall_sec"] > 0)
        sub = sub.copy()
        sub["is_in_race_dnf"] = ((sub["finish_status"] == "DNF") & ~has_time).astype(int)
        sub["is_finish_time_dnf"] = has_time.astype(int)

        annual = sub.groupby("race_year").agg(
            N=("is_dnf", "count"),
            inr=("is_in_race_dnf", "sum"),
            ft=("is_finish_time_dnf", "sum"),
        )
        annual["inr_rate"] = annual["inr"] / annual["N"] * 100
        annual["ft_rate"] = annual["ft"] / annual["N"] * 100

        years = annual.index
        # 2026 is a partial season (collection 15 October 2026): hatched bars.
        hatches = ["//" if y == 2026 else None for y in years]
        bars_v = ax.bar(years, annual["inr_rate"], color=COLOR_NEUTRAL, width=0.8, label="In-race DNF")
        bars_t = ax.bar(years, annual["ft_rate"], bottom=annual["inr_rate"], width=0.8,
                        color=COLOR_LIGHT, label="Finish-time DNF")
        for bv, bt, h in zip(bars_v, bars_t, hatches):
            if h:
                bv.set_hatch(h); bt.set_hatch(h)
                bv.set_edgecolor("white"); bt.set_edgecolor("white")
        # 95% Wilson CI of the total DNF rate per season.
        total = (annual["inr"] + annual["ft"]).to_numpy()
        tot_rate = total / annual["N"].to_numpy() * 100
        lo, hi = proportion_confint(total, annual["N"].to_numpy(), alpha=0.05, method="wilson")
        ax.errorbar(years, tot_rate, yerr=[tot_rate - lo * 100, hi * 100 - tot_rate],
                    fmt="none", ecolor=COLOR_AXIS, elinewidth=1.0, capthick=1.0, capsize=1.2)
        partial = f", 2026 (partial) = {int(annual.loc[2026, 'N']):,}" if 2026 in annual.index else ""
        print(f"  Figure 2 {label}: {int(annual['N'].sum()):,} starters{partial}")

        # 2020: races were held but with a reduced field (COVID-19) — shaded, not "no race".
        ax.axvspan(2019.5, 2020.5, color=COLOR_SHADE, zorder=0, lw=0)

        ax.set_xlabel("Year")
        if rt == "him":   # one legend, in the empty upper-left area of panel b
            ax.legend(loc="upper left", frameon=False)
        ax.set_xlim(2001.5, 2026.5)
        ymax = max(ymax, float((annual["inr_rate"] + annual["ft_rate"]).max()))

    # y axis is shared: set the limit once, then place the 2020 label inside it.
    ax1.set_ylim(0, ymax * 1.18)
    for ax in (ax1, ax2):
        ax.text(2020, ymax * 1.18 * 0.80, "2020", ha="center", va="top", fontsize=FIG_FONT_SIZE, color=COLOR_REFERENCE)
    ax1.set_ylabel("DNF rate (%)")
    label_panels((ax1, ax2))
    plt.tight_layout()
    save_figure(fig, FIGURES_DIR / "Figure2_temporal_dnf.tiff")


# Figure 3c, d: one legend shared by the two hazard panels, outside the axes and centred below the bottom row. The
# strip it needs is measured on the rendered legend and reserved before the final layout (tight_layout rect), so the
# placement does not depend on the font face or its width; legend_clear_of_axes() fails the build only on a real
# overlap. The y axis of each panel reaches FIG3_HEADROOM times its highest plotted CI bound.
FIG3_SIZE_IN = (FIG_WIDTH_IN, 6.45)
FIG3_HEADROOM = 1.1
FIG3_LEGEND_GAP = 0.01   # figure fraction left between the top of the legend and the strip reserved for the axes


def legend_clear_of_axes(fig, leg) -> str:
    """Fail if the figure legend leaves the figure or overlaps the tight bounding box of any axes of the figure (data
    area, ticks, tick labels and axis labels, colour bars included); returns a line for the log."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lb = leg.get_window_extent(renderer)
    fb = fig.bbox
    if lb.x0 < fb.x0 or lb.x1 > fb.x1 or lb.y0 < fb.y0 or lb.y1 > fb.y1:
        raise AssertionError("Figure 3 legend extends beyond the figure")
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    hit = [i for i, b in enumerate(boxes) if lb.overlaps(b)]
    if hit:
        raise AssertionError(f"Figure 3 legend overlaps axes {hit} of the figure")
    gap_mm = (min(b.y0 for b in boxes) - lb.y1) / fig.dpi * 25.4
    return (f"legend below the bottom row, {lb.width / fb.width * 100:.0f}% of the figure width, "
            f"{gap_mm:.1f} mm under the lowest axes element")


def figure3_dropout(df, course):
    """Figure 3: Dropout localization.
    Top row (a, b): agreement of the two localization methods — cross-tabulation of the last
                    completed segment by splits (rows) against the segment by distance (columns;
                    each event's own course positions, dropout_point.py section 3) for in-race DNFs
                    with both, as row percentages (the two methods measure different constructs; see
                    caption). The distances of events with an unusable course are missing
                    (_common.attach_course, through infer_dropout_from_distance).
    Bottom row (c, d): the dropout hazard — interval-censored hazard per km along the run (main
                    analysis, T2 rule; pooled over official-source events, 95% cluster-bootstrap
                    CI), read from results/dropout_hazard_curve.csv, the output of dropout_hazard.py.
                    Each event's terminal bin is excluded from the curve; the plot stops at the first
                    bin dominated by terminal bins (terminal_bin). The dashed line marks the median
                    position of the first run checkpoint at or beyond 1 km. One legend for both panels
                    sits below the bottom row, outside the axes (checked by legend_clear_of_axes); the
                    caption carries the details of each element.
    """
    curve = pd.read_csv(RESULTS_DIR / "dropout_hazard_curve.csv", comment="#")
    curve = curve[curve["analysis"] == "main"]
    events = pd.read_csv(RESULTS_DIR / "dropout_hazard_events.csv", comment="#")
    events = events[events["analysis"] == "main"]

    dnf = df[df["finish_status"] == "DNF"].copy()
    dnf = infer_last_completed_segment(dnf)
    in_race = ~(dnf["overall_sec"].notna() & (dnf["overall_sec"] > 0))
    dnf_off = dnf[(dnf["source"] == "official") & in_race].copy()
    dnf_off = infer_dropout_from_distance(dnf_off, course)
    both = dnf_off[
        (~dnf_off["last_completed_segment"].astype(str).isin(["unknown", "finish_time"])) &
        (dnf_off["dropout_distance"].astype(str) != "unknown")
    ].copy()

    run_lengths = {"im": 42.20, "him": 21.10}   # x-axis limits (nominal run lengths)
    fig, axes = plt.subplots(2, 2, figsize=FIG3_SIZE_IN)

    # --- Bottom row (c, d): dropout hazard, per-km curves ---
    first_cps = []
    for col_idx, rt in enumerate(["im", "him"]):
        ax = axes[1, col_idx]
        # Bins up to the first one dominated by the events' terminal bins (intervals ending at the
        # finish mat); every plotted bin is drawn over its full width, so the last start gets its
        # right edge appended.
        c = curve[curve["race_type"] == rt].sort_values("km_bin_start")
        first_term = c.loc[c["terminal_bin"].astype(bool), "km_bin_start"].min()
        c = c[c["km_bin_start"] < first_term]
        x = np.append(c["km_bin_start"].to_numpy(dtype=float), float(c["km_bin_end"].iloc[-1]))

        def ext(col: str) -> np.ndarray:
            v = c[col].to_numpy(dtype=float)
            return np.append(v, v[-1])

        band = ax.fill_between(x, ext("ci_low"), ext("ci_high"), step="post", color=COLOR_REFERENCE, alpha=0.3,
                               linewidth=0, label="95% CI")
        (curve_line,) = ax.step(x, ext("hazard_per_1e4"), where="post", color=COLOR_NEUTRAL, linewidth=1.4,
                                label="Pooled hazard")
        ev = events[events["race_type"] == rt]
        first_cp = float(ev["first_checkpoint_km"].median())
        first_cps.append(first_cp)
        ref = ax.axvline(first_cp, color=COLOR_AXIS, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Kilometres into the running segment")
        ax.set_ylabel("DNFs per 10,000 athlete-km")
        ax.set_xlim(0, run_lengths[rt])
        ax.set_ylim(0, FIG3_HEADROOM * float(c["ci_high"].max()))
        legend_handles = [curve_line, band, ref]   # the same three elements in both panels

    # --- Top row (a, b): agreement of the two localization methods (cross-tabulation heatmaps) ---
    for col_idx, rt in enumerate(["im", "him"]):
        ax = axes[0, col_idx]
        col_order = ["swim", "bike", "run"]
        # Row percentages over ALL records of the row (as in dropout_point.txt). No record of the
        # cross-tabulation is in the "no timing mat crossed" class (distance 0): the three plotted columns
        # hold every record, so each row sums to 100% (checked here, so the figure cannot drop a column).
        sub = both[both["race_type"] == rt]
        ct = pd.crosstab(
            sub["last_completed_segment"].astype(str),
            sub["dropout_distance"].astype(str),
            normalize="index",
        ) * 100
        hidden = sorted(set(ct.columns) - set(col_order))
        if hidden:
            raise AssertionError(f"Figure 3 {rt}: cross-tabulation records outside the plotted columns {hidden}; "
                                 "the rows would not sum to 100%")

        row_order = ["swim", "T1", "bike", "T2", "run"]
        ct = ct.reindex(index=[r for r in row_order if r in ct.index],
                        columns=[c for c in col_order if c in ct.columns],
                        fill_value=0)
        display = {"swim": "Swim", "bike": "Cycling", "run": "Run"}   # the manuscript's segment names
        ct = ct.rename(index=display, columns=display)

        sns.heatmap(ct, annot=True, fmt=".1f", cmap="Greys", ax=ax, vmin=0, vmax=100,
                    linewidths=1.0, linecolor="white", annot_kws={"fontsize": FIG_FONT_SIZE},
                    cbar_kws={"label": "Row %"})
        # Colour bar drawn as vectors in the PDF (matplotlib rasterizes a gradient of 50 or more steps), its steps
        # edged in their own colour, so that no viewer shows seams between them.
        solids = ax.collections[0].colorbar.solids
        solids.set_rasterized(False)
        solids.set_edgecolor("face")
        ax.set_xlabel("Distance method (segment)")
        ax.set_ylabel("Split method (last completed)")
        ax.tick_params(axis="y", labelrotation=0)
        print(f"  Figure 3 {rt}: cross-tabulation n = {len(sub):,}")

    label_panels(axes.flat)

    # Shared legend of c and d below the bottom row: the value of the reference line is given once when both
    # panels share it at the printed precision, otherwise per panel.
    cps = [f"{v:.1f}" for v in first_cps]
    cp_label = f"{cps[0]} km" if cps[0] == cps[1] else f"c {cps[0]} km, d {cps[1]} km"
    leg = fig.legend(legend_handles, ["Pooled hazard", "95% CI", f"Median first checkpoint ({cp_label})"],
                     loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.canvas.draw()
    legend_top = leg.get_window_extent(fig.canvas.get_renderer()).y1 / fig.bbox.height   # figure fraction
    plt.tight_layout(rect=(0, legend_top + FIG3_LEGEND_GAP, 1, 1))
    print(f"  Figure 3: {legend_clear_of_axes(fig, leg)}")   # after the final layout
    save_figure(fig, FIGURES_DIR / "Figure3_dropout_localization.tiff")


def figure1_age_sex(df):
    """Figure 1: DNF rate by age group × sex × distance."""
    # Exclude PROs and finish-time DNFs (the Figure 1 population)
    sub = df[~df["age_group"].astype(str).isin(["MPRO", "FPRO"])].copy()
    is_ft = (sub["finish_status"] == "DNF") & sub["overall_sec"].notna() & (sub["overall_sec"] > 0)
    sub = sub[~is_ft]
    sub = sub[sub["gender"].isin(["Male", "Female"])]

    sub["age_bin"] = sub["age_group"].astype(str).apply(map_age_bin)
    sub = sub[~sub["age_bin"].isin(["Other", "PRO"])]

    # Consolidate 70+
    sub["age_bin"] = sub["age_bin"].apply(
        lambda x: "70+" if x in ("70-74", "75-79", "80-84", "80+", "85-89", "85+", "90+", "70+") else x
    )

    bin_order = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                 "50-54", "55-59", "60-64", "65-69", "70+"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 3.1), sharey=True)

    for ax, rt, title in [(axes[0], "im", "IRONMAN®"),
                           (axes[1], "him", "IRONMAN® 70.3")]:
        rt_sub = sub[sub["race_type"] == rt]
        print(f"  Figure 1 {title}: n = {len(rt_sub):,} starters "
              f"(Male {int((rt_sub['gender'] == 'Male').sum()):,}, Female {int((rt_sub['gender'] == 'Female').sum()):,})")

        for gender, color, marker in [("Male", COLOR_MALE, "o"), ("Female", COLOR_FEMALE, "s")]:
            g_sub = rt_sub[rt_sub["gender"] == gender]
            agg = g_sub.groupby("age_bin")["is_dnf"].agg(n_dnf="sum", n="count").reindex(bin_order).dropna()
            rates = (agg["n_dnf"] / agg["n"] * 100).to_numpy()
            lo, hi = proportion_confint(agg["n_dnf"].to_numpy(), agg["n"].to_numpy(), alpha=0.05, method="wilson")
            x = [bin_order.index(b) for b in agg.index]   # keep x aligned with the tick labels (bins may be missing)
            ax.errorbar(x, rates, yerr=[rates - lo * 100, hi * 100 - rates],
                        marker=marker, color=color, label=gender, linewidth=1.3, markersize=3.5,
                        capsize=1.5, elinewidth=1.0, capthick=1.0)

        ax.set_xticks(range(len(bin_order)))
        ax.set_xticklabels([b.replace("-", "–") for b in bin_order], rotation=45, ha="right")
        ax.set_xlabel("Age group (years)")
        ax.legend(loc="upper left", frameon=False)

    axes[0].set_ylabel("DNF rate (%)")
    label_panels(axes)
    plt.tight_layout()
    save_figure(fig, FIGURES_DIR / "Figure1_age_sex_dnf.tiff")


def figure5_roc(df):
    """Figure 5: ROC curves for IM and HIM logistic regression models."""
    from sklearn.metrics import roc_curve, roc_auc_score
    import statsmodels.api as sm
    from statsmodels.tools import add_constant
    from dnf_prediction import prepare_data

    fig, ax = plt.subplots(figsize=(FIG_HALF_WIDTH_IN, FIG_HALF_WIDTH_IN))

    for rt, color, label in [("im", COLOR_IM, "IRONMAN®"), ("him", COLOR_HIM, "IRONMAN® 70.3")]:
        sub = prepare_data(df, rt)

        # Same encoding as dnf_prediction.py
        ref_age = "30-34"
        sub["age_bin"] = sub["age_group"].astype(str).apply(map_age_bin)
        sub = sub[~sub["age_bin"].isin(["Other", "PRO"])]
        sub["age_bin"] = sub["age_bin"].apply(
            lambda x: "70+" if x in ("70-74", "75-79", "80-84", "80+", "85-89", "85+", "90+", "70+") else x
        )

        sub["female"] = (sub["gender"] == "Female").astype(int)
        age_bins = sorted([b for b in sub["age_bin"].unique() if b != ref_age])
        for ab in age_bins:
            sub[f"age_{ab}"] = (sub["age_bin"] == ab).astype(int)

        ref_cont = "North America"
        for c in sorted([c for c in sub["athlete_continent"].unique() if c != ref_cont]):
            sub[f"ath_{c.replace(' ', '_')}"] = (sub["athlete_continent"] == c).astype(int)
        for c in sorted([c for c in sub["race_continent"].unique() if c != ref_cont]):
            sub[f"race_{c.replace(' ', '_')}"] = (sub["race_continent"] == c).astype(int)

        feature_cols = (
            ["female"]
            + [f"age_{ab}" for ab in age_bins]
            + [f"ath_{c.replace(' ', '_')}" for c in sorted([c for c in sub["athlete_continent"].unique() if c != ref_cont])]
            + [f"race_{c.replace(' ', '_')}" for c in sorted([c for c in sub["race_continent"].unique() if c != ref_cont])]
            + ["race_year"]
        )

        X = add_constant(sub[feature_cols].astype(float))
        y = sub["dnf"]

        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        y_prob = model.predict(X)
        auc = roc_auc_score(y, y_prob)

        fpr, tpr, _ = roc_curve(y, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=1.3, label=f"{label} (AUC-ROC {auc:.3f})")

    ax.plot([0, 1], [0, 1], color=COLOR_REFERENCE, linestyle="--", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal")
    plt.tight_layout()
    save_figure(fig, FIGURES_DIR / "Figure5_roc_curves.tiff")


# Figure 4 axes: IRONMAN clipped at 20 h (records beyond it are counted in dropout_point.txt
# section 1), IRONMAN 70.3 over its natural range (maximum 11.9 h). Constant bin widths of
# 15 min (IRONMAN) and 10 min (IRONMAN 70.3): both typical cut-offs fall on a bin edge, and the
# panels carry nearly the same number of bins per unit of width (80 and 72).
FIG4_AXIS = {"im": (0, 20, 15), "him": (0, 12, 10)}   # x min (h), x max (h), bin width (min)
FIG4_CUTOFF_H = {"im": 17.0, "him": 8.5}


def figure4_finish_time(df):
    """Figure 4: overall time of finish-time DNFs (DNFs with a recorded finish time) vs the typical cut-off."""
    dnf = df[df["finish_status"] == "DNF"].copy()
    ft = dnf[dnf["overall_sec"].notna() & (dnf["overall_sec"] > 0)].copy()
    ft["hours"] = ft["overall_sec"].astype("float64") / 3600

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 2.9))

    for ax, rt in [(axes[0], "im"), (axes[1], "him")]:
        sub = ft[ft["race_type"] == rt]
        lo, hi, width = FIG4_AXIS[rt]
        edges = np.arange(lo * 60, hi * 60 + 1, width) / 60   # whole minutes: cut-offs are exact edges
        cutoff = FIG4_CUTOFF_H[rt]
        # No bar edges: no line may be thinner than 1 pt, and a 1-pt edge would cover a third of each 1-mm bar.
        ax.hist(sub["hours"], bins=edges, color=COLOR_NEUTRAL, linewidth=0)
        ax.axvline(cutoff, color=COLOR_AXIS, linestyle="--", linewidth=1.0,
                   label=f"Typical cut-off ({cutoff:.1f} h)")
        ax.set_xlim(lo, hi)
        ax.set_xticks(np.arange(lo, hi + 1e-9, 2.0))
        ax.set_xlabel("Overall time (hours)")
        ax.set_ylabel("Finish-time DNFs (count)")
        ax.legend(loc="upper left", frameon=False)
        n_out = int(((sub["hours"] < lo) | (sub["hours"] > hi)).sum())
        print(f"  Figure 4 {rt}: n = {len(sub):,}; outside the {lo}-{hi} h axis: {n_out:,}; "
              f"bin width {width} min; cut-off on a bin edge: {bool(np.isclose(edges, cutoff).any())}")

    label_panels(axes)
    plt.tight_layout()
    save_figure(fig, FIGURES_DIR / "Figure4_finish_time_distribution.tiff")


def main():
    print("Loading dataset...")
    df, course = load_starters()
    print(f"Loaded {len(df):,} starters")
    print()

    print("Generating figures...")
    figure1_age_sex(df)
    figure2_temporal(df)
    figure3_dropout(df, course)
    figure4_finish_time(df)
    figure5_roc(df)
    print(f"\nDone. TIFF and PDF figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
