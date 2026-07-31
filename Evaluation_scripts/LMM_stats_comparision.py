#!/usr/bin/env python3
"""
run_lmm_comparison.py
======================

Combines the biological (DeepLabCut) and computational (simulation) phase
data into one tidy long-format dataframe, then fits a linear mixed-effects
model to test whether the disease trajectory (Healthy -> p45 -> p63 -> p112)
looks the same in biology and in the model.

INPUT FILES
-----------
1. Biological per-file data — produced by the biological pipeline as
   `systemic_raw_data.csv`. Needs at least these columns:
       genotype, p_id, mouse_number, dlc_num, mean_phase_deg
   (this is one row per DLC trial file — exactly the granularity we need).

2. Computational per-seed data — produced by the computational pipeline's
   `export_seed_summary_csv()` as one CSV per timepoint:
       computational_seed_summary_<timepoint>.csv
   with columns: timepoint, seed_id, phase_deg
   Point BIO_CSV / COMP_CSV_GLOB below at wherever your pipelines saved
   these, or pass paths on the command line (see `python run_lmm_comparison.py -h`).

TIDY DATAFRAME
--------------
        Source  Timepoint  Subject     Observation  Phase
0       Bio     Healthy    Mouse554    DLC1          176.2
1       Bio     Healthy    Mouse554    DLC2          178.4
2       Bio     Healthy    Mouse554    DLC3          177.1
3       Bio     Healthy    Mouse672    DLC1          181.3
4       Comp    Healthy    Seed1       Seed1         179.0
5       Comp    Healthy    Seed2       Seed2         178.5
...

MODEL
-----
    Phase ~ Timepoint * Source            (fixed effects)
    random intercept for Subject          (random effects)

    - Timepoint tests whether phase changes with degeneration.
    - Source tests whether there's an overall bio/computational offset.
    - Timepoint:Source is the key term — if it's NOT significant, the
      computational model's trajectory across degeneration matches the
      biological trajectory.
    - Subject as the grouping factor gives every mouse's 3 DLC trials a
      shared random intercept (repeated-measures structure). Each seed is
      its own singleton group, so it contributes no extra within-group
      variance — exactly as it should, since a seed has no repeats.

The interaction is tested two ways:
    1. The individual Timepoint:Source coefficients in the full model
       (each vs. the reference timepoint/source).
    2. A likelihood-ratio test (LRT) of the full model against a reduced
       model without the interaction term — this is the test to report,
       since it tests the whole interaction (all levels at once, correct
       degrees of freedom) rather than one coefficient at a time.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    sys.exit(
        "statsmodels is required for this script.\n"
        "Install it with:  pip install statsmodels --break-system-packages"
    )

from scipy import stats


# =============================================================================
# SETTINGS — edit these to match your file locations and timepoint naming
# =============================================================================

BIO_CSV        = r"/mnt/d/Users/ag399/Dissertation/phase_comparision/systemic_raw_data.csv"
COMP_CSV_GLOB  = r"/mnt/d/Users/ag399/Dissertation/phase_comparision/computational_seed_summary_*.csv"
OUTPUT_DIR     = r"/mnt/d/Users/ag399/Dissertation/phase_comparision/LMM_Output"

# Map each (genotype, p_id) pair from the biological pipeline onto a common
# Timepoint label shared with the computational pipeline. WT is only sampled
# at p49 and stands in for the "Healthy" baseline; SOD1 p49/p63/p112 are the
# three degeneration stages.
BIO_TIMEPOINT_MAP = {
    ("WT",   "p49"):  "Healthy",
    ("SOD1", "p49"):  "p45",
    ("SOD1", "p63"):  "p63",
    ("SOD1", "p112"): "p112",
}
 
# Map each computational timepoint folder label onto the same common labels.
COMP_TIMEPOINT_MAP = {
    "p0":   "Healthy",
    "p45":  "p45",
    "p63":  "p63",
    "p112": "p112",
}
 
# Order matters: this fixes the reference level (first entry) for the model
# and keeps plots/tables in disease-progression order.
TIMEPOINT_ORDER = ["Healthy", "p45", "p63", "p112"]
SOURCE_REFERENCE = "Bio"
 
 
# =============================================================================
# Loaders — build the tidy long dataframe
# =============================================================================
 
def load_biological(bio_csv_path, timepoint_map):
    """
    One row per DLC trial file -> Source='Bio', Timepoint, Subject=Mouse###,
    Observation=DLC#, Phase=mean_phase_deg.
    """
    df = pd.read_csv(bio_csv_path)
 
    required = {"genotype", "p_id", "mouse_number", "dlc_num", "mean_phase_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Biological CSV is missing columns {missing}. "
            f"Expected the per-file output of the biological pipeline "
            f"(systemic_raw_data.csv)."
        )
 
    rows = []
    for _, r in df.iterrows():
        key = (r["genotype"], r["p_id"])
        if key not in timepoint_map:
            continue  # not part of the comparison (e.g. an unused genotype/timepoint)
        rows.append({
            "Source":      "Bio",
            "Timepoint":   timepoint_map[key],
            "Subject":     f"Mouse{int(r['mouse_number'])}",
            "Observation": f"DLC{int(r['dlc_num'])}",
            "Phase":       float(r["mean_phase_deg"]),
        })
 
    tidy = pd.DataFrame(rows)
    print(f"[LOAD] Biological: {len(tidy)} trial-level rows "
          f"from {tidy['Subject'].nunique()} mice.")
    return tidy
 
 
def load_computational(comp_csv_glob, timepoint_map):
    """
    One row per seed -> Source='Comp', Timepoint, Subject=seed_id,
    Observation=seed_id, Phase=phase_deg.
 
    Accepts either a glob pattern matching several per-timepoint CSVs
    (one per computational pipeline run) or a single CSV that already
    contains a 'timepoint' column with all timepoints combined.
    """
    paths = sorted(glob.glob(comp_csv_glob))
    if not paths:
        # maybe the caller passed a single literal file rather than a glob
        if os.path.exists(comp_csv_glob):
            paths = [comp_csv_glob]
        else:
            raise FileNotFoundError(
                f"No computational CSVs found matching: {comp_csv_glob}"
            )
 
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
 
    required = {"timepoint", "seed_id", "phase_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Computational CSV(s) missing columns {missing}. "
            f"Expected the output of export_seed_summary_csv() "
            f"(timepoint, seed_id, phase_deg)."
        )
 
    # Normalise both sides (strip whitespace, lowercase) so "P45", " p45 ",
    # "p45" etc. all match the same map entry. Keeps the lookup forgiving
    # without silently matching the wrong thing.
    norm_map = {str(k).strip().lower(): v for k, v in timepoint_map.items()}
 
    raw_values = sorted(df["timepoint"].astype(str).str.strip().unique().tolist())
    unmatched  = [v for v in raw_values if v.lower() not in norm_map]
 
    rows = []
    for _, r in df.iterrows():
        key = str(r["timepoint"]).strip().lower()
        if key not in norm_map:
            continue
        rows.append({
            "Source":      "Comp",
            "Timepoint":   norm_map[key],
            "Subject":     str(r["seed_id"]),
            "Observation": str(r["seed_id"]),
            "Phase":       float(r["phase_deg"]),
        })
 
    tidy = pd.DataFrame(rows)
    print(f"[LOAD] Computational: {len(tidy)} seed-level rows "
          f"from {len(paths)} file(s).")
    print(f"       Raw timepoint values found in files: {raw_values}")
    if unmatched:
        print(f"       [WARN] These values did NOT match COMP_TIMEPOINT_MAP "
              f"and were dropped: {unmatched}")
        print(f"       COMP_TIMEPOINT_MAP keys are: "
              f"{list(timepoint_map.keys())}")
 
    if tidy.empty:
        raise ValueError(
            "Computational data loaded 0 rows -- every 'timepoint' value in "
            "your CSV(s) failed to match COMP_TIMEPOINT_MAP. See the raw "
            "values printed above and update COMP_TIMEPOINT_MAP (or your "
            "computational script's timepoint naming) so they line up. "
            "Refusing to continue with an empty Comp group, since that "
            "silently turns the interaction test meaningless."
        )
 
    return tidy
 
 
def build_combined_dataframe(bio_csv_path, comp_csv_glob):
    bio_df  = load_biological(bio_csv_path, BIO_TIMEPOINT_MAP)
    comp_df = load_computational(comp_csv_glob, COMP_TIMEPOINT_MAP)
 
    combined = pd.concat([bio_df, comp_df], ignore_index=True)
 
    # Lock in category order so the model's reference levels are Healthy/Bio
    combined["Timepoint"] = pd.Categorical(
        combined["Timepoint"], categories=TIMEPOINT_ORDER, ordered=True
    )
    combined["Source"] = pd.Categorical(
        combined["Source"],
        categories=[SOURCE_REFERENCE] + [s for s in combined["Source"].unique()
                                          if s != SOURCE_REFERENCE],
    )
 
    n_missing_tp = combined["Timepoint"].isna().sum()
    if n_missing_tp:
        print(f"[WARN] Dropping {n_missing_tp} rows with an unmapped timepoint.")
        combined = combined.dropna(subset=["Timepoint"])
 
    return combined.reset_index(drop=True)
 
 
# =============================================================================
# Model fitting
# =============================================================================
 
def fit_models(combined_df):
    """
    Fits the full model (with Timepoint:Source interaction) and a reduced
    model (without it), both by maximum likelihood (REML=False) so their
    log-likelihoods are directly comparable via a likelihood-ratio test.
 
    Random effects: random intercept per Subject. Mice contribute 3 grouped
    DLC observations; seeds are singleton groups and simply add no within-
    group variance, which is exactly the desired behaviour.
    """
    full_formula    = "Phase ~ C(Timepoint) * C(Source)"
    reduced_formula = "Phase ~ C(Timepoint) + C(Source)"
 
    full_model = smf.mixedlm(
        full_formula, combined_df, groups=combined_df["Subject"]
    ).fit(reml=False)
 
    reduced_model = smf.mixedlm(
        reduced_formula, combined_df, groups=combined_df["Subject"]
    ).fit(reml=False)
 
    return full_model, reduced_model
 
 
def likelihood_ratio_test(full_model, reduced_model):
    """
    LRT for the Timepoint:Source interaction as a whole (all interaction
    coefficients simultaneously), which is the statistically correct way
    to test a multi-df categorical interaction rather than eyeballing each
    coefficient's p-value individually.
    """
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_modelwc - reduced_model.df_modelwc
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value
 
 
def print_report(combined_df, full_model, reduced_model):
    print("\n" + "=" * 78)
    print("LINEAR MIXED-EFFECTS MODEL — Biological vs Computational Phase")
    print("=" * 78)
    print(f"  N observations : {len(combined_df)}")
    print(f"  N subjects     : {combined_df['Subject'].nunique()} "
          f"({combined_df.loc[combined_df.Source=='Bio','Subject'].nunique()} mice, "
          f"{combined_df.loc[combined_df.Source=='Comp','Subject'].nunique()} seeds)")
    print(f"  Fixed effects  : Timepoint * Source   (reference: "
          f"{TIMEPOINT_ORDER[0]} / {SOURCE_REFERENCE})")
    print(f"  Random effects : (1 | Subject)   -- mouse-level intercept for repeated DLC trials")
 
    print("\n--- Full model (Timepoint * Source) ---")
    print(full_model.summary())
 
    lr_stat, df_diff, p_value = likelihood_ratio_test(full_model, reduced_model)
    print("\n" + "-" * 78)
    print("LIKELIHOOD-RATIO TEST for the Timepoint:Source interaction")
    print("-" * 78)
    print(f"  chi2({df_diff:.0f}) = {lr_stat:.3f},   p = {p_value:.4f}")
    if df_diff == 0 or np.isnan(p_value):
        print(
            "  [WARN] df_diff is 0 / p is nan -- the full and reduced models are\n"
            "  identical, which means 'Source' has only one level in this data\n"
            "  (e.g. the Comp group loaded 0 rows). This is NOT a valid test --\n"
            "  fix the data loading (see [WARN]s above) before interpreting anything."
        )
    elif p_value >= 0.05:
        print(
            "  -> Interaction NOT significant: no evidence that the computational\n"
            "     trajectory across degeneration differs from the biological one.\n"
            "     This is the result you're hoping for -- the model tracks the\n"
            "     biological disease trajectory."
        )
    else:
        print(
            "  -> Interaction IS significant: the computational and biological\n"
            "     trajectories across degeneration differ from one another."
        )
    print("=" * 78)
 
    return lr_stat, df_diff, p_value
 
 
# =============================================================================
# Main
# =============================================================================
 
def main():
    parser = argparse.ArgumentParser(
        description="Combine biological and computational phase data and fit an LMM."
    )
    parser.add_argument("--bio-csv", default=BIO_CSV,
                         help="Path to the biological systemic_raw_data.csv")
    parser.add_argument("--comp-csv-glob", default=COMP_CSV_GLOB,
                         help="Glob pattern (or single path) for computational "
                              "computational_seed_summary_*.csv files")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                         help="Where to save the combined dataframe / results")
    args = parser.parse_args()
 
    combined_df = build_combined_dataframe(args.bio_csv, args.comp_csv_glob)
 
    os.makedirs(args.output_dir, exist_ok=True)
    combined_csv_path = os.path.join(args.output_dir, "combined_phase_data.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\n[CSV] Saved combined tidy dataframe -> {combined_csv_path}")
 
    full_model, reduced_model = fit_models(combined_df)
    lr_stat, df_diff, p_value = print_report(combined_df, full_model, reduced_model)
 
    # Save a compact results summary alongside the full statsmodels output
    results_txt_path = os.path.join(args.output_dir, "lmm_results.txt")
    with open(results_txt_path, "w") as f:
        f.write(full_model.summary().as_text())
        f.write("\n\nLikelihood-ratio test for Timepoint:Source interaction\n")
        f.write(f"chi2({df_diff:.0f}) = {lr_stat:.3f}, p = {p_value:.4f}\n")
    print(f"[TXT] Saved full model summary + LRT -> {results_txt_path}")
 
 
if __name__ == "__main__":
    main()