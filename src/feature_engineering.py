"""
Feature engineering — client_pm_clean.csv & client_pp_clean.csv
Insurance Payment Facility Risk Scoring project (real data).

Produces, per segment (PM, PP):
  - A WOE-encoded feature set (for the logistic/scorecard model)
  - A raw encoded feature set (for Random Forest / XGBoost)
  - Both split into train/test

IMPORTANT LIMITATION (documented, not hidden): neither file contains a facility-grant
or transaction date -- only date_creation_entreprise (company age, PM only), which is
NOT the same as when a payment facility was granted. A genuine time-based train/test
split (used on the synthetic rehearsal) is therefore NOT POSSIBLE on this real data yet.
A stratified RANDOM split is used instead, which means evaluation on this v1 model is
somewhat optimistic (not tested on a genuinely future cohort) -- flag this explicitly
to the supervisor and revisit if a transaction date becomes available in another system.

Missing values: categorical missingness is treated as its own WOE category ("Manquant"),
not imputed -- this preserves information (missingness itself may be informative) rather
than erasing it. The small amount of real numeric missingness (PP only, ~1%) uses median
imputation + a "was_missing" flag column, not mean -- more robust given the outlier
capping already applied, and the flag keeps the imputation visible to the model instead
of hiding it.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RNG = 42
TARGET = "target_default_proxy"
log = []


def record(step, description, detail=None):
    log.append({"step": step, "description": description, "detail": detail})
    print(f"[{step}] {description}" + (f" -- {detail}" if detail else ""))


def bucket_rare(series, top_n=15, other_label="Autres"):
    top = series.value_counts().head(top_n).index
    return series.where(series.isin(top), other_label)


def fit_woe_categorical(train, col, target):
    tab = train.groupby(col, dropna=False)[target].agg(["count", "sum"])
    tab.columns = ["total", "bad"]
    tab["good"] = tab["total"] - tab["bad"]
    total_bad, total_good = tab["bad"].sum(), tab["good"].sum()
    tab["dist_bad"] = (tab["bad"] + 0.5) / (total_bad + 0.5 * len(tab))
    tab["dist_good"] = (tab["good"] + 0.5) / (total_good + 0.5 * len(tab))
    tab["woe"] = np.log(tab["dist_good"] / tab["dist_bad"])
    return tab["woe"].to_dict()


def fit_woe_numeric(train, col, target, bins=5):
    binned, edges = pd.qcut(train[col], q=bins, retbins=True, duplicates="drop")
    tmp = pd.DataFrame({col: binned, target: train[target]})
    tab = tmp.groupby(col, observed=True)[target].agg(["count", "sum"])
    tab.columns = ["total", "bad"]
    tab["good"] = tab["total"] - tab["bad"]
    total_bad, total_good = tab["bad"].sum(), tab["good"].sum()
    tab["dist_bad"] = (tab["bad"] + 0.5) / (total_bad + 0.5 * len(tab))
    tab["dist_good"] = (tab["good"] + 0.5) / (total_good + 0.5 * len(tab))
    tab["woe"] = np.log(tab["dist_good"] / tab["dist_bad"])
    return edges, tab["woe"].to_dict()


def apply_woe_categorical(frame, col, mapping):
    default = np.mean(list(mapping.values()))
    return frame[col].map(mapping).fillna(default)


def apply_woe_numeric(frame, col, edges, mapping):
    binned = pd.cut(frame[col], bins=edges, include_lowest=True)
    default = np.mean(list(mapping.values()))
    return binned.map(mapping).astype(float).fillna(default)


def engineer(df, segment_name, numeric_cols, categorical_cols, high_card_cols, out_prefix):
    df = df[df[TARGET].notna()].copy()
    record(f"{segment_name}_scope", f"Rows with a derivable target: {len(df)}")

    # Categorical missingness -> own category, never imputed with a "mean"
    for col in categorical_cols + high_card_cols:
        df[col] = df[col].fillna("Manquant")
    record(f"{segment_name}_categorical_missing", "Missing categoricals set to 'Manquant' (own WOE bin), not imputed")

    # Numeric missingness (small, PP only) -> median + was_missing flag, not mean
    numeric_final = []
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            flag_col = f"{col}_was_missing"
            df[flag_col] = df[col].isna().astype(int)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            numeric_final.append(flag_col)
            record(f"{segment_name}_numeric_impute", f"{col}: median imputation ({median_val:.2f}) + was_missing flag",
                   {"n_missing": int(df[flag_col].sum())})
        numeric_final.append(col)

    # High-cardinality columns: bucket to top 15 + "Autres" before WOE (stability)
    for col in high_card_cols:
        df[col] = bucket_rare(df[col], top_n=15)
    record(f"{segment_name}_bucket_rare", f"Bucketed {high_card_cols} to top 15 categories + 'Autres'")

    all_categorical = categorical_cols + high_card_cols

    # Train/test split -- RANDOM (stratified), not time-based: no grant/transaction date exists.
    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[TARGET], random_state=RNG)
    record(f"{segment_name}_split", "Stratified RANDOM split 75/25 (no transaction date available -- see module docstring)",
           {"train_rows": len(train_df), "test_rows": len(test_df)})

    # --- WOE feature set (fit on train, applied to test) ---
    train_woe = pd.DataFrame(index=train_df.index)
    test_woe = pd.DataFrame(index=test_df.index)

    for col in all_categorical:
        mapping = fit_woe_categorical(train_df, col, TARGET)
        train_woe[f"{col}_woe"] = apply_woe_categorical(train_df, col, mapping)
        test_woe[f"{col}_woe"] = apply_woe_categorical(test_df, col, mapping)

    for col in numeric_cols:
        edges, mapping = fit_woe_numeric(train_df, col, TARGET)
        train_woe[f"{col}_woe"] = apply_woe_numeric(train_df, col, edges, mapping)
        test_woe[f"{col}_woe"] = apply_woe_numeric(test_df, col, edges, mapping)

    train_woe[TARGET] = train_df[TARGET].values
    test_woe[TARGET] = test_df[TARGET].values

    # --- Raw feature set (for tree models): numeric as-is (+ flags), categorical as strings ---
    raw_cols = numeric_final + all_categorical
    train_raw = train_df[raw_cols + [TARGET]].copy()
    test_raw = test_df[raw_cols + [TARGET]].copy()

    train_woe.to_csv(f"{out_prefix}_train_woe.csv", index=False)
    test_woe.to_csv(f"{out_prefix}_test_woe.csv", index=False)
    train_raw.to_csv(f"{out_prefix}_train_raw.csv", index=False)
    test_raw.to_csv(f"{out_prefix}_test_raw.csv", index=False)

    record(f"{segment_name}_output", "Saved WOE + raw feature sets (train/test)",
           {"woe_features": len(all_categorical) + len(numeric_cols), "raw_features": len(raw_cols)})
    return train_woe, test_woe, train_raw, test_raw


# =============================================================================
pm = pd.read_csv("cleaned/client_pm_clean.csv")
pp = pd.read_csv("cleaned/client_pp_clean.csv")

# PM: produit_principal (60 categories, IV=0.570) and segment_produit (6 categories, IV=0.350)
# were checked via crosstab -- segment_produit is a coarser roll-up of produit_principal
# (same underlying information at lower resolution). Keeping both would just dilute the
# WOE signal across two collinear features, so segment_produit is DROPPED and
# produit_principal (bucketed) is kept as the primary product feature.
record("pm_redundancy_decision",
       "Dropped segment_produit -- confirmed via crosstab to be a coarser grouping of produit_principal; kept the latter (higher IV, more granular)")

pm_numeric = ["anciennete_entreprise", "capitaux_log", "nb_produits"]
pm_categorical = ["fidelite_client", "is_active", "segment_taille_entreprise", "segment_secteur", "segment_contact", "region_entreprise", "agent_code_conflict"]
pm_high_card = ["produit_principal", "activite_principale", "Ville"]
# agent_code_clean (299 distinct values) excluded from v1 features -- too high-cardinality
# to use directly without overfitting; revisit as an aggregated agent-level feature later.

pp_numeric = ["age_capped", "anciennete_annees_capped", "capitaux_log", "nb_produits"]
pp_categorical = ["fidelite_client", "is_active", "agent_code_conflict", "region"]
pp_high_card = ["produit_principal", "profession", "ville"]
# segment_produit dropped for PP too, for the same redundancy reason as PM.

engineer(pm, "PM", pm_numeric, pm_categorical, pm_high_card, "engineered/pm")
engineer(pp, "PP", pp_numeric, pp_categorical, pp_high_card, "engineered/pp")

import os
os.makedirs("engineered", exist_ok=True)
with open("engineered/feature_engineering_log.json", "w") as f:
    json.dump(log, f, indent=2, default=str)

print("\nDone. Feature sets + log written to ./engineered/")
