"""
Data cleaning & preparation — client_pm (personne morale) & client_pp (personne physique)
Real data extract, Assurance Maghrebia payment facility scoring project.

Every transformation below is logged (row counts before/after) into `transformation_log`,
which is written out to cleaning_log.json and used to build the documentation report.
Nothing here is applied silently — see the printed log at the end for a full audit trail.

IMPORTANT — on synthetic data: this script does NOT fabricate any claims, premium, or
payment-facility-history values and attach them to these real client records. Real client
IDs should never be paired with invented financial behavior, even for prototyping — that
risk was already covered by the fully-synthetic rehearsal dataset built earlier in this
project. See DECISIONS.md / the documentation report for the full reasoning.
"""

import json
import numpy as np
import pandas as pd

RAW_DIR = "."
OUT_DIR = "cleaned"
import os
os.makedirs(OUT_DIR, exist_ok=True)

log = []  # transformation_log: list of dicts {step, description, before, after, detail}

def record(step, description, before=None, after=None, detail=None):
    log.append({"step": step, "description": description, "before": before, "after": after, "detail": detail})
    print(f"[{step}] {description}" + (f" ({before} -> {after})" if before is not None else ""))


# =============================================================================
# Shared helpers
# =============================================================================
def parse_dmp(series):
    """DMP is stored as French-formatted text, e.g. '365,00'. Convert to float days."""
    return series.astype(str).str.replace(",", ".", regex=False).replace("nan", np.nan).astype(float)


def parse_products(series):
    """produit is a comma-separated multi-value string (up to 14 products for one client).
    Returns (nb_produits, primary_produit) — a usable count + the first-listed product,
    rather than leaving the raw multi-value string unusable as a model feature."""
    lists = series.fillna("").apply(lambda x: [p.strip() for p in x.split(",") if p.strip()])
    nb = lists.apply(len)
    primary = lists.apply(lambda x: x[0] if x else np.nan)
    return nb, primary


def dedup_client_id(df, id_col="client_id"):
    """Real duplicate client_id rows were traced to a conflict between two agent-code
    columns (code_agent vs CodeAgent) that disagree on ~24% of PM rows — a classic
    join-multiplication artifact, not genuinely repeated business records. Rather than
    silently keeping one row and discarding the conflicting agent code, we consolidate:
    keep one row per client_id, record every distinct agent code seen for that client,
    and flag whether a conflict existed, so the anomaly is preserved and reportable
    instead of being quietly erased."""
    before = len(df)
    agent_cols = [c for c in ["code_agent", "CodeAgent"] if c in df.columns]

    if agent_cols:
        all_codes = (
            df.groupby(id_col)[agent_cols]
            .agg(lambda s: sorted(set(s.dropna())))
        )
        combined = all_codes.apply(lambda row: sorted(set(sum(row.tolist(), []))), axis=1)
        conflict_flag = combined.apply(lambda x: len(x) > 1)
        agent_map = combined.apply(lambda x: x[0] if x else np.nan)

    df_dedup = df.drop_duplicates(subset=id_col, keep="first").copy()

    if agent_cols:
        df_dedup["agent_code_clean"] = df_dedup[id_col].map(agent_map)
        df_dedup["agent_code_conflict"] = df_dedup[id_col].map(conflict_flag)
        df_dedup = df_dedup.drop(columns=agent_cols)

    after = len(df_dedup)
    n_conflicts = int(df_dedup["agent_code_conflict"].sum()) if agent_cols else 0
    record("dedup_client_id", "Collapsed duplicate client_id rows caused by agent-code join conflict",
           before, after, {"rows_removed": before - after, "clients_with_agent_conflict": n_conflicts})
    return df_dedup


def drop_empty_columns(df, threshold=0.99):
    empty_cols = [c for c in df.columns if df[c].isna().mean() >= threshold]
    df2 = df.drop(columns=empty_cols)
    record("drop_empty_columns", "Dropped columns that are entirely (or almost entirely) empty",
           len(df.columns), len(df2.columns), {"dropped": empty_cols})
    return df2


# =============================================================================
# Clean PM (personne morale)
# =============================================================================
def clean_pm(path):
    df = pd.read_excel(path)
    n0 = len(df)

    df = dedup_client_id(df)
    df = drop_empty_columns(df)

    # statut_client and fidelite_client are redundant (every "Inactif" status maps 1:1
    # onto "Inactif" loyalty tier; "Actif" status simply splits into loyalty sub-tiers).
    # Keep fidelite_client (more granular) and derive a clean boolean from it.
    if "fidelite_client" in df.columns:
        df["is_active"] = df["fidelite_client"].apply(
            lambda x: False if x == "Inactif" else (np.nan if pd.isna(x) else True)
        )
        drop_cols = [c for c in ["statut_client"] if c in df.columns]
        df = df.drop(columns=drop_cols)
        record("resolve_status_redundancy",
               "statut_client dropped (redundant with fidelite_client); derived is_active flag",
               detail={"dropped": drop_cols})

    # anciennete_annees is unreliable for PM: 51% of rows show exactly 0 and 24% show
    # exactly 6 — too clustered to reflect real tenure. anciennete_entreprise (computed
    # from date_creation_entreprise) is far more granular (86 distinct values) and is used
    # instead; anciennete_annees is dropped rather than silently trusted.
    if "anciennete_annees" in df.columns:
        df = df.drop(columns=["anciennete_annees"])
        record("drop_unreliable_tenure", "Dropped anciennete_annees (PM) — unreliable, superseded by anciennete_entreprise")

    # Future-dated company creation is impossible; null these out rather than guess a date.
    if "date_creation_entreprise" in df.columns:
        dates = pd.to_datetime(df["date_creation_entreprise"], errors="coerce")
        future_mask = dates > pd.Timestamp.today()
        n_future = int(future_mask.sum())
        dates[future_mask] = pd.NaT
        df["date_creation_entreprise"] = dates
        record("fix_future_dates", "Nulled out impossible future company-creation dates",
               detail={"rows_affected": n_future})

    # DMP: French-formatted text -> float days. Extreme outliers (e.g. 1462 days) are
    # capped (winsorized) at the 99th percentile rather than dropped, since they are
    # plausible (very delinquent clients) but would distort scaling/log-features if left raw.
    if "DMP" in df.columns:
        dmp = parse_dmp(df["DMP"])
        cap = dmp.quantile(0.99)
        n_capped = int((dmp > cap).sum())
        dmp_capped = dmp.clip(upper=cap)
        df["dmp_days"] = dmp_capped
        df = df.drop(columns=["DMP"])
        record("parse_and_cap_dmp", "Parsed DMP to float days, capped at 99th percentile",
               detail={"p99_cap_days": round(cap, 1), "rows_capped": n_capped, "missing_dmp": int(dmp.isna().sum())})

    # produit: multi-valued comma-separated string -> usable numeric + categorical features.
    if "produit" in df.columns:
        nb, primary = parse_products(df["produit"])
        df["nb_produits"] = nb
        df["produit_principal"] = primary
        record("parse_products", "Parsed multi-value produit field into nb_produits + produit_principal")

    # Capitaux_Totaux is heavily right-skewed (median 100K, max 322M) — add a log feature
    # alongside the raw value rather than replacing it, so both scales are available.
    if "Capitaux_Totaux" in df.columns:
        df["capitaux_log"] = np.log1p(df["Capitaux_Totaux"])
        record("log_transform_capitaux", "Added log1p(Capitaux_Totaux) alongside raw value")

    record("summary_pm", "PM cleaning complete", n0, len(df))
    return df


# =============================================================================
# Clean PP (personne physique)
# =============================================================================
def clean_pp(path):
    df = pd.read_excel(path)
    n0 = len(df)

    df = dedup_client_id(df)
    df = drop_empty_columns(df)

    if "fidelite_client" in df.columns:
        df["is_active"] = df["fidelite_client"].apply(
            lambda x: False if x == "Inactif" else (np.nan if pd.isna(x) else True)
        )
        drop_cols = [c for c in ["statut_client"] if c in df.columns]
        df = df.drop(columns=drop_cols)
        record("resolve_status_redundancy_pp",
               "statut_client dropped (redundant with fidelite_client); derived is_active flag")

    # anciennete_annees (PP) has the same clustering artifact as PM (0 and 6 overrepresented)
    # plus an impossible max of 184 years. Cap implausible values rather than trust them raw.
    if "anciennete_annees" in df.columns:
        n_impossible = int((df["anciennete_annees"] > 70).sum())
        df["anciennete_annees_capped"] = df["anciennete_annees"].where(df["anciennete_annees"] <= 70, np.nan)
        record("cap_tenure_pp", "Capped implausible client tenure values (>70 years) to missing",
               detail={"rows_capped": n_impossible})

    # Age: insurance clients under 16 or over 100 are implausible for this population.
    if "age" in df.columns:
        n_low = int((df["age"] < 16).sum())
        n_high = int((df["age"] > 100).sum())
        df["age_capped"] = df["age"].where((df["age"] >= 16) & (df["age"] <= 100), np.nan)
        record("cap_age", "Nulled implausible ages (<16 or >100)",
               detail={"below_16": n_low, "above_100": n_high})

    dmp_col = "DMP(delai moyen de paiement du client  (en jours))"
    if dmp_col in df.columns:
        dmp = parse_dmp(df[dmp_col])
        cap = dmp.quantile(0.99)
        n_capped = int((dmp > cap).sum())
        dmp_capped = dmp.clip(upper=cap)
        df["dmp_days"] = dmp_capped
        df = df.drop(columns=[dmp_col])
        record("parse_and_cap_dmp_pp", "Parsed DMP to float days, capped at 99th percentile",
               detail={"p99_cap_days": round(cap, 1), "rows_capped": n_capped, "missing_dmp": int(dmp.isna().sum())})

    if "produit" in df.columns:
        nb, primary = parse_products(df["produit"])
        df["nb_produits"] = nb
        df["produit_principal"] = primary
        record("parse_products_pp", "Parsed multi-value produit field into nb_produits + produit_principal")

    cap_col = "Capitaux_Totale" if "Capitaux_Totale" in df.columns else None
    if cap_col:
        df["capitaux_log"] = np.log1p(df[cap_col])
        record("log_transform_capitaux_pp", "Added log1p(Capitaux_Totale) alongside raw value")

    record("summary_pp", "PP cleaning complete", n0, len(df))
    return df


# =============================================================================
# Target variable — derived from real DMP, not fabricated
# =============================================================================
def add_target(df, threshold_days=30):
    """Provisional target: default_proxy = 1 if average payment delay exceeds the
    threshold, else 0. Rows with no DMP value (client never had a payment cycle
    observed, e.g. brand-new clients) get target = NaN and are EXCLUDED from any
    supervised training set — imputing a fabricated outcome for them would invent
    the very thing we are trying to predict.
    This is the "Option B — moderate" definition from the benchmark (30+ days late),
    applied here as a first pass. It is NOT yet validated with the supervisor and
    should be revisited once the target-variable discussion happens."""
    has_dmp = df["dmp_days"].notna()
    df["target_default_proxy"] = np.where(has_dmp, (df["dmp_days"] > threshold_days).astype(float), np.nan)
    n_labeled = int(has_dmp.sum())
    n_unlabeled = int((~has_dmp).sum())
    default_rate = df.loc[has_dmp, "target_default_proxy"].mean()
    record("derive_target", f"Derived target_default_proxy from dmp_days > {threshold_days} days",
           detail={"labeled_rows": n_labeled, "unlabeled_rows_excluded": n_unlabeled,
                   "default_rate_among_labeled": round(float(default_rate), 4)})
    return df


# =============================================================================
# Run
# =============================================================================
pm_clean = clean_pm(f"{RAW_DIR}/client_pm__personne_morale_.xlsx")
pm_clean = add_target(pm_clean)
pm_clean.to_csv(f"{OUT_DIR}/client_pm_clean.csv", index=False)

pp_clean = clean_pp(f"{RAW_DIR}/client_pp__perosonne_physique_.xlsx")
pp_clean = add_target(pp_clean)
pp_clean.to_csv(f"{OUT_DIR}/client_pp_clean.csv", index=False)

# Cross-file ID overlap — flagged, NOT modified (ambiguous which record is authoritative)
overlap = set(pm_clean["client_id"]) & set(pp_clean["client_id"])
record("cross_file_overlap_flagged",
       "client_id values appearing in BOTH pm and pp — flagged for supervisor review, not altered",
       detail={"overlapping_ids": len(overlap)})

with open(f"{OUT_DIR}/cleaning_log.json", "w") as f:
    json.dump(log, f, indent=2, default=str)

print(f"\nDone. Cleaned files + cleaning_log.json written to ./{OUT_DIR}/")
print(f"PM: {len(pm_clean)} rows, {pm_clean.shape[1]} columns")
print(f"PP: {len(pp_clean)} rows, {pp_clean.shape[1]} columns")
