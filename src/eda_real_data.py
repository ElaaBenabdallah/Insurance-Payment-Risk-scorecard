"""
EDA — Real cleaned client data (client_pm & client_pp)
Insurance Payment Facility Risk Scoring project.

Run separately per segment since PM (legal entities) and PP (individuals) have
different feature sets. dmp_days is DELIBERATELY EXCLUDED from the feature analysis
below (it's shown once, separately, only to confirm the target construction) --
including it as a "predictive" feature would be leakage, since target_default_proxy
and risk_tier are both directly derived from it.

High-cardinality categorical columns (activite_principale, produit_principal,
agent_code_clean, profession, Ville) are bucketed to their top 12 categories + "Autres"
for plotting only; Information Value is computed on the full, unbucketed categories.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

TARGET = "target_default_proxy"


def information_value(frame, col, target, top_n_for_plot=12):
    tab = frame.groupby(col, dropna=False)[target].agg(["count", "sum"])
    tab.columns = ["total", "bad"]
    tab["good"] = tab["total"] - tab["bad"]
    total_bad, total_good = tab["bad"].sum(), tab["good"].sum()
    tab["dist_bad"] = (tab["bad"] + 0.5) / (total_bad + 0.5 * len(tab))
    tab["dist_good"] = (tab["good"] + 0.5) / (total_good + 0.5 * len(tab))
    tab["woe"] = np.log(tab["dist_good"] / tab["dist_bad"])
    tab["iv_component"] = (tab["dist_good"] - tab["dist_bad"]) * tab["woe"]
    tab["bad_rate"] = tab["bad"] / tab["total"]
    iv = tab["iv_component"].sum()
    return tab, iv


def bucket_top_n(series, n=12):
    top = series.value_counts().head(n).index
    return series.where(series.isin(top), "Autres")


def run_eda(df, segment_name, numeric_cols, categorical_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = df[df[TARGET].notna()].copy()  # only rows with a real, derivable target
    target_rate = df[TARGET].mean()
    print(f"\n{'='*60}\nSegment: {segment_name} | n={len(df)} | default rate={target_rate:.1%}\n{'='*60}")

    # --- Missingness ---
    missing = df[numeric_cols + categorical_cols].isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("\nMissingness (%):\n", (missing * 100).round(1))
    if len(missing):
        fig, ax = plt.subplots()
        (missing * 100).plot(kind="barh", ax=ax, color="#2E5395")
        ax.set_xlabel("% missing"); ax.set_title(f"[{segment_name}] Missing values")
        fig.tight_layout(); fig.savefig(f"{out_dir}/01_missingness.png", dpi=150); plt.close(fig)

    # --- Target distribution + risk tier ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df[TARGET].value_counts().sort_index().plot(kind="bar", ax=axes[0], color=["#2E5395", "#C0392B"])
    axes[0].set_xticklabels(["Non-défaut (0)", "Défaut (1)"], rotation=0)
    axes[0].set_title(f"Cible binaire — taux={target_rate:.1%}")
    if "risk_tier" in df.columns:
        df["risk_tier"].value_counts().reindex(["Faible", "Moyen", "Élevé"]).plot(
            kind="bar", ax=axes[1], color=["#2E5395", "#7A8FC4", "#C0392B"])
        axes[1].set_title("Répartition par palier de risque"); axes[1].tick_params(axis="x", rotation=0)
    fig.suptitle(f"[{segment_name}] Distribution de la cible")
    fig.tight_layout(); fig.savefig(f"{out_dir}/02_target_distribution.png", dpi=150); plt.close(fig)

    # --- Numeric features: distributions + vs target ---
    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, numeric_cols):
        df[col].dropna().hist(ax=ax, bins=30, color="#2E5395", edgecolor="white")
        ax.set_title(col, fontsize=10)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle(f"[{segment_name}] Distributions numériques"); fig.tight_layout()
    fig.savefig(f"{out_dir}/03_numeric_distributions.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, numeric_cols):
        d0 = df.loc[df[TARGET] == 0, col].dropna()
        d1 = df.loc[df[TARGET] == 1, col].dropna()
        ax.boxplot([d0, d1], labels=["Non-défaut", "Défaut"])
        ax.set_title(col, fontsize=10)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle(f"[{segment_name}] Variables numériques vs cible"); fig.tight_layout()
    fig.savefig(f"{out_dir}/04_numeric_vs_target.png", dpi=150); plt.close(fig)

    # --- Categorical features: IV (full categories) + plot (bucketed) ---
    iv_summary = {}
    ncat = len(categorical_cols)
    ncols2 = 3
    nrows2 = int(np.ceil(ncat / ncols2))
    fig, axes = plt.subplots(nrows2, ncols2, figsize=(16, 4.5 * nrows2))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, categorical_cols):
        d = df[[col, TARGET]].copy()
        d[col] = d[col].fillna("Manquant")
        _, iv = information_value(d, col, TARGET)
        iv_summary[col] = iv
        d[col + "_bucket"] = bucket_top_n(d[col], 12)
        rates = d.groupby(col + "_bucket")[TARGET].mean().sort_values()
        rates.plot(kind="barh", ax=ax, color="#C0392B")
        ax.axvline(target_rate, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{col} (IV={iv:.3f})", fontsize=10)
        ax.set_xlabel("Taux de défaut")
    for ax in axes[ncat:]: ax.axis("off")
    fig.suptitle(f"[{segment_name}] Taux de défaut par catégorie (pointillé = taux global)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{out_dir}/05_categorical_vs_target_IV.png", dpi=150); plt.close(fig)

    iv_table = pd.Series(iv_summary, name="IV").sort_values(ascending=False)
    print("\nInformation Value (catégorielles):\n", iv_table.round(3))

    # --- Correlation matrix (numeric features only, target-derived dmp_days excluded) ---
    corr = df[numeric_cols + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title(f"[{segment_name}] Corrélations")
    fig.tight_layout(); fig.savefig(f"{out_dir}/06_correlation_matrix.png", dpi=150); plt.close(fig)

    corr_with_target = corr[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\nCorrélation avec la cible:\n", corr_with_target.round(3))

    # --- Report ---
    report = f"""# EDA — {segment_name} (données réelles)

**n = {len(df)}** (clients avec DMP observé) | **Taux de défaut (DPD90) = {target_rate:.1%}**

## Variables catégorielles les plus prédictives (Information Value)
{iv_table.head(5).to_frame().to_markdown()}

## Corrélations numériques les plus fortes avec la cible
{corr_with_target.abs().sort_values(ascending=False).head(5).to_frame(name='abs_corr').to_markdown()}

## Note méthodologique
dmp_days est exclu de cette analyse en tant que variable explicative : la cible en est
directement dérivée (fuite de données/leakage garantie si utilisé comme feature).
"""
    with open(f"{out_dir}/eda_report.md", "w") as f:
        f.write(report)

    return iv_table, corr_with_target


# =============================================================================
pm = pd.read_csv("cleaned/client_pm_clean.csv")
pp = pd.read_csv("cleaned/client_pp_clean.csv")

pm_numeric = ["anciennete_entreprise", "Capitaux_Totaux", "capitaux_log", "nb_produits"]
pm_categorical = ["region_entreprise", "segment_secteur", "segment_produit", "segment_contact",
                   "segment_taille_entreprise", "fidelite_client", "is_active", "agent_code_conflict",
                   "Ville", "activite_principale", "produit_principal"]

pp_numeric = ["age_capped", "anciennete_annees_capped", "Capitaux_Totale", "capitaux_log", "nb_produits"]
pp_categorical = ["region", "segment_produit", "fidelite_client", "is_active", "agent_code_conflict",
                   "ville", "profession", "produit_principal"]

iv_pm, corr_pm = run_eda(pm, "PM (Personnes Morales)", pm_numeric, pm_categorical, "eda_real_pm")
iv_pp, corr_pp = run_eda(pp, "PP (Personnes Physiques)", pp_numeric, pp_categorical, "eda_real_pp")

print("\n\n=== Comparaison avec le rehearsal sur données synthétiques ===")
print("Rehearsal synthétique : les variables comportementales/historique de paiement dominaient les variables démographiques.")
print("Données réelles PM — variable catégorielle la plus forte:", iv_pm.index[0], f"(IV={iv_pm.iloc[0]:.3f})")
print("Données réelles PP — variable catégorielle la plus forte:", iv_pp.index[0], f"(IV={iv_pp.iloc[0]:.3f})")
