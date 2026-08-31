"""
Model Development & Evaluation — REAL data (client_pm / client_pp)
Insurance Payment Facility Risk Scoring project.

Run separately for PM and PP using the pre-split, pre-engineered feature sets
(pm/pp_{train,test}_{woe,raw}.csv). Same structure as the synthetic rehearsal:
Logistic/WOE scorecard vs. Random Forest vs. XGBoost, evaluated with Gini/AUC/KS,
calibration, feature importance + SHAP, and a cost-based threshold analysis.

KNOWN LIMITATION (carried over from feature engineering): train/test is a random
stratified split, not time-based -- no transaction/grant date exists in the real
extract. Evaluation here is therefore somewhat optimistic versus a true future-cohort
test. This is not hidden in the output report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import shap

RNG = 42
np.random.seed(RNG)
TARGET = "target_default_proxy"


def ks_statistic(y_true, y_score):
    order = np.argsort(y_score)
    y_true_sorted = np.array(y_true)[order]
    cum_good = np.cumsum(y_true_sorted == 0) / max((y_true_sorted == 0).sum(), 1)
    cum_bad = np.cumsum(y_true_sorted == 1) / max((y_true_sorted == 1).sum(), 1)
    return np.max(np.abs(cum_good - cum_bad))


def run_modeling(segment_name, engineered_dir, out_dir, avg_facility_value):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}\n{segment_name}\n{'='*70}")

    train_woe = pd.read_csv(f"{engineered_dir}/{segment_name.lower()}_train_woe.csv")
    test_woe = pd.read_csv(f"{engineered_dir}/{segment_name.lower()}_test_woe.csv")
    train_raw = pd.read_csv(f"{engineered_dir}/{segment_name.lower()}_train_raw.csv")
    test_raw = pd.read_csv(f"{engineered_dir}/{segment_name.lower()}_test_raw.csv")

    y_train, y_test = train_woe[TARGET].values, test_woe[TARGET].values
    X_train_woe = train_woe.drop(columns=[TARGET]).values
    X_test_woe = test_woe.drop(columns=[TARGET]).values

    print(f"Train: {len(train_woe)} rows ({y_train.mean():.1%} default) | Test: {len(test_woe)} rows ({y_test.mean():.1%} default)")

    raw_cols = [c for c in train_raw.columns if c != TARGET]
    cat_cols = [c for c in raw_cols if not pd.api.types.is_numeric_dtype(train_raw[c]) or train_raw[c].dtype == bool]
    num_cols = [c for c in raw_cols if c not in cat_cols]

    for c in cat_cols:
        train_raw[c] = train_raw[c].astype(str)
        test_raw[c] = test_raw[c].astype(str)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat_ohe = ohe.fit_transform(train_raw[cat_cols]) if cat_cols else np.empty((len(train_raw), 0))
    test_cat_ohe = ohe.transform(test_raw[cat_cols]) if cat_cols else np.empty((len(test_raw), 0))
    ohe_names = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []

    X_train_raw = np.hstack([train_raw[num_cols].values, train_cat_ohe])
    X_test_raw = np.hstack([test_raw[num_cols].values, test_cat_ohe])
    raw_feature_names = num_cols + ohe_names

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

    logit = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RNG)
    logit.fit(X_train_woe, y_train)

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RNG),
        {"n_estimators": [150, 300], "max_depth": [5, 8, None], "min_samples_leaf": [10, 20]},
        n_iter=5, scoring="roc_auc", cv=3, random_state=RNG, n_jobs=-1,
    )
    rf_search.fit(X_train_raw, y_train)
    rf = rf_search.best_estimator_

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=RNG, n_jobs=2),
        {"n_estimators": [150, 300], "max_depth": [3, 4, 5], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
        n_iter=5, scoring="roc_auc", cv=3, random_state=RNG, n_jobs=2,
    )
    xgb_search.fit(X_train_raw, y_train)
    xgb_model = xgb_search.best_estimator_

    models = {
        "Logistique / WOE": (logit, X_test_woe),
        "Random Forest": (rf, X_test_raw),
        "XGBoost": (xgb_model, X_test_raw),
    }

    results, test_scores = [], {}
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    for name, (model, X_te) in models.items():
        proba = model.predict_proba(X_te)[:, 1]
        test_scores[name] = proba
        auc = roc_auc_score(y_test, proba)
        gini = 2 * auc - 1
        ks = ks_statistic(y_test, proba)
        results.append({"model": name, "AUC": auc, "Gini": gini, "KS": ks})
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set_xlabel("Taux de faux positifs"); ax_roc.set_ylabel("Taux de vrais positifs")
    ax_roc.set_title(f"[{segment_name}] Courbes ROC (test set)"); ax_roc.legend()
    fig_roc.tight_layout(); fig_roc.savefig(f"{out_dir}/01_roc_curves.png", dpi=150); plt.close(fig_roc)

    results_df = pd.DataFrame(results).set_index("model").round(3)
    print("\nComparaison des modèles:\n", results_df)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in test_scores.items():
        calib_df = pd.DataFrame({"proba": proba, "actual": y_test})
        calib_df["decile"] = pd.qcut(calib_df["proba"], 10, labels=False, duplicates="drop")
        calib = calib_df.groupby("decile").agg(mean_pred=("proba", "mean"), actual_rate=("actual", "mean"))
        ax.plot(calib["mean_pred"], calib["actual_rate"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Calibration parfaite")
    ax.set_xlabel("Probabilité prédite (décile)"); ax.set_ylabel("Taux de défaut réel (décile)")
    ax.set_title(f"[{segment_name}] Calibration"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_dir}/02_calibration.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    logit_coef = pd.Series(logit.coef_[0], index=train_woe.drop(columns=[TARGET]).columns).sort_values()
    logit_coef.tail(10).plot(kind="barh", ax=axes[0], color="#2E5395"); axes[0].set_title("Logistique/WOE")
    rf_imp = pd.Series(rf.feature_importances_, index=raw_feature_names).sort_values()
    rf_imp.tail(10).plot(kind="barh", ax=axes[1], color="#2E5395"); axes[1].set_title("Random Forest")
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=raw_feature_names).sort_values()
    xgb_imp.tail(10).plot(kind="barh", ax=axes[2], color="#2E5395"); axes[2].set_title("XGBoost")
    fig.suptitle(f"[{segment_name}] Importance des variables"); fig.tight_layout()
    fig.savefig(f"{out_dir}/03_feature_importance.png", dpi=150); plt.close(fig)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_raw)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_raw, feature_names=raw_feature_names, plot_type="bar", show=False)
    plt.title(f"[{segment_name}] SHAP — XGBoost"); plt.tight_layout()
    plt.savefig(f"{out_dir}/04_shap_summary.png", dpi=150); plt.close()

    LOSS_GIVEN_DEFAULT_RATE = 0.6
    LOST_MARGIN_RATE = 0.10
    best_model_name = results_df["Gini"].idxmax()
    best_proba = test_scores[best_model_name]
    thresholds = np.linspace(0.05, 0.95, 37)
    costs = []
    for t in thresholds:
        pred = (best_proba >= t).astype(int)
        false_refusals = ((pred == 1) & (y_test == 0)).sum()
        false_approvals = ((pred == 0) & (y_test == 1)).sum()
        cost = (false_refusals * avg_facility_value * LOST_MARGIN_RATE
                + false_approvals * avg_facility_value * LOSS_GIVEN_DEFAULT_RATE)
        costs.append(cost)
    costs = np.array(costs)
    best_threshold = thresholds[np.argmin(costs)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, costs, color="#C0392B")
    ax.axvline(best_threshold, color="gray", linestyle="--", label=f"Seuil optimal = {best_threshold:.2f}")
    ax.set_xlabel("Seuil de décision"); ax.set_ylabel("Coût total estimé (TND, hypothèses illustratives)")
    ax.set_title(f"[{segment_name}] Analyse de seuil par coût — {best_model_name}"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_dir}/05_cost_threshold.png", dpi=150); plt.close(fig)

    gap = results_df["Gini"].max() - results_df.loc["Logistique / WOE", "Gini"]
    if gap < 0.03:
        reco = "Ecart de Gini faible (<0.03) entre le scorecard interpretable et le meilleur modele -- recommandation : deployer le scorecard en premier pour la transparence."
    else:
        reco = f"{best_model_name} depasse le scorecard d'un ecart significatif (Gini +{gap:.3f}) -- recommandation : l'utiliser en production, accompagne de SHAP pour l'explicabilite."

    report = f"""# Modelisation — {segment_name} (donnees reelles)

**Train:** {len(train_woe)} lignes ({y_train.mean():.1%} defaut) | **Test:** {len(test_woe)} lignes ({y_test.mean():.1%} defaut)
**Split:** aleatoire stratifie (75/25) -- pas de separation temporelle possible, aucune date de transaction disponible dans les donnees reelles actuelles.

## Comparaison des modeles (test set)
{results_df.to_markdown()}

## Recommandation
{reco}

## Limites
- Evaluation optimiste par rapport a un vrai test "futur" (split aleatoire, pas temporel).
- Hypotheses de cout illustratives (perte en cas de defaut = {LOSS_GIVEN_DEFAULT_RATE:.0%} de la valeur, marge perdue en cas de refus injustifie = {LOST_MARGIN_RATE:.0%}) -- a remplacer par de vrais chiffres.
- Biais de selection (reject inference) non resolu : ces donnees ne couvrent que des clients deja acceptes.
"""
    with open(f"{out_dir}/model_comparison_report.md", "w") as f:
        f.write(report)

    return results_df


pm_clean = pd.read_csv("client_pm_clean.csv")
pp_clean = pd.read_csv("client_pp_clean.csv")
pm_avg_value = float(pm_clean["Capitaux_Totaux"].median())
pp_avg_value = float(pp_clean["Capitaux_Totale"].median())

results_pm = run_modeling("PM", ".", "modeling_real_pm", pm_avg_value)
results_pp = run_modeling("PP", ".", "modeling_real_pp", pp_avg_value)

print("\n\n=== Resume final ===")
print("PM:\n", results_pm)
print("\nPP:\n", results_pp)
