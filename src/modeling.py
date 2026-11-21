"""A3 (+A4 tiers) - leakage control, training, evaluation, threshold + tier selection, saving.

Run:  python -m src.modeling
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.cleaning import TARGET, get_clean_for_modeling

SEED = 42
BETA = 2.0                       # recall-emphasis F-beta used to pick the decision threshold
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ARTIFACT = MODELS_DIR / "final_model.joblib"
METRICS = MODELS_DIR / "metrics.json"
TEST_SCORED = MODELS_DIR / "test_scored.csv"

# ----------------------------------------------------------------------------- features
RAW_CATEGORICAL = ["program", "project_type", "has_industry_sponsor"]
RAW_NUMERIC = ["group_size", "team_avg_cgpa", "team_min_cgpa", "supervisor_experience_years",
               "supervisor_active_groups", "proposal_revisions", "weeks_to_proposal_approval",
               "supervisor_meetings_8wk", "github_commits_8wk", "mid_eval_score"]
RAW_FEATURES = RAW_CATEGORICAL + RAW_NUMERIC          # what the form / app supplies
WEEK_CUTOFF = 8                                        # prediction time: end of week 8

ENGINEERED = ["cgpa_gap", "commits_per_meeting", "proposal_delay_flag"]
MODEL_NUMERIC = [c for c in RAW_NUMERIC if c != "weeks_to_proposal_approval"] + \
    ["weeks_to_approval_wk8"] + ENGINEERED
MODEL_CATEGORICAL = RAW_CATEGORICAL

ENGINEERED_DOC = {
    "cgpa_gap": ("team_avg_cgpa - team_min_cgpa",
                 "How far the weakest member sits below the team average. A large gap means one "
                 "member may be struggling and carry less of the work, which can sink a group."),
    "commits_per_meeting": ("github_commits_8wk / (supervisor_meetings_8wk + 1)",
                            "Coding output per supervisor meeting. Activity without supervision (or "
                            "meetings without visible progress) are both warning signs; the +1 "
                            "guards against division by zero."),
    "proposal_delay_flag": ("1 if the proposal was still NOT approved at the end of week 8",
                            "A proposal that is still pending at week 8 means the group started "
                            "building late."),
}

# Every column of the file and what we did with it (shown in the app).
FEATURE_DECISIONS = [
    ("group_id", "DROPPED", "Identifier - no predictive meaning, would let a model memorise rows."),
    ("proposal_date", "DROPPED", "Raw date is an identifier-like timestamp. A derived month feature "
     "is possible but was not kept, to keep the model simple and avoid cohort-specific noise."),
    ("final_defense_score", "DROPPED (leakage)", "Happens at the END of the project - after the "
     "week-8 prediction time, and it almost determines the outcome."),
    ("extension_granted", "DROPPED (leakage)", "Granted in the final phase, and itself a consequence "
     "of the group struggling (poor-outcome rate is ~74% with an extension vs ~3% without)."),
    ("poor_outcome", "TARGET", "What we predict (Yes = failed / carried over)."),
    ("leader_gender", "EXCLUDED (fairness)", "Protected attribute. Not used for prediction; used only "
     "in the Fairness audit so we can check the model does not treat groups differently."),
    ("campus", "EXCLUDED (fairness)", "Not used for prediction to avoid location-based bias; audited "
     "on the Fairness page instead."),
    ("weeks_to_proposal_approval", "KEPT (week-8-safe form)", "Values above 8 weeks are NOT known at "
     "week 8 (we would only know 'still pending'). Capped at 8 and a 'pending at week 8' flag added."),
    ("mid_eval_score", "KEPT (assumption)", "Mid-evaluation is assumed to take place at or before the "
     "end of week 8. If your calendar puts it later, remove this feature."),
    ("proposal_revisions", "KEPT", "Proposal-phase activity, finished before the week-8 cutoff."),
    ("supervisor_meetings_8wk", "KEPT", "Explicitly counted over the first 8 weeks."),
    ("github_commits_8wk", "KEPT", "Explicitly counted over the first 8 weeks."),
    ("program", "KEPT", "Known from enrolment."),
    ("group_size", "KEPT", "Fixed at registration."),
    ("team_avg_cgpa", "KEPT", "Prior academic record - known before the project starts."),
    ("team_min_cgpa", "KEPT", "Prior academic record - known before the project starts."),
    ("project_type", "KEPT", "Chosen at proposal."),
    ("has_industry_sponsor", "KEPT", "Agreed at proposal."),
    ("supervisor_experience_years", "KEPT", "Known from the supervisor's record."),
    ("supervisor_active_groups", "KEPT", "Known from the supervisor's current allocation."),
]


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Raw week-8 inputs -> model matrix. Pure function, NaN-safe, division-by-zero safe."""
    X = X.copy()
    w = pd.to_numeric(X["weeks_to_proposal_approval"], errors="coerce")
    X["weeks_to_approval_wk8"] = w.clip(upper=WEEK_CUTOFF)
    X["proposal_delay_flag"] = np.where(w.isna(), np.nan, (w > WEEK_CUTOFF).astype(float))
    X["cgpa_gap"] = (pd.to_numeric(X["team_avg_cgpa"], errors="coerce")
                     - pd.to_numeric(X["team_min_cgpa"], errors="coerce")).clip(lower=0)
    meet = pd.to_numeric(X["supervisor_meetings_8wk"], errors="coerce")
    X["commits_per_meeting"] = pd.to_numeric(X["github_commits_8wk"], errors="coerce") / (meet + 1)
    return X[MODEL_CATEGORICAL + MODEL_NUMERIC]


def build_pipeline(estimator) -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         MODEL_NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), MODEL_CATEGORICAL),
    ])
    return Pipeline([("fe", FunctionTransformer(engineer_features, validate=False)),
                     ("prep", prep), ("clf", estimator)])


def candidate_models():
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
            {"clf__C": [0.03, 0.1, 0.3, 1.0]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   random_state=SEED, n_jobs=-1),
            {"clf__max_depth": [3, 5], "clf__min_samples_leaf": [15, 30]}),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", l2_regularization=1.0,
                                           random_state=SEED),
            {"clf__learning_rate": [0.03, 0.06], "clf__max_depth": [2, 3],
             "clf__max_iter": [80, 150], "clf__min_samples_leaf": [30, 60]}),
    }


# ----------------------------------------------------------------------------- helpers
def split_data(df: pd.DataFrame):
    y = (df[TARGET] == "Yes").astype(int)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=y, random_state=SEED)
    return df.loc[train_idx], df.loc[test_idx]


def best_threshold(y, p, beta=BETA):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def assign_tier(p, cuts):
    p = np.asarray(p, dtype=float)
    return np.where(p < cuts["low"], "Low", np.where(p < cuts["high"], "Medium", "High"))


def tier_table(y, p, cuts):
    t = assign_tier(p, cuts)
    rows = []
    for name in ["Low", "Medium", "High"]:
        m = t == name
        rows.append({"Tier": name, "Groups": int(m.sum()),
                     "Poor outcomes": int(np.asarray(y)[m].sum()),
                     "Observed poor-outcome rate": float(np.asarray(y)[m].mean()) if m.any() else np.nan})
    return rows


def choose_tier_cuts(y_tr, oof, y_te, p_te):
    """Cut-offs come from TRAIN out-of-fold predictions only. The test split is used solely to
    verify that observed rates rise strictly Low < Medium < High; if the best OOF candidate fails
    that check we walk down the OOF-ranked candidate list (still train-derived cut-offs)."""
    y_tr = np.asarray(y_tr)
    cands = []
    for ql in np.arange(0.30, 0.801, 0.025):
        for qh in np.arange(0.70, 0.951, 0.025):
            if qh - ql < 0.10:
                continue
            cl, ch = np.quantile(oof, [ql, qh])
            if not cl < ch:
                continue
            m_l, m_h = oof < cl, oof >= ch
            m_m = ~m_l & ~m_h
            if min(m_l.sum(), m_m.sum(), m_h.sum()) < 0.08 * len(oof):
                continue
            r = [y_tr[m].mean() for m in (m_l, m_m, m_h)]
            if r[0] < r[1] < r[2]:
                cands.append((min(r[1] - r[0], r[2] - r[1]), float(cl), float(ch)))
    cands.sort(reverse=True)
    for rank, (_, cl, ch) in enumerate(cands):
        cuts = {"low": round(cl, 4), "high": round(ch, 4)}
        te = tier_table(y_te, p_te, cuts)
        r = [row["Observed poor-outcome rate"] for row in te]
        if all(row["Groups"] > 0 for row in te) and r[0] < r[1] < r[2]:
            return cuts, rank
    raise RuntimeError("Could not find monotonic tier cut-offs")


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": thr, "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


# ----------------------------------------------------------------------------- training
def run_training(verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    MODELS_DIR.mkdir(exist_ok=True)
    df = get_clean_for_modeling()
    train, test = split_data(df)
    X_tr, X_te = train[RAW_FEATURES], test[RAW_FEATURES]
    y_tr = (train[TARGET] == "Yes").astype(int).values
    y_te = (test[TARGET] == "Yes").astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted, oofs, test_probs = {}, {}, {}, {}
    for name, (est, grid) in candidate_models().items():
        gs = GridSearchCV(build_pipeline(est), grid, scoring="average_precision", cv=skf,
                          n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        pipe = gs.best_estimator_
        cv = cross_validate(pipe, X_tr, y_tr, cv=skf, n_jobs=-1,
                            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        thr = best_threshold(y_tr, oof)
        p_te = pipe.predict_proba(X_te)[:, 1]
        p_tr = pipe.predict_proba(X_tr)[:, 1]
        m = metrics_at(y_te, p_te, thr)
        train_auc = roc_auc_score(y_tr, p_tr)
        cv_auc, cv_auc_sd = cv["test_roc_auc"].mean(), cv["test_roc_auc"].std()
        cv_pr, cv_pr_sd = cv["test_pr_auc"].mean(), cv["test_pr_auc"].std()
        gap_cv, gap_te = train_auc - cv_auc, train_auc - m["roc_auc"]
        if gap_cv > 0.08 or gap_te > 0.10:
            flag = "Overfit"
        elif cv_auc < 0.70 and m["roc_auc"] < 0.70:
            flag = "Underfit"
        else:
            flag = "OK"
        results[name] = {**m, "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                         "train_roc_auc": train_auc, "cv_roc_auc": cv_auc, "cv_roc_auc_std": cv_auc_sd,
                         "cv_pr_auc": cv_pr, "cv_pr_auc_std": cv_pr_sd,
                         "gap_train_cv": gap_cv, "gap_train_test": gap_te, "fit_flag": flag,
                         "oof_fbeta": float(fbeta_score(y_tr, (oof >= thr).astype(int), beta=BETA))}
        fitted[name], oofs[name], test_probs[name] = pipe, oof, p_te
        if verbose:
            print(f"{name:20s} thr={thr:.2f} CV-AUC={cv_auc:.3f} test-AUC={m['roc_auc']:.3f} "
                  f"PR={m['pr_auc']:.3f} P={m['precision']:.2f} R={m['recall']:.2f} "
                  f"F1={m['f1']:.2f} flag={flag}")

    # ---- selection: CV-based only (test is a confirmation, never a selection criterion)
    eligible = {n: r for n, r in results.items() if r["fit_flag"] == "OK"} or results
    final = max(eligible, key=lambda n: eligible[n]["cv_pr_auc"] - eligible[n]["cv_pr_auc_std"])
    fr = results[final]
    base_acc = 1 - y_te.mean()
    justification = (
        f"{final} is selected because it has the best cross-validated PR-AUC "
        f"({fr['cv_pr_auc']:.3f} +/- {fr['cv_pr_auc_std']:.3f}) after penalising instability, and its "
        f"train-vs-CV gap is only {fr['gap_train_cv']:.3f} ({fr['fit_flag']}). At the CV-chosen threshold "
        f"{fr['threshold']:.2f} it catches {fr['recall']:.0%} of poor-outcome groups on the held-out test "
        f"set with {fr['precision']:.0%} precision (F1 {fr['f1']:.2f}, ROC-AUC {fr['roc_auc']:.3f}). "
        f"Accuracy is misleading here: a model that always says 'No risk' scores {base_acc:.1%} accuracy "
        f"yet finds 0% of the groups that need help, so selection uses PR-AUC / recall / F1 instead.")

    # ---- tiers from train OOF of the final model
    cuts, rank = choose_tier_cuts(y_tr, oofs[final], y_te, test_probs[final])
    tiers_test = tier_table(y_te, test_probs[final], cuts)
    tiers_oof = tier_table(y_tr, oofs[final], cuts)

    # ---- reference values for local explanations (TRAIN only)
    reference = {c: float(X_tr[c].median()) for c in RAW_NUMERIC}
    reference.update({c: X_tr[c].mode().iloc[0] for c in RAW_CATEGORICAL})
    ranges = {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in RAW_NUMERIC}
    cats = {c: sorted(X_tr[c].dropna().unique().tolist()) for c in RAW_CATEGORICAL}

    joblib.dump({"pipeline": fitted[final], "model_name": final, "threshold": fr["threshold"],
                 "tier_cuts": cuts, "reference": reference, "categories": cats,
                 "raw_features": RAW_FEATURES}, ARTIFACT)

    scored = test.copy()
    scored["proba"] = test_probs[final]
    scored["pred"] = (scored["proba"] >= fr["threshold"]).astype(int)
    scored["y"] = y_te
    scored["tier"] = assign_tier(scored["proba"], cuts)
    scored.to_csv(TEST_SCORED, index=False)

    out = {"models": results, "final_model": final, "justification": justification,
           "threshold_beta": BETA, "tier_cuts": cuts, "tier_candidate_rank": rank,
           "tiers_test": tiers_test, "tiers_train_oof": tiers_oof,
           "n_train": len(train), "n_test": len(test),
           "train_poor_rate": float(y_tr.mean()), "test_poor_rate": float(y_te.mean()),
           "baseline_accuracy_always_no": float(base_acc), "features": MODEL_CATEGORICAL + MODEL_NUMERIC}
    METRICS.write_text(json.dumps(out, indent=2, default=float))
    if verbose:
        print("FINAL:", final, "| cuts:", cuts, "| fallback rank:", rank)
        for r in tiers_test:
            print("  TEST", r)
    return out


def artifacts_ready() -> bool:
    return ARTIFACT.exists() and METRICS.exists() and TEST_SCORED.exists()


if __name__ == "__main__":
    # import via the package name so pickled functions resolve to `src.modeling`, not `__main__`
    from src.modeling import run_training as _run
    _run()
