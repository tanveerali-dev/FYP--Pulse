"""A5 - local explanations, support actions, and fairness audit (all offline, model-agnostic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling import RAW_FEATURES, assign_tier

LABELS = {
    "program": "degree program", "project_type": "project type",
    "has_industry_sponsor": "industry sponsorship", "group_size": "group size",
    "team_avg_cgpa": "team average CGPA", "team_min_cgpa": "weakest member's CGPA",
    "supervisor_experience_years": "supervisor experience (years)",
    "supervisor_active_groups": "number of groups the supervisor handles",
    "proposal_revisions": "proposal revisions", "weeks_to_proposal_approval": "weeks to proposal approval",
    "supervisor_meetings_8wk": "supervisor meetings in 8 weeks",
    "github_commits_8wk": "code commits in 8 weeks", "mid_eval_score": "mid-evaluation score",
}

# rule-based support action for EVERY model input (engineered features roll up to their raw inputs)
ACTIONS = {
    "supervisor_meetings_8wk": "Schedule a fixed weekly supervisor slot for this group.",
    "github_commits_8wk": "Set weekly code-commit milestones and review the repository each week.",
    "mid_eval_score": "Arrange a mentoring session to address the gaps found in the mid-evaluation.",
    "proposal_revisions": "Hold a proposal-writing workshop to reduce revision cycles.",
    "team_min_cgpa": "Pair the weakest member with a mentor and agree clear individual tasks.",
    "team_avg_cgpa": "Offer academic-skills support and a realistic, scoped project plan.",
    "supervisor_active_groups": "Consider co-supervision or reducing this supervisor's load.",
    "supervisor_experience_years": "Pair the supervisor with a senior mentor or give a supervision checklist.",
    "weeks_to_proposal_approval": "Fast-track proposal approval and agree a week-by-week catch-up plan.",
    "group_size": "Review role allocation inside the group; split work so every member has a clear module.",
    "project_type": "Check that the project scope fits the group's skills; consider simplifying it.",
    "has_industry_sponsor": "Try to link the group with an industry mentor for guidance and feedback.",
    "program": "Connect the group with program-specific project advisors.",
}


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"
    return str(v)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def local_explanation(bundle: dict, row: dict) -> pd.DataFrame:
    """Model-agnostic, offline: replace each raw feature by its TRAIN median/mode and measure how the
    predicted probability changes. `delta` is the change in log-odds (ranks drivers correctly even
    when the probability is saturated near 0 or 1); `dprob` is the plain probability change.
    Positive = this feature value raises the group's risk.

    CGPA note: team_avg_cgpa and team_min_cgpa are strongly linked, so when the average is replaced the
    minimum is shifted by the same amount (the gap is held constant). Otherwise replacing one alone
    would silently change cgpa_gap and blame the wrong feature."""
    pipe, ref = bundle["pipeline"], bundle["reference"]
    base = pd.DataFrame([row])[RAW_FEATURES]
    p0 = float(pipe.predict_proba(base)[0, 1])
    rows = []
    for f in RAW_FEATURES:
        alt = base.copy()
        alt[f] = ref[f]
        if f == "team_avg_cgpa":
            shifted = row["team_min_cgpa"] + (ref[f] - row["team_avg_cgpa"])
            alt["team_min_cgpa"] = float(np.clip(shifted, 0.0, ref[f]))
        p1 = float(pipe.predict_proba(alt)[0, 1])
        rows.append({"feature": f, "label": LABELS[f], "value": row[f], "typical": ref[f],
                     "delta": _logit(p0) - _logit(p1), "dprob": p0 - p1})
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def top_drivers(expl: pd.DataFrame, k: int = 3):
    up = expl[expl["delta"] > 0.02].head(k)
    down = expl[expl["delta"] < -0.02].sort_values("delta").head(k)
    return up, down


def _phrase(r) -> str:
    return f"{r.label} of {_fmt(r.value)} (typical: {_fmt(r.typical)})"


def plain_english(tier: str, up: pd.DataFrame, down: pd.DataFrame) -> str:
    def join(rs):
        ph = [_phrase(r) for r in rs.itertuples()]
        return ph[0] if len(ph) == 1 else ", ".join(ph[:-1]) + " and " + ph[-1]
    if tier in ("High", "Medium") and len(up):
        s = f"This group is at **{tier.lower()} risk** mainly because of its {join(up)}."
    elif len(up):
        s = f"This group is at **low risk**; the biggest watch-point is its {join(up.head(1))}."
    else:
        s = f"This group is at **{tier.lower()} risk** and no single factor stands out as pushing the risk up."
    if len(down):
        s += f" In its favour: {join(down.head(2))}."
    return s


def support_action(up: pd.DataFrame) -> tuple[str, str]:
    if not len(up):
        return "", "No single risk driver stands out - keep the normal monitoring schedule."
    f = up.iloc[0]["feature"]
    return f, ACTIONS[f]


# ----------------------------------------------------------------------------- fairness
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _group_row(g: pd.DataFrame, name, label):
    n = len(g)
    pred, pos = int(g["pred"].sum()), int(g["y"].sum())
    tp = int(((g["pred"] == 1) & (g["y"] == 1)).sum())
    pl, ph = wilson(pred, n)
    rl, rh = wilson(tp, pos)
    return {label: name, "Groups": n, "Actual poor": pos,
            "Predicted poor-outcome rate": pred / n if n else np.nan,
            "Pred CI low": pl, "Pred CI high": ph,
            "Actual poor-outcome rate": pos / n if n else np.nan,
            "Recall": tp / pos if pos else np.nan, "Recall CI low": rl, "Recall CI high": rh}


def gender_table(scored: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_group_row(scored[scored["leader_gender"] == g], g, "Leader gender")
                         for g in ["Male", "Female"]])


def campus_table(scored: pd.DataFrame, min_n: int = 30) -> tuple[pd.DataFrame, list]:
    rows, skipped = [], []
    for c, g in scored.groupby("campus"):
        if len(g) >= min_n:
            rows.append(_group_row(g, c, "Campus"))
        else:
            skipped.append((c, len(g)))
    return pd.DataFrame(rows).sort_values("Predicted poor-outcome rate", ascending=False).reset_index(drop=True), skipped


def overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    return not (a_hi < b_lo or b_hi < a_lo)


def fairness_summary(gt: pd.DataFrame, ct: pd.DataFrame) -> dict:
    out = {}
    m, f = gt.iloc[0], gt.iloc[1]
    out["gender_pred_gap"] = abs(m["Predicted poor-outcome rate"] - f["Predicted poor-outcome rate"])
    out["gender_recall_gap"] = abs(m["Recall"] - f["Recall"]) if pd.notna(m["Recall"]) and pd.notna(f["Recall"]) else np.nan
    out["gender_overlap"] = overlap(m["Pred CI low"], m["Pred CI high"], f["Pred CI low"], f["Pred CI high"])
    out["gender_min_n"] = int(min(m["Groups"], f["Groups"]))
    hi, lo = ct.iloc[0], ct.iloc[-1]
    out["campus_hi"], out["campus_lo"] = hi["Campus"], lo["Campus"]
    out["campus_gap"] = hi["Predicted poor-outcome rate"] - lo["Predicted poor-outcome rate"]
    out["campus_overlap"] = overlap(hi["Pred CI low"], hi["Pred CI high"], lo["Pred CI low"], lo["Pred CI high"])
    rg = ct["Recall"].dropna()
    if len(rg) >= 2:
        out["recall_hi"] = ct.loc[rg.idxmax(), "Campus"]
        out["recall_lo"] = ct.loc[rg.idxmin(), "Campus"]
        out["recall_gap"] = rg.max() - rg.min()
    out["campus_min_n"] = int(ct["Groups"].min())
    return out


def fairness_text(s: dict) -> str:
    g = (f"Leader gender: predicted-risk rates differ by {s['gender_pred_gap']*100:.1f} points and recall by "
         f"{s['gender_recall_gap']*100:.1f} points; their confidence intervals "
         f"{'overlap, so the evidence of disparity is weak' if s['gender_overlap'] else 'do not overlap, so this gap is likely real'}.")
    c = (f" Campuses: the largest gap in predicted-risk rate is {s['campus_gap']*100:.1f} points between "
         f"{s['campus_hi']} and {s['campus_lo']}; intervals "
         f"{'overlap' if s['campus_overlap'] else 'do not overlap'}"
         f"{' - with as few as ' + str(s['campus_min_n']) + ' groups per campus the evidence is weak' if s['campus_overlap'] or s['campus_min_n'] < 50 else ''}.")
    return g + c


# ----------------------------------------------------------------------------- A6: weekly watchlist
def suggested_action(bundle: dict, row: dict) -> str:
    """One support action tied to the group's biggest risk driver."""
    up, _ = top_drivers(local_explanation(bundle, row))
    return support_action(up)[1]


def watchlist_table(scored: pd.DataFrame, bundle: dict, n: int = 100) -> pd.DataFrame:
    """The n highest-risk TEST groups, sorted by probability (ties broken by group_id), each with its
    top risk driver and the matching support action. `_y` (true outcome) is kept only so the app can
    report how many truly poor groups the list captured; the app drops it from display / download."""
    top = scored.sort_values(["proba", "group_id"], ascending=[False, True]).head(n).reset_index(drop=True)
    rows = []
    for i, r in top.iterrows():
        row = {f: r[f] for f in RAW_FEATURES}
        up, _ = top_drivers(local_explanation(bundle, row))
        rows.append({"Rank": i + 1, "group_id": r["group_id"], "Probability": float(r["proba"]),
                     "Tier": r["tier"],
                     "Top risk driver": up.iloc[0]["label"] if len(up) else "no single driver stands out",
                     "Suggested action": support_action(up)[1], "_y": int(r["y"])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- A7: what-if simulator
# Only inputs the department or the group can realistically influence. Everything else (CGPA, program,
# project type, group size, revisions, proposal timing, mid-evaluation) is history or fixed.
LEVERS = {
    "supervisor_meetings_8wk": "Supervisor meetings (first 8 weeks)",
    "github_commits_8wk": "GitHub commits (first 8 weeks)",
    "supervisor_active_groups": "Groups handled by the supervisor",
    "supervisor_experience_years": "Supervisor experience (years, via re-assignment)",
    "has_industry_sponsor": "Industry sponsor",
}


def score_row(bundle: dict, row: dict) -> tuple[float, str]:
    p = float(bundle["pipeline"].predict_proba(pd.DataFrame([row])[RAW_FEATURES])[0, 1])
    return p, str(assign_tier([p], bundle["tier_cuts"])[0])


def lever_candidates(df: pd.DataFrame, bundle: dict) -> dict:
    """Realistic values to try for each lever: the 10th-90th percentile range seen in the data."""
    out = {}
    for f in LEVERS:
        if f == "has_industry_sponsor":
            out[f] = list(bundle["categories"][f])
        else:
            lo, hi = df[f].quantile([0.10, 0.90])
            out[f] = sorted({int(round(v)) for v in np.linspace(lo, hi, 25)})
    return out


def best_moves(bundle: dict, row: dict, candidates: dict) -> pd.DataFrame:
    """Change ONE lever at a time to its most risk-reducing realistic value; rank levers by the drop."""
    pipe = bundle["pipeline"]
    p0, _ = score_row(bundle, row)
    rows = []
    for f, vals in candidates.items():
        alts = pd.DataFrame([{**row, f: v} for v in vals])[RAW_FEATURES]
        ps = pipe.predict_proba(alts)[:, 1]
        i = int(np.argmin(ps))
        best_v, best_p = (vals[i], float(ps[i])) if ps[i] < p0 else (row[f], p0)
        rows.append({"Lever": LEVERS[f], "Current": _fmt(row[f]), "Best realistic value": _fmt(best_v),
                     "Probability now": p0, "Probability if changed": best_p,
                     "Change (pts)": (best_p - p0) * 100})
    return pd.DataFrame(rows).sort_values("Change (pts)").reset_index(drop=True)