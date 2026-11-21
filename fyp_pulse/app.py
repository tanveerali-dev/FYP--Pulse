"""FYP-Pulse - Final Year Project Early Warning System (Streamlit)."""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import charts, cleaning, explain, modeling

st.set_page_config(page_title="FYP-Pulse", page_icon="🎓", layout="wide")

PAGES = ["Data Quality", "Dashboard", "Models", "Predict", "Fairness", "Watchlist", "What-if"]
PAGE_ICONS = {"Data Quality": "🧹", "Dashboard": "📊", "Models": "🤖", "Predict": "🔮", "Fairness": "⚖️",
              "Watchlist": "📋", "What-if": "🎛️"}
TIER_COLORS = {"Low": ("#dcfce7", "#166534"), "Medium": ("#fef3c7", "#92400e"), "High": ("#ffe4e6", "#9f1239")}
TIER_DOT = {"Low": "#16a34a", "Medium": "#f59e0b", "High": "#e11d48"}

CSS = """
<style>
:root{--ink:#1b1f3b;--muted:#6b7280;--line:#e5e7f0;--bg:#f6f7fb;--card:#ffffff;
      --primary:#4f46e5;--primary-soft:#eef0ff;--radius:16px;
      --shadow:0 1px 2px rgba(27,31,59,.04),0 8px 24px rgba(27,31,59,.06);}
html,body,[class*="css"],.stApp{font-family:Inter,"Segoe UI","Helvetica Neue",Arial,sans-serif;}
.stApp{background:radial-gradient(1200px 400px at 85% -10%,#e8eaff 0%,rgba(246,247,251,0) 60%),var(--bg);}
.block-container{padding-top:2.2rem;padding-bottom:4rem;max-width:1280px;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}

/* page header */
.pulse-hero{margin:0 0 1.4rem 0;}
.pulse-eyebrow{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--primary);background:var(--primary-soft);padding:.28rem .7rem;border-radius:999px;}
.pulse-hero h1{font-size:2.15rem;font-weight:800;letter-spacing:-.02em;color:var(--ink);margin:.55rem 0 .25rem;padding:0;}
.pulse-hero p{color:var(--muted);font-size:.98rem;margin:0;max-width:780px;line-height:1.5;}
h2,h3{color:var(--ink);letter-spacing:-.01em;}
h3{font-size:1.1rem !important;font-weight:700 !important;}

/* sidebar */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#151a33 0%,#1f2547 100%);border-right:none;}
section[data-testid="stSidebar"] *{color:#e7e9ff;}
.pulse-brand{display:flex;align-items:center;gap:.7rem;margin:.4rem 0 .2rem;}
.pulse-logo{width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  font-size:1.25rem;background:linear-gradient(135deg,#6366f1,#8b5cf6);box-shadow:0 6px 16px rgba(99,102,241,.45);}
.pulse-brand b{font-size:1.2rem;letter-spacing:-.01em;color:#fff;}
.pulse-tag{font-size:.78rem;color:#a5abd6 !important;margin:.1rem 0 1.2rem;line-height:1.4;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:.55rem .8rem;border-radius:10px;width:100%;
  transition:background .15s;cursor:pointer;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(255,255,255,.08);}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:rgba(129,140,248,.22);
  box-shadow:inset 3px 0 0 #818cf8;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child{display:none;}
section[data-testid="stSidebar"] [data-testid="stRadio"] p{font-weight:600;font-size:.95rem;}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{font-size:.7rem;font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;color:#8087b8 !important;}

/* metric cards */
div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
  padding:1rem 1.15rem;box-shadow:var(--shadow);}
div[data-testid="stMetricLabel"] p{color:var(--muted);font-size:.8rem;font-weight:600;}
div[data-testid="stMetricValue"]{color:var(--ink);font-weight:800;letter-spacing:-.02em;}

/* tabs as pills */
.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--line);padding-bottom:.4rem;}
.stTabs [data-baseweb="tab"]{height:2.4rem;padding:0 1rem;border-radius:999px;background:transparent;
  color:var(--muted);font-weight:600;}
.stTabs [data-baseweb="tab"]:hover{background:var(--primary-soft);color:var(--primary);}
.stTabs [aria-selected="true"]{background:var(--primary) !important;color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}

/* cards: charts, tables, forms, alerts */
div[data-testid="stForm"]{background:var(--card);border:1px solid var(--line);border-radius:20px;
  padding:1.6rem 1.6rem 1.2rem;box-shadow:var(--shadow);}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;
  box-shadow:0 1px 2px rgba(27,31,59,.04);}
div[data-testid="stImage"] img,div[data-testid="stPyplot"] img{background:#fff;border:1px solid var(--line);
  border-radius:var(--radius);padding:.6rem;box-shadow:var(--shadow);}
div[data-testid="stAlert"]{border-radius:14px;border:1px solid var(--line);}
div[data-testid="stCaptionContainer"]{color:var(--muted);}
hr{border-color:var(--line) !important;margin:1.8rem 0 !important;}

/* buttons */
.stButton>button,.stDownloadButton>button,div[data-testid="stFormSubmitButton"]>button{border-radius:12px;
  font-weight:700;padding:.55rem 1.4rem;border:1px solid var(--line);transition:all .15s;}
.stButton>button[kind="primary"],div[data-testid="stFormSubmitButton"]>button[kind="primary"],
div[data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"]{
  background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;border:none;box-shadow:0 6px 18px rgba(79,70,229,.35);}
.stButton>button:hover,.stDownloadButton>button:hover,div[data-testid="stFormSubmitButton"]>button:hover{
  transform:translateY(-1px);box-shadow:0 8px 22px rgba(79,70,229,.3);}

/* inputs */
div[data-baseweb="input"],div[data-baseweb="select"]>div{border-radius:10px !important;}

/* risk badge + insight callout */
.risk-badge{display:inline-flex;align-items:center;gap:.5rem;padding:.45rem 1.1rem;border-radius:999px;
  font-weight:800;font-size:1.05rem;border:1px solid rgba(0,0,0,.05);}
.risk-dot{width:.7rem;height:.7rem;border-radius:50%;display:inline-block;}
.insight{background:var(--primary-soft);border-left:4px solid var(--primary);border-radius:10px;
  padding:.7rem .95rem;color:var(--ink);font-size:.88rem;line-height:1.5;margin:.5rem 0 1.4rem;}
.panel-title{font-size:.78rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:.2rem 0 .4rem;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"<div class='pulse-hero'><span class='pulse-eyebrow'>{eyebrow}</span><h1>{title}</h1>{sub}</div>",
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def load_cleaned():
    raw = cleaning.load_raw()
    return cleaning.clean_data(raw)


@st.cache_resource(show_spinner=False)
def load_bundle():
    return joblib.load(modeling.ARTIFACT)


@st.cache_data(show_spinner=False)
def load_metrics():
    return json.loads(modeling.METRICS.read_text())


@st.cache_data(show_spinner=False)
def load_test_scored():
    return pd.read_csv(modeling.TEST_SCORED)


def ensure_artifacts() -> bool:
    """Train on first start when artifacts are missing. Returns True when ready."""
    if modeling.artifacts_ready():
        return True
    try:
        with st.spinner("First start: training models (about a minute)..."):
            modeling.run_training(verbose=False)
        st.cache_resource.clear()
        st.cache_data.clear()
        return True
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not train the models: {e}")
        return False


def pct(x):
    return "-" if pd.isna(x) else f"{x*100:.1f}%"


# ----------------------------------------------------------------------------- pages
def page_data_quality():
    page_header("Step 1 · Cleaning", "Data Quality",
                "What was wrong with the raw file and how each problem was fixed (rule-based cleaning).")
    df, rules, per_col, s = load_cleaned()
    c = st.columns(5)
    c[0].metric("Initial rows", f"{s['initial_rows']:,}")
    c[1].metric("Cleaned rows", f"{s['cleaned_rows']:,}", delta=f"-{s['rows_removed']} removed", delta_color="off")
    c[2].metric("Values corrected", f"{s['values_corrected']:,}")
    c[3].metric("Poor outcomes", f"{s['total_poor']:,}")
    c[4].metric("Poor-outcome rate", f"{s['poor_rate']*100:.1f}%")

    t1, t2, t3 = st.tabs(["Rules broken & fixes", "Corrections per column", "Cleaned data"])
    with t1:
        st.markdown("**Rows that broke each rule and how each was fixed**")
        st.dataframe(rules[rules["Rows affected"] > 0].reset_index(drop=True), hide_index=True)
        st.caption("Rules with zero affected rows are hidden. Missing values are imputed (median / mode) "
                   "inside the model pipeline using TRAIN data only, to avoid leakage.")
    with t2:
        st.dataframe(per_col)
        st.bar_chart(per_col[["Values corrected", "Missing (imputed)"]])
    with t3:
        full = cleaning.impute_for_report(df)
        st.caption("Download copy: every missing value filled with the column median/mode. "
                   "(The models do NOT use this copy; they impute after the train/test split.)")
        st.dataframe(full.head(200), hide_index=True)
        st.download_button("⬇️ Download cleaned CSV", full.to_csv(index=False).encode("utf-8"),
                           file_name="fyp_pulse_cleaned.csv", mime="text/csv")


def page_dashboard():
    page_header("Step 2 · Insights", "Dashboard",
                "Which factors are linked to a poor outcome? Rates are computed on the cleaned data; "
                "all features are known by the end of week 8.")
    df, *_ = load_cleaned()
    items = charts.dashboard_charts(df)
    for row in (items[:2], items[2:]):
        cols = st.columns(2)
        for col, it in zip(cols, row):
            with col:
                st.subheader(it["title"])
                st.pyplot(it["fig"])
                st.markdown(f"<div class='insight'>💡 {it['insight']}</div>", unsafe_allow_html=True)
                plt.close(it["fig"])
    st.markdown("### About the engineered feature: `cgpa_gap`")
    st.markdown("`cgpa_gap = team_avg_cgpa - team_min_cgpa` measures how far the **weakest member** sits below the "
                "team average. A big gap suggests an uneven team where one member may struggle or be carried, which "
                "is a common reason groups fail. It is used in the model together with `commits_per_meeting` "
                "(commits ÷ (meetings + 1), division-by-zero safe) and `proposal_delay_flag` (proposal still "
                "unapproved at week 8).")


def page_models():
    page_header("Step 3 · Modeling", "Models")
    if not ensure_artifacts():
        return
    m = load_metrics()
    st.caption(f"Stratified 80/20 split (seed 42): {m['n_train']} train / {m['n_test']} test groups. "
               f"Poor-outcome rate: {m['train_poor_rate']*100:.1f}% train, {m['test_poor_rate']*100:.1f}% test.")
    st.success(f"**Selected model: {m['final_model']}**\n\n{m['justification']}")

    t1, t2, t3, t4 = st.tabs(["Dropped features & why", "Model comparison (test)", "Over/underfitting check", "Confusion matrices"])
    with t1:
        st.markdown("**Dropped features and why** - every column is judged against *what is known at the end of week 8*.")
        st.dataframe(pd.DataFrame(modeling.FEATURE_DECISIONS, columns=["Column", "Decision", "Reason"]),
                     hide_index=True)
        st.markdown("**Engineered features**")
        st.dataframe(pd.DataFrame([(k, v[0], v[1]) for k, v in modeling.ENGINEERED_DOC.items()],
                                  columns=["Feature", "Formula", "Why it should matter"]),
                     hide_index=True)
    with t2:
        rows = []
        for n, r in m["models"].items():
            rows.append({"Model": n, "Threshold": r["threshold"], "Accuracy": r["accuracy"], "Precision": r["precision"],
                         "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"]})
        tbl = pd.DataFrame(rows).set_index("Model")
        st.dataframe(tbl.style.format("{:.3f}").highlight_max(axis=0, color="#dcfce7"))
        st.info(f"Why not accuracy? Always predicting 'No risk' would score {m['baseline_accuracy_always_no']*100:.1f}% "
                f"accuracy on the test set while catching 0% of at-risk groups. Thresholds were tuned on out-of-fold "
                f"TRAIN predictions (F{m['threshold_beta']:.0f}, recall-weighted), never on the test set.")
        tr = pd.DataFrame([{"Model": n, "Tuned parameters": json.dumps(r["best_params"])} for n, r in m["models"].items()])
        st.dataframe(tr, hide_index=True)
    with t3:
        rows = [{"Model": n, "Train ROC-AUC": r["train_roc_auc"], "CV ROC-AUC": r["cv_roc_auc"],
                 "CV std": r["cv_roc_auc_std"], "Test ROC-AUC": r["roc_auc"],
                 "Gap train-CV": r["gap_train_cv"], "Gap train-test": r["gap_train_test"], "Verdict": r["fit_flag"]}
                for n, r in m["models"].items()]
        tbl = pd.DataFrame(rows).set_index("Model")
        st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns if c != "Verdict"}))
        st.caption("Rule: 'Overfit' if train-CV gap > 0.08 or train-test gap > 0.10; 'Underfit' if CV and test ROC-AUC are "
                   "both < 0.70. Models flagged 'Overfit' are not eligible for selection.")
    with t4:
        cols = st.columns(len(m["models"]))
        for col, (n, r) in zip(cols, m["models"].items()):
            with col:
                fig = charts.confusion_figure(r["confusion_matrix"], f"{n}\nthr={r['threshold']:.2f}")
                st.pyplot(fig)
                plt.close(fig)
        st.caption("Rows = actual, columns = predicted. TP = at-risk groups caught, FN = at-risk groups missed, "
                   "FP = false alarms. Percentages are out of each actual class (so the TP % is the recall).")


def risk_badge(tier: str) -> str:
    bg, fg = TIER_COLORS[tier]
    return (f"<span class='risk-badge' style='background:{bg};color:{fg}'>"
            f"<span class='risk-dot' style='background:{TIER_DOT[tier]}'></span>{tier} risk</span>")


def page_predict():
    page_header("Step 4 · Early warning", "Predict")
    if not ensure_artifacts():
        return
    try:
        bundle, m = load_bundle(), load_metrics()
    except Exception as e:  # noqa: BLE001
        st.error(f"Model files could not be loaded ({e}). Re-run `python -m src.modeling`.")
        return
    ref, cats, R = bundle["reference"], bundle["categories"], cleaning.VALID_RANGES
    st.caption(f"Model: **{bundle['model_name']}** - uses only information known by the end of week 8.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Team & project**")
            program = st.selectbox("Program", cats["program"], index=cats["program"].index(ref["program"]))
            project_type = st.selectbox("Project type", cats["project_type"], index=cats["project_type"].index(ref["project_type"]))
            sponsor = st.selectbox("Industry sponsor", cats["has_industry_sponsor"], index=cats["has_industry_sponsor"].index(ref["has_industry_sponsor"]))
            group_size = st.number_input("Group size", R["group_size"][0], R["group_size"][1], int(ref["group_size"]), 1)
            avg = st.number_input("Team average CGPA", 0.0, 4.0, round(ref["team_avg_cgpa"], 2), 0.01)
            mn = st.number_input("Team minimum CGPA", 0.0, 4.0, min(round(ref["team_min_cgpa"], 2), 4.0), 0.01)
        with c2:
            st.markdown("**Supervisor**")
            exp = st.number_input("Supervisor experience (years)", 0, 45, int(ref["supervisor_experience_years"]), 1)
            act = st.number_input("Supervisor's active groups", 1, 15, int(ref["supervisor_active_groups"]), 1)
            meet = st.slider("Supervisor meetings (first 8 weeks)", 0, 30, int(ref["supervisor_meetings_8wk"]))
        with c3:
            st.markdown("**Progress**")
            rev = st.slider("Proposal revisions", 0, 10, int(ref["proposal_revisions"]))
            wk = st.slider("Weeks to proposal approval", 0, 30, int(ref["weeks_to_proposal_approval"]),
                           help="Values above 8 mean the proposal was still pending at week 8.")
            commits = st.number_input("GitHub commits (first 8 weeks)", 0, 1000, int(ref["github_commits_8wk"]), 1)
            mid = st.number_input("Mid-evaluation score", 0.0, 100.0, round(ref["mid_eval_score"], 1), 0.1)
        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        if mn > avg:
            st.session_state.pop("pred", None)
            st.error("Team minimum CGPA cannot be higher than the team average CGPA. Please correct the CGPA values.")
        elif group_size == 1 and abs(mn - avg) > 1e-9:
            st.session_state.pop("pred", None)
            st.error("For a one-person group the minimum CGPA must equal the average CGPA.")
        else:
            st.session_state["pred"] = {
                "program": program, "project_type": project_type, "has_industry_sponsor": sponsor,
                "group_size": int(group_size), "team_avg_cgpa": float(avg), "team_min_cgpa": float(mn),
                "supervisor_experience_years": int(exp), "supervisor_active_groups": int(act),
                "proposal_revisions": int(rev), "weeks_to_proposal_approval": int(wk),
                "supervisor_meetings_8wk": int(meet), "github_commits_8wk": int(commits), "mid_eval_score": float(mid)}

    row = st.session_state.get("pred")
    if row:
        try:
            p = float(bundle["pipeline"].predict_proba(pd.DataFrame([row])[modeling.RAW_FEATURES])[0, 1])
            tier = str(modeling.assign_tier([p], bundle["tier_cuts"])[0])
            expl = explain.local_explanation(bundle, row)
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")
            return
        st.divider()
        a, b, c = st.columns(3)
        a.metric("Predicted poor outcome", "Yes" if p >= bundle["threshold"] else "No")
        b.metric("Probability score", f"{p:.4f}")
        c.markdown("Risk tier<br>" + risk_badge(tier), unsafe_allow_html=True)
        st.caption(f"Decision threshold {bundle['threshold']:.2f} (chosen on out-of-fold train predictions). "
                   f"Tier cut-offs: Low < {bundle['tier_cuts']['low']:.4f} <= Medium < {bundle['tier_cuts']['high']:.4f} <= High.")

        up, down = explain.top_drivers(expl)
        st.subheader("Why this score?")
        left, right = st.columns(2)
        with left:
            st.markdown("**🔺 Top factors increasing risk**")
            st.dataframe(up[["label", "value", "typical", "delta", "dprob"]].rename(columns={
                "label": "Feature", "value": "This group", "typical": "Typical", "delta": "Risk impact (log-odds)",
                "dprob": "Probability change"}).style.format(
                {"Risk impact (log-odds)": "{:+.2f}", "Probability change": "{:+.3f}"}), hide_index=True)
        with right:
            st.markdown("**🔻 Top factors decreasing risk**")
            st.dataframe(down[["label", "value", "typical", "delta", "dprob"]].rename(columns={
                "label": "Feature", "value": "This group", "typical": "Typical", "delta": "Risk impact (log-odds)",
                "dprob": "Probability change"}).style.format(
                {"Risk impact (log-odds)": "{:+.2f}", "Probability change": "{:+.3f}"}), hide_index=True)
        shown = pd.concat([up, down]).sort_values("delta")
        if len(shown):
            fig, ax = plt.subplots(figsize=(6, 2 + 0.3 * len(shown)))
            ax.barh(shown["label"], shown["delta"], height=0.62,
                    color=[charts.GOOD if d < 0 else charts.RED for d in shown["delta"]])
            ax.axvline(0, color=charts.INK, lw=0.9)
            ax.set_xlabel("Effect on risk (log-odds) vs. a typical group", fontsize=8)
            charts.clean_axes(ax, grid_axis="x")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.markdown(explain.plain_english(tier, up, down))
        _, action = explain.support_action(up)
        st.info(f"**Suggested support action:** {action}")

    st.divider()
    st.subheader("How reliable are the risk tiers? (held-out test set)")
    cuts = m["tier_cuts"]
    st.caption(f"Cut-offs were chosen from out-of-fold TRAIN predictions: Low < {cuts['low']:.4f}, "
               f"Medium {cuts['low']:.4f}-{cuts['high']:.4f}, High >= {cuts['high']:.4f}.")
    t = pd.DataFrame(m["tiers_test"]).set_index("Tier")
    st.dataframe(t.style.format({"Observed poor-outcome rate": "{:.1%}"}))
    r = t["Observed poor-outcome rate"].tolist()
    if r[0] < r[1] < r[2]:
        st.success("Observed poor-outcome rate rises strictly Low < Medium < High.")


def page_fairness():
    page_header("Step 5 · Audit", "Fairness")
    if not ensure_artifacts():
        return
    try:
        scored, m = load_test_scored(), load_metrics()
    except Exception as e:  # noqa: BLE001
        st.error(f"Scored test data is missing ({e}). Re-run `python -m src.modeling`.")
        return
    st.caption(f"Audit on the held-out test split ({len(scored)} groups) using the final model "
               f"({m['final_model']}) at threshold {m['models'][m['final_model']]['threshold']:.2f}. "
               "Gender and campus are NOT model inputs; they are used only here.")
    gt = explain.gender_table(scored)
    ct, skipped = explain.campus_table(scored, 30)
    summ = explain.fairness_summary(gt, ct)

    t1, t2 = st.tabs(["Gender of group leader", "Campus"])
    fmt = {"Predicted poor-outcome rate": "{:.1%}", "Actual poor-outcome rate": "{:.1%}", "Recall": "{:.1%}",
           "Pred CI low": "{:.1%}", "Pred CI high": "{:.1%}", "Recall CI low": "{:.1%}", "Recall CI high": "{:.1%}"}
    with t1:
        st.dataframe(gt.style.format(fmt, na_rep="-"), hide_index=True)
        g1, g2 = st.columns(2)
        g1.metric("Gap in predicted poor-outcome rate", f"{summ['gender_pred_gap']*100:.1f} pts")
        g2.metric("Gap in recall", "-" if pd.isna(summ["gender_recall_gap"]) else f"{summ['gender_recall_gap']*100:.1f} pts")
    with t2:
        st.dataframe(ct.style.format(fmt, na_rep="-"), hide_index=True)
        if skipped:
            st.caption("Excluded (fewer than 30 test groups): " + ", ".join(f"{c} (n={n})" for c, n in skipped))
        h1, h2 = st.columns(2)
        h1.metric("Largest gap in predicted rate", f"{summ['campus_gap']*100:.1f} pts",
                  help=f"{summ['campus_hi']} vs {summ['campus_lo']}")
        h1.caption(f"{summ['campus_hi']} vs {summ['campus_lo']}")
        if "recall_gap" in summ:
            h2.metric("Largest gap in recall", f"{summ['recall_gap']*100:.1f} pts")
            h2.caption(f"{summ['recall_hi']} vs {summ['recall_lo']}")
        fig, ax = plt.subplots(figsize=(7, 3))
        y = ct["Predicted poor-outcome rate"] * 100
        err = [y - ct["Pred CI low"] * 100, ct["Pred CI high"] * 100 - y]
        ax.bar(ct["Campus"], y, yerr=err, color=charts.PRIMARY, width=0.6, capsize=3,
               error_kw={"ecolor": charts.INK, "elinewidth": 1})
        ax.set_ylabel("Predicted poor-outcome rate (%)", fontsize=8)
        ax.tick_params(axis="x", rotation=25)
        charts.clean_axes(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.info(explain.fairness_text(summ))
    st.caption("Wilson 95% confidence intervals. Small groups give wide intervals - treat gaps as indications to "
               "investigate, not proof of bias.")


# ----------------------------------------------------------------------------- A6: weekly watchlist
def _tier_cell(v):
    bg, fg = TIER_COLORS.get(v, ("#ffffff", "#000000"))
    return f"background-color:{bg};color:{fg};font-weight:700"


@st.cache_data(show_spinner="Scoring and explaining the 100 highest-risk groups (one-time, ~10 s)...")
def load_watchlist(n: int = 100):
    return explain.watchlist_table(load_test_scored(), load_bundle(), n)


def page_watchlist():
    page_header("Bonus · A6", "Weekly Watchlist",
                "The coordinator can support only K groups at a time. Pick K and get the K highest-risk "
                "groups, ready to download.")
    if not ensure_artifacts():
        return
    try:
        scored = load_test_scored()
        full = load_watchlist(100)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build the watchlist ({e}). Re-run `python -m src.modeling`.")
        return
    k = st.slider("Coordinator capacity K (groups supported at a time)", 10, 100, 30, 1)
    wl = full.head(k)
    table = wl.drop(columns="_y")
    total_poor, base_rate = int(scored["y"].sum()), float(scored["y"].mean())
    caught = int(wl["_y"].sum())
    prec = caught / k
    random_exp = k * base_rate

    c = st.columns(4)
    c[0].metric("Truly poor-outcome groups captured", f"{caught} of {total_poor}")
    c[1].metric(f"Precision at K = {k}", f"{prec:.1%}")
    c[2].metric("Expected from K random picks", f"{random_exp:.1f}",
                help=f"K x overall poor-outcome rate ({base_rate:.1%}) on the test groups")
    c[3].metric("Lift over random", f"{prec / base_rate:.1f}x")
    st.markdown(f"<div class='insight'>Picking the <b>{k}</b> highest-risk groups captures <b>{caught}</b> of the "
                f"<b>{total_poor}</b> groups that really ended badly ({caught / total_poor:.0%}). "
                f"Picking {k} groups at random would capture only about <b>{random_exp:.1f}</b>.</div>",
                unsafe_allow_html=True)

    styler = table.style.format({"Probability": "{:.4f}"})
    styled = (styler.map if hasattr(styler, "map") else styler.applymap)(_tier_cell, subset=["Tier"])  # pandas 2.0 / 2.1+
    st.dataframe(styled, hide_index=True, height=min(640, 38 + 35 * len(table)))
    st.download_button("⬇️ Download watchlist CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"fyp_pulse_watchlist_top{k}.csv", mime="text/csv", type="primary")
    st.caption("Built from the held-out test groups, sorted by predicted probability. Outcomes are used only to "
               "report how many truly poor groups the list captured; they are not part of the download.")


# ----------------------------------------------------------------------------- A7: what-if simulator
def page_whatif():
    page_header("Bonus · A7", "What-if simulator",
                "Change only the levers the department or the group can influence and watch the probability, "
                "tier and suggested action update live.")
    if not ensure_artifacts():
        return
    try:
        bundle, scored = load_bundle(), load_test_scored()
        df, *_ = load_cleaned()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load the model or data ({e}). Re-run `python -m src.modeling`.")
        return
    ref, cats = bundle["reference"], bundle["categories"]

    # ---- choose the baseline group
    options = ["A test-set group"]
    if st.session_state.get("pred"):
        options.insert(0, "The group from the Predict page")
    src = st.radio("Start from", options, horizontal=True) if len(options) > 1 else options[0]
    if src.startswith("The group"):
        base, gid = dict(st.session_state["pred"]), "predict"
    else:
        ordered = scored.sort_values("proba", ascending=False)
        info = dict(zip(ordered["group_id"], zip(ordered["proba"], ordered["tier"])))
        gid = st.selectbox("Test group (highest risk first)", list(info),
                           format_func=lambda g: f"{g}  ·  {info[g][0]:.0%}  ·  {info[g][1]} risk")
        r = scored[scored["group_id"] == gid].iloc[0]
        base = {}
        for f in modeling.RAW_FEATURES:
            v = ref[f] if pd.isna(r[f]) else r[f]
            base[f] = str(v) if f in modeling.RAW_CATEGORICAL else float(v)
    sig = f"{gid}_{abs(hash(tuple(sorted(base.items()))))}"
    cand = explain.lever_candidates(df, bundle)

    left, right = st.columns([1, 1.15], gap="large")
    new = dict(base)
    with left:
        st.markdown("<div class='panel-title'>Levers you can change</div>", unsafe_allow_html=True)
        for f, label in explain.LEVERS.items():
            if f == "has_industry_sponsor":
                opts = cats[f]
                new[f] = st.radio(label, opts, index=opts.index(base[f]), horizontal=True, key=f"wi_{f}_{sig}")
            else:
                cur = int(round(base[f]))
                lo, hi = min(int(df[f].min()), cur), max(int(df[f].max()), cur)
                new[f] = float(st.slider(label, lo, hi, cur, key=f"wi_{f}_{sig}"))
        fixed = [explain.LABELS[f] for f in modeling.RAW_FEATURES if f not in explain.LEVERS]
        st.caption("Held fixed (history or not controllable): " + ", ".join(fixed) + ".")

    p0, t0 = explain.score_row(bundle, base)
    p1, t1 = explain.score_row(bundle, new)
    with right:
        st.markdown("<div class='panel-title'>Before vs. after</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Before", f"{p0:.1%}")
        b.metric("After", f"{p1:.1%}", delta=f"{(p1 - p0) * 100:+.1f} pts", delta_color="inverse")
        a.markdown("Tier before<br>" + risk_badge(t0), unsafe_allow_html=True)
        b.markdown("Tier after<br>" + risk_badge(t1), unsafe_allow_html=True)

        cuts = bundle["tier_cuts"]
        fig, ax = plt.subplots(figsize=(6, 1.7))
        ax.barh(["After", "Before"], [p1, p0], height=0.55, color=[TIER_DOT[t1], TIER_DOT[t0]], zorder=2)
        for x in (cuts["low"], cuts["high"]):
            ax.axvline(x, color=charts.INK, ls=(0, (3, 3)), lw=1, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Probability of a poor outcome (dashed lines = tier cut-offs {cuts['low']:.2f} / {cuts['high']:.2f})",
                      fontsize=7.5)
        charts.clean_axes(ax, grid_axis="x")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        changed = [(explain.LEVERS[f], explain._fmt(base[f]), explain._fmt(new[f]))
                   for f in explain.LEVERS if new[f] != base[f]]
        if changed:
            st.dataframe(pd.DataFrame(changed, columns=["Lever", "Before", "After"]), hide_index=True)
        else:
            st.caption("Move a lever on the left to see the effect.")
        st.info(f"**Action before:** {explain.suggested_action(bundle, base)}\n\n"
                f"**Action after:** {explain.suggested_action(bundle, new)}")

        if st.button("🎯 Which lever moves this group's score the most?", key=f"wi_btn_{sig}"):
            st.session_state["wi_best"] = sig
        if st.session_state.get("wi_best") == sig:
            moves = explain.best_moves(bundle, base, cand)
            top = moves.iloc[0]
            if top["Change (pts)"] < -0.05:
                st.success(f"**{top['Lever']}** moves this group's score the most: changing it from "
                           f"{top['Current']} to {top['Best realistic value']} alone changes the probability by "
                           f"{top['Change (pts)']:+.1f} points.")
            else:
                st.info("No single lever lowers this group's risk within the typical range.")
            st.dataframe(moves.style.format({"Probability now": "{:.1%}", "Probability if changed": "{:.1%}",
                                             "Change (pts)": "{:+.1f}"}), hide_index=True)
            st.caption("Each lever is changed on its own, to its most risk-reducing value inside the typical "
                       "range seen in the data (10th-90th percentile), starting from the original group.")

    st.warning("**The model shows association, not a guaranteed effect of the change.** Groups with more "
               "supervisor meetings or commits tend to do better, but forcing those numbers up does not by itself "
               "guarantee a better outcome.")


# ----------------------------------------------------------------------------- router
def main():
    inject_css()
    st.sidebar.markdown("<div class='pulse-brand'><div class='pulse-logo'>🎓</div><b>FYP-Pulse</b></div>"
                        "<div class='pulse-tag'>Final Year Project<br>Early Warning System</div>",
                        unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", PAGES, format_func=lambda p: f"{PAGE_ICONS[p]}  {p}")
    try:
        {"Data Quality": page_data_quality, "Dashboard": page_dashboard, "Models": page_models,
         "Predict": page_predict, "Fairness": page_fairness,
         "Watchlist": page_watchlist, "What-if": page_whatif}[page]()
    except FileNotFoundError as e:
        st.error(f"{e}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Something went wrong on this page: {e}")


main()