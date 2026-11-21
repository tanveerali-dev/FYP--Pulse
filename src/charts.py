"""A2 - dashboard charts + data-driven insight sentences (matplotlib, offline)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.cleaning import TARGET
from src.modeling import RAW_FEATURES, engineer_features

# ----------------------------------------------------------------------------- design tokens (shared with app.py)
INK, MUTED, GRID = "#1b1f3b", "#6b7280", "#e5e7f0"
PRIMARY, PRIMARY_SOFT = "#4f46e5", "#c7d2fe"
BLUE, RED, GREY = PRIMARY, "#e11d48", MUTED          # BLUE/RED/GREY kept for backward compatibility
GOOD, WARN = "#16a34a", "#f59e0b"
CMAP = LinearSegmentedColormap.from_list("pulse", ["#ffffff", "#c7d2fe", "#4f46e5"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "DejaVu Sans", "Arial"],
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def clean_axes(ax, grid_axis: str = "y"):
    """Minimal chart chrome: no top/right/left spines, light grid on one axis only."""
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8)
    ax.tick_params(length=0)


def with_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eng = engineer_features(df[RAW_FEATURES])
    for c in ["cgpa_gap", "commits_per_meeting", "proposal_delay_flag"]:
        out[c] = eng[c].values
    out["y"] = (out[TARGET] == "Yes").astype(int)
    return out


def rate_table(d: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    t = d.groupby(group, observed=True)["y"].agg(rate="mean", n="size", poor="sum").reset_index()
    t.columns = ["group", "rate", "n", "poor"]
    t["group"] = t["group"].astype(str)
    return t


def bar_rate(t: pd.DataFrame, title: str, xlabel: str, overall: float):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    top = t["rate"].max()
    colors = [PRIMARY if r == top else PRIMARY_SOFT for r in t["rate"]]   # highest-risk bar stands out
    bars = ax.bar(range(len(t)), t["rate"] * 100, color=colors, width=0.62, zorder=2)
    ax.set_xticks(range(len(t)), [f"{g}\n(n={n})" for g, n in zip(t["group"], t["n"])])
    for b, r in zip(bars, t["rate"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{r*100:.0f}%", ha="center", va="bottom",
                fontsize=8.5, weight="bold", color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.axhline(overall * 100, color=RED, ls=(0, (4, 3)), lw=1.2, label=f"overall {overall*100:.1f}%", zorder=3)
    ax.set_title(title, fontsize=10.5, pad=12)
    ax.set_xlabel(xlabel, fontsize=8, labelpad=8)
    ax.set_ylabel("Poor-outcome rate (%)", fontsize=8)
    ax.set_ylim(0, max(top * 100 * 1.4, 5))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    clean_axes(ax)
    fig.tight_layout()
    return fig


def _ratio_text(hi, lo):
    return f"{hi/lo:.1f}x" if lo > 0 else "far higher"


def dashboard_charts(df: pd.DataFrame):
    """Returns list of dicts: {key, fig, insight, title}. Insights are computed, not hardcoded."""
    d = with_features(df)
    overall = d["y"].mean()
    out = []

    # 1. TEAM: team_avg_cgpa quintile bands
    q = pd.qcut(d["team_avg_cgpa"], 5, duplicates="drop")
    t = rate_table(d.dropna(subset=["team_avg_cgpa"]), q.dropna())
    t["group"] = [f"{i.left:.2f}-{i.right:.2f}" for i in sorted(q.dropna().unique())]
    lo, hi = t.iloc[0], t.iloc[-1]
    out.append({"key": "team", "title": "Team factor: team average CGPA",
                "fig": bar_rate(t, "Poor-outcome rate by team average CGPA", "team_avg_cgpa band", overall),
                "insight": f"Groups with the lowest average CGPA ({lo['group']}) end badly {lo['rate']*100:.0f}% of the time "
                           f"versus {hi['rate']*100:.0f}% for the highest band ({hi['group']}) - {_ratio_text(lo['rate'], hi['rate'])} the risk."})

    # 2. SUPERVISOR: meetings in 8 weeks
    bins, labels = [-0.1, 2, 4, 6, 9, 30], ["0-2", "3-4", "5-6", "7-9", "10+"]
    sub = d.dropna(subset=["supervisor_meetings_8wk"])
    t = rate_table(sub, pd.cut(sub["supervisor_meetings_8wk"], bins, labels=labels))
    lo, hi = t.iloc[0], t.iloc[-1]
    out.append({"key": "supervisor", "title": "Supervisor factor: meetings in the first 8 weeks",
                "fig": bar_rate(t, "Poor-outcome rate by supervisor meetings (8 wk)", "meetings in 8 weeks", overall),
                "insight": f"Groups meeting their supervisor 0-2 times fail {lo['rate']*100:.0f}% of the time compared with "
                           f"{hi['rate']*100:.0f}% for groups with 10+ meetings."})

    # 3. PROJECT CONTEXT: project type
    t = rate_table(d.dropna(subset=["project_type"]), d["project_type"].dropna()).sort_values("rate", ascending=False)
    top, bot = t.iloc[0], t.iloc[-1]
    out.append({"key": "project", "title": "Project context: project type",
                "fig": bar_rate(t, "Poor-outcome rate by project type", "project_type", overall),
                "insight": f"{top['group']} projects are the riskiest ({top['rate']*100:.0f}% poor outcomes) and "
                           f"{bot['group']} the safest ({bot['rate']*100:.0f}%)."})

    # 4. ENGINEERED: cgpa_gap
    sub = d.dropna(subset=["cgpa_gap"])
    q = pd.qcut(sub["cgpa_gap"], 5, duplicates="drop")
    cats = sorted(q.unique())
    t = rate_table(sub, q)
    t["group"] = [f"{max(c.left, 0):.2f}-{c.right:.2f}" for c in cats]
    lo, hi = t.iloc[0], t.iloc[-1]
    out.append({"key": "engineered", "title": "Engineered feature: cgpa_gap",
                "fig": bar_rate(t, "Poor-outcome rate by CGPA gap (avg - min)", "cgpa_gap band", overall),
                "insight": f"When the weakest member sits far below the team average (gap {hi['group']}) the poor-outcome rate is "
                           f"{hi['rate']*100:.0f}% versus {lo['rate']*100:.0f}% for tightly matched teams (gap {lo['group']})."})
    return out


def confusion_figure(cm, title: str, size=(3.8, 3.4)):
    """Confusion matrix with counts, row-% and TN/FP/FN/TP labels. cm = [[TN, FP], [FN, TP]]."""
    cm = np.array(cm)
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    names = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(cm, cmap=CMAP)
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.6
            ax.text(j, i - 0.12, str(cm[i, j]), ha="center", va="center", fontsize=16, weight="bold",
                    color="white" if dark else INK)
            ax.text(j, i + 0.2, f"{names[i][j]} · {row_pct[i, j]:.0f}% of actual",
                    ha="center", va="center", fontsize=7, color="#e0e7ff" if dark else MUTED)
    ax.set_xticks([0, 1], ["No", "Yes"])
    ax.set_yticks([0, 1], ["No", "Yes"])
    ax.set_xlabel("Predicted poor outcome", fontsize=8)
    ax.set_ylabel("Actual poor outcome", fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    return fig 