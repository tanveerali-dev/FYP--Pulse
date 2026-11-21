"""A1 - Data cleaning and data-quality reporting for FYP-Pulse."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "fyp_pulse_train.csv"

TARGET = "poor_outcome"

# Valid (inclusive) ranges. Anything outside is an IMPOSSIBLE value -> NaN.
VALID_RANGES = {
    "group_size": (1, 5),
    "team_avg_cgpa": (0, 4),
    "team_min_cgpa": (0, 4),
    "supervisor_experience_years": (0, 45),
    "supervisor_active_groups": (1, 15),
    "proposal_revisions": (0, 10),
    "weeks_to_proposal_approval": (0, 30),
    "supervisor_meetings_8wk": (0, 30),
    "github_commits_8wk": (0, 1000),
    "mid_eval_score": (0, 100),
    "final_defense_score": (0, 100),
}
INTEGER_COLS = ["group_size", "supervisor_experience_years", "supervisor_active_groups",
                "proposal_revisions", "weeks_to_proposal_approval",
                "supervisor_meetings_8wk", "github_commits_8wk"]
CATEGORICAL_COLS = ["campus", "program", "leader_gender", "project_type",
                    "has_industry_sponsor", "extension_granted"]
NUMERIC_COLS = list(VALID_RANGES)

# Explicit spelling maps (built after inspecting the real unique values).
CAMPUS_MAP = {
    "lahore": "Lahore", "lhr": "Lahore",
    "karachi": "Karachi", "khi": "Karachi",
    "islamabad": "Islamabad", "isb": "Islamabad",
    "faisalabad": "Faisalabad", "fsd": "Faisalabad",
    "multan": "Multan",
    "rawalpindi": "Rawalpindi", "rwp": "Rawalpindi", "pindi": "Rawalpindi",
    "peshawar": "Peshawar", "pesh": "Peshawar",
    "quetta": "Quetta",
}
GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
YESNO_MAP = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
             "no": "No", "n": "No", "false": "No", "0": "No"}
PROGRAM_MAP = {p.lower(): p for p in ["BSCS", "BSSE", "BSAI", "BSIT", "BSDS"]}
PROJECT_MAP = {"web app": "Web App", "mobile app": "Mobile App", "ai-ml": "AI-ML",
               "iot": "IoT", "research": "Research"}
CAT_MAPS = {
    "campus": CAMPUS_MAP, "program": PROGRAM_MAP, "leader_gender": GENDER_MAP,
    "project_type": PROJECT_MAP, "has_industry_sponsor": YESNO_MAP,
    "extension_granted": YESNO_MAP, TARGET: YESNO_MAP,
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    # the brief calls it "extension_grant"; the CSV header is "extension_granted"
    return df.rename(columns={"extension_grant": "extension_granted"})


def parse_date(value) -> pd.Timestamp:
    """Mixed formats: YYYY-MM-DD, DD/MM/YYYY (day-first: 332 rows have day>12, none have month>12),
    DD-Mon-YY. Anything unparseable -> NaT."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d-%b-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place fyp_pulse_train.csv in the data/ folder.")
    return standardize_columns(pd.read_csv(path))


def clean_data(raw: pd.DataFrame):
    """Rule-based cleaning.

    Returns (cleaned_df, rules_table, per_column_table, summary).
    Missing values are deliberately NOT imputed here: modeling imputes inside the sklearn
    Pipeline (fit on TRAIN only). `impute_for_report` makes the fully-imputed copy that is
    used only for the downloadable cleaned CSV.
    """
    df = standardize_columns(raw)
    n0 = len(df)
    rules: list[tuple] = []
    fixed: dict[str, int] = {}

    def bump(col, n):
        fixed[col] = fixed.get(col, 0) + int(n)

    # 1. strings: strip + categorical normalisation
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    spelling_total = 0
    for col, mapping in CAT_MAPS.items():
        if col not in df.columns:
            continue
        before = df[col].copy()
        mapped = before.map(
            lambda v: mapping.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan)
        unknown = before.notna() & mapped.isna()
        changed = before.notna() & mapped.notna() & (before != mapped)
        df[col] = mapped
        bump(col, changed.sum() + unknown.sum())
        spelling_total += int(changed.sum())
        if unknown.sum():
            rules.append((f"Unknown category in {col}", int(unknown.sum()),
                          "Set to missing (mode-imputed in pipeline)"))
    rules.insert(0, ("Inconsistent spelling / case / abbreviations (campus, gender, Yes/No ...)",
                     spelling_total,
                     "Mapped with explicit dictionaries (e.g. LHR/lahore -> Lahore, M -> Male)"))

    # 2. dates
    parsed = df["proposal_date"].map(parse_date)
    n_bad_date = int(parsed.isna().sum() - df["proposal_date"].isna().sum())
    iso = parsed.dt.strftime("%Y-%m-%d")
    n_reformatted = int((df["proposal_date"].notna() & parsed.notna()
                         & (df["proposal_date"].astype(str) != iso)).sum())
    df["proposal_date"] = iso
    bump("proposal_date", n_reformatted + max(n_bad_date, 0))
    rules.append(("Mixed date formats (DD/MM/YYYY, DD-Mon-YY)", n_reformatted,
                  "Converted to YYYY-MM-DD"))
    rules.append(("Unparseable dates", max(n_bad_date, 0), "Set to missing"))

    # 3. target: drop missing
    miss_t = int(df[TARGET].isna().sum())
    df = df[df[TARGET].notna()].copy()
    rules.append(("Missing target (poor_outcome)", miss_t, "Row dropped (never imputed)"))

    # 4. duplicates (after normalisation so near-identical rows are caught)
    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="first")
    rules.append(("Exact duplicate rows", n_exact, "Removed, first occurrence kept"))
    df = df.assign(_c=df.notna().sum(axis=1)).sort_values("_c", ascending=False, kind="stable")
    n_id = int(df["group_id"].duplicated().sum())
    df = df.drop_duplicates("group_id", keep="first").drop(columns="_c").sort_index()
    rules.append(("Duplicate group_id (non-identical rows)", n_id,
                  "Removed, most complete row kept"))

    # 5. impossible values -> NaN
    for col, (lo, hi) in VALID_RANGES.items():
        v = pd.to_numeric(df[col], errors="coerce")
        nonnum = int((df[col].notna() & v.isna()).sum())
        bad = (v < lo) | (v > hi)
        n_bad = int(bad.sum()) + nonnum
        v[bad] = np.nan
        df[col] = v
        if n_bad:
            bump(col, n_bad)
        rules.append((f"Impossible {col} (outside {lo}-{hi})", n_bad,
                      "Set to missing, then imputed"))
    for c in INTEGER_COLS:
        df[c] = df[c].round()

    # 6. CGPA consistency
    bad_order = (df["team_min_cgpa"] > df["team_avg_cgpa"]).fillna(False)
    df.loc[bad_order, "team_min_cgpa"] = df.loc[bad_order, "team_avg_cgpa"]
    bump("team_min_cgpa", bad_order.sum())
    rules.append(("team_min_cgpa > team_avg_cgpa", int(bad_order.sum()),
                  "Set min = avg. The average uses every member, so it is the safer anchor; "
                  "min = avg is the neutral choice (gap 0) and never invents a weaker member"))
    solo = df["group_size"].eq(1)
    need = solo & df["team_avg_cgpa"].notna() & df["team_min_cgpa"].notna() \
        & (df["team_min_cgpa"] != df["team_avg_cgpa"])
    need_fill = solo & df["team_avg_cgpa"].notna() & df["team_min_cgpa"].isna()
    df.loc[need | need_fill, "team_min_cgpa"] = df.loc[need | need_fill, "team_avg_cgpa"]
    fill_avg = solo & df["team_avg_cgpa"].isna() & df["team_min_cgpa"].notna()
    df.loc[fill_avg, "team_avg_cgpa"] = df.loc[fill_avg, "team_min_cgpa"]
    bump("team_min_cgpa", need.sum() + need_fill.sum())
    bump("team_avg_cgpa", fill_avg.sum())
    rules.append(("group_size = 1 but min CGPA != avg CGPA (or one of them missing)",
                  int(need.sum() + need_fill.sum() + fill_avg.sum()),
                  "min = avg (a one-person group has a single CGPA)"))

    df = df.reset_index(drop=True)

    # missing values that remain (imputed later, inside the Pipeline for modeling)
    miss = df.drop(columns=[TARGET]).isna().sum()
    for col, n in miss[miss > 0].items():
        rules.append((f"Missing value in {col}", int(n),
                      "Median imputation (fit on TRAIN only)" if col in NUMERIC_COLS
                      else "Mode imputation (fit on TRAIN only)"))

    rules_df = pd.DataFrame(rules, columns=["Rule", "Rows affected", "Fix applied"])
    per_col = pd.DataFrame({"Values corrected": pd.Series(fixed),
                            "Missing (imputed)": df.drop(columns=[TARGET]).isna().sum()}
                           ).fillna(0).astype(int)
    per_col = per_col[(per_col > 0).any(axis=1)]
    per_col["Total touched"] = per_col.sum(axis=1)
    per_col = per_col.sort_values("Total touched", ascending=False)
    n_poor = int((df[TARGET] == "Yes").sum())
    summary = {"initial_rows": n0, "cleaned_rows": len(df), "rows_removed": n0 - len(df),
               "total_poor": n_poor, "poor_rate": n_poor / max(len(df), 1),
               "exact_dups": n_exact, "id_dups": n_id,
               "values_corrected": int(sum(fixed.values()))}
    return df, rules_df, per_col, summary


def impute_for_report(df: pd.DataFrame) -> pd.DataFrame:
    """Fully imputed copy (median / mode over the whole file) - ONLY for the downloadable
    cleaned CSV. Modeling never uses this; it imputes inside the Pipeline on TRAIN data."""
    out = df.copy()
    for c in NUMERIC_COLS:
        out[c] = out[c].fillna(out[c].median())
    for c in INTEGER_COLS:
        out[c] = out[c].round().astype(int)
    for c in CATEGORICAL_COLS:
        out[c] = out[c].fillna(out[c].mode().iloc[0])
    solo = out["group_size"] == 1
    bad = (out["team_min_cgpa"] > out["team_avg_cgpa"]) | solo
    out.loc[bad, "team_min_cgpa"] = out.loc[bad, "team_avg_cgpa"]
    out["proposal_date"] = out["proposal_date"].fillna(out["proposal_date"].mode().iloc[0])
    return out


def get_clean_for_modeling(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean_data(load_raw(path))[0]


if __name__ == "__main__":
    d, r, p, s = clean_data(load_raw())
    print(s)
    print(r.to_string())
    print(p)
