# FYP-Pulse: Final Year Project Early Warning System

Streamlit app that estimates, using only information known at the **end of week 8**, which FYP groups are
likely to end with a poor outcome (failed / carried over), so support can be offered early.
Works fully offline from `data/fyp_pulse_train.csv` (no external data, no APIs).

## Run
```bash
pip install -r requirements.txt
python -m src.modeling        # trains + saves models/ (optional: the app trains on first start if missing)
streamlit run app.py
```
Trained artifacts are already included in `models/`, so `streamlit run app.py` works immediately.

## Structure
| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point, sidebar navigation, one render function per page |
| `src/cleaning.py` | A1 - cleaning rules, data-quality report |
| `src/charts.py` | A2 - dashboard charts + computed insights |
| `src/modeling.py` | A3/A4 - leakage control, 3 models, CV, thresholds, tier cut-offs, saving |
| `src/explain.py` | A5 - local explanations, support actions, fairness audit |
| `models/` | `final_model.joblib`, `metrics.json`, `test_scored.csv` |

## Key decisions
- **Target:** `poor_outcome`. Accuracy is not used for selection (always "No risk" = ~84% accuracy).
- **Dropped (leakage):** `final_defense_score`, `extension_granted` (the brief calls it `extension_grant`), `group_id`, raw `proposal_date`.
- **Week-8 safety:** `weeks_to_proposal_approval` is capped at 8 plus a "pending at week 8" flag, because longer waits are not known at week 8.
  `mid_eval_score` is assumed to be available by week 8 (remove it if your calendar differs).
- **Fairness:** `leader_gender` and `campus` are NOT model inputs; they are used only in the Fairness audit.
- **No leakage:** imputers, scalers, encoders, thresholds and tier cut-offs are fit on TRAIN only (inside the Pipeline / on out-of-fold predictions).
  The test set is used only for reporting (and to verify the tier rates rise Low < Medium < High).
- **Model selection** uses cross-validated PR-AUC (minus its std), and excludes models flagged as overfit.
- Cleaned CSV download is median/mode imputed over the whole file for convenience only; models do not use it.
- Seeds fixed to 42.
