<div align="center">

# 🎓 FYP-Pulse

### Final Year Project Early Warning System

*Spot the FYP groups that are likely to struggle **at the end of week 8**, while there is still time to help them.*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Math-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Charts-11557C)
![Offline](https://img.shields.io/badge/Works-100%25%20Offline-2ea44f)

</div>

---

## 📖 About

FYP-Pulse is a **Streamlit web app** that uses only information known at the **end of week 8** to estimate which Final Year Project groups are likely to end with a **poor outcome** (failed / carried over). Coordinators can then offer support early.

- 📂 Works **fully offline** from one CSV file (`data/fyp_pulse_train.csv`)
- 🚫 No external datasets, no paid APIs
- 🔒 Leakage-safe: nothing that is only known later in the project is used

---

## ✨ Features

| | Page | What it does |
|---|---|---|
| 🧹 | **Data Quality** | Cleans the raw CSV and reports every problem found and how it was fixed |
| 📊 | **Dashboard** | 4 charts (team, supervisor, project, engineered feature), each with a computed one-line insight |
| 🤖 | **Models** | Trains and compares 3 models, shows dropped features, over/underfitting check and confusion matrices |
| 🔮 | **Predict** | Enter one group's week-8 details and get the probability, a colour-coded risk tier and a plain-English explanation |
| ⚖️ | **Fairness** | Compares predicted rate and recall across leader gender and campus, with confidence intervals |
| 📋 | **Watchlist** *(bonus)* | Pick K (10-100) and download the K highest-risk groups with their top risk driver |
| 🎛️ | **What-if** *(bonus)* | Change only the levers the department can influence and watch the risk update live |

---

## 🛠️ Technologies Used

| Area | Technology | Used for |
|---|---|---|
| 🐍 Language | **Python 3** | Everything in the project |
| 🌐 Web app | **Streamlit** | Interface, sidebar navigation, forms, sliders, tables, downloads |
| 🐼 Data | **pandas** | Loading, cleaning and grouping the data |
| 🔢 Numbers | **NumPy** | Numeric operations and thresholds |
| 🧠 Machine learning | **scikit-learn** | Pipelines, imputation, scaling, one-hot encoding, models, cross-validation, metrics |
| 📈 Charts | **Matplotlib** | Dashboard charts, confusion matrices and explanation plots |
| 💾 Saving models | **joblib** | Saves and loads the trained model (`models/final_model.joblib`) |
| 🎨 Styling | **Custom CSS** + `.streamlit/config.toml` | Theme, dark sidebar, cards, pill tabs and risk badges |

### 🤖 Machine learning models

| Model | Role |
|---|---|
| **Logistic Regression** ⭐ | Selected as final model |
| Random Forest | Compared (flagged *Overfit*, so not eligible) |
| Gradient Boosting (`HistGradientBoostingClassifier`) | Compared (flagged *Overfit*, so not eligible) |

All three use **class-imbalance handling** (`class_weight`), **GridSearchCV** and **stratified 5-fold cross-validation**.

---

## 🔬 How It Works

```
CSV  ➜  🧹 Cleaning  ➜  ✂️ Stratified 80/20 split  ➜  🔧 Pipeline (impute ➜ scale ➜ encode)
     ➜  🤖 Train 3 models  ➜  🎯 Pick threshold + risk tiers (TRAIN only)  ➜  📊 Report on TEST
     ➜  🔮 Predict + 🧾 Explain + ⚖️ Fairness + 📋 Watchlist + 🎛️ What-if
```

1. **Cleaning** 🧹: standardises column names and spellings (`LHR` ➜ `Lahore`), fixes mixed date formats, removes duplicates, turns impossible values (e.g. CGPA 7.5) into missing, and enforces `team_min_cgpa ≤ team_avg_cgpa` (equal for solo groups).
2. **Leakage control** 🚫: columns not available at week 8 are dropped, and the reasons are shown inside the app.
3. **Feature engineering** 🧪: `cgpa_gap`, `commits_per_meeting` and `proposal_delay_flag`.
4. **Training** 🤖: three models with tuned hyperparameters; the final model is chosen by cross-validated PR-AUC (minus its std) and must not be overfit.
5. **Risk tiers** 🚦: 🟢 Low, 🟡 Medium, 🔴 High, with cut-offs chosen on train out-of-fold predictions.
6. **Explainability** 🧾: each feature is swapped for its typical value and the change in risk is measured, giving the top factors that raise or lower risk plus one concrete support action.

---

## 📊 Results (held-out test set: 360 groups)

| Metric | Logistic Regression ⭐ |
|---|---|
| Decision threshold | 0.41 |
| Recall | **81%** of at-risk groups caught |
| Precision | 30% |
| F1 | 0.44 |
| ROC-AUC | 0.81 |

> ❓ **Why not accuracy?** Only about 16% of groups have a poor outcome. A model that always says "No risk" scores about 84% accuracy but finds **0%** of the groups that need help, so models are judged on PR-AUC, recall and F1 instead.

| Risk tier | Groups | Observed poor-outcome rate |
|---|---|---|
| 🟢 Low | 236 | 6.8% |
| 🟡 Medium | 83 | 26.5% |
| 🔴 High | 41 | 48.8% |

The observed rate rises strictly **Low < Medium < High**, as required.

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) retrain the models. The trained files are already in models/
python -m src.modeling

# 4. Start the app
streamlit run app.py
```

The trained artifacts are included in `models/`, so `streamlit run app.py` works straight away. If they are missing, the app trains on first start (about a minute).

---

## 🗂️ Project Structure

```
├── app.py                  # Streamlit app: theme, sidebar and all pages
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme colours
├── data/
│   └── fyp_pulse_train.csv # The only data source
├── models/
│   ├── final_model.joblib  # Trained pipeline + threshold + tier cut-offs
│   ├── metrics.json        # Metrics for all models
│   └── test_scored.csv     # Test groups with predicted probabilities
└── src/
    ├── cleaning.py         # A1: cleaning rules and data-quality report
    ├── charts.py           # A2: dashboard charts, insights and chart styling
    ├── modeling.py         # A3/A4: leakage control, 3 models, CV, thresholds, tiers
    └── explain.py          # A5-A7: explanations, fairness audit, watchlist, what-if
```

---

## 🧠 Key Decisions

- 🎯 **Target:** `poor_outcome`. Accuracy is not used for model selection.
- 🚫 **Dropped (leakage):** `final_defense_score`, `extension_granted`, `group_id` and the raw `proposal_date`.
- ⏱️ **Week-8 safety:** `weeks_to_proposal_approval` is capped at 8 and a "pending at week 8" flag is added, because longer waits are not known at week 8. `mid_eval_score` is assumed to be available by week 8.
- ⚖️ **Fairness:** `leader_gender` and `campus` are **not** model inputs. They are used only in the Fairness audit.
- 🧪 **No data leakage:** imputers, scalers, encoders, thresholds and tier cut-offs are fit on TRAIN data only. The test set is used only for reporting.
- 🎲 **Reproducible:** random seeds fixed to 42.

---

## ⚠️ Notes

- The What-if simulator and the explanations show **association, not a guaranteed effect**. Changing a number does not by itself guarantee a better outcome.
- Fairness gaps on small groups have wide confidence intervals, so treat them as signals to investigate, not proof of bias.

---

<div align="center">

Made with ❤️ using Python and Streamlit

</div>
