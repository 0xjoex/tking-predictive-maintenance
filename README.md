# Th King — Predictive Maintenance Project

A synthetic-data project simulating a predictive maintenance pipeline for refrigeration
unit telematics, built as prep. Covers the full pipeline: data generation → feature engineering →
model training → evaluation → SQL validation → Tableau dashboarding.

## Problem

Given daily sensor readings from a refrigeration unit (vibration, coolant pressure,
compressor temperature, cargo temperature deviation), predict whether that unit will
fail within the next 7 days — enabling proactive maintenance before a breakdown
strands a shipment.

## Pipeline

1. **Synthetic data generation** — simulated a fleet of 400 refrigeration units over
   up to 500 days each, with ~18% experiencing a failure event. Sensor readings
   gradually degrade in the 60 days before a unit's failure (mirroring real
   mechanical degradation), rather than jumping instantly from healthy to failed.

2. **Feature engineering** — computed 7-day rolling mean, rolling std, and trend
   (value now vs. 7 days ago) per sensor, per unit, to capture degradation
   trajectories rather than noisy point-in-time readings.

3. **Model training** — `RandomForestClassifier` with `class_weight="balanced"` to
   handle the severe class imbalance (~0.26% positive rate — failures are rare
   events). Train/test split performed by **unit_id** (`GroupShuffleSplit`), not by
   row, to avoid data leakage from correlated consecutive days of the same unit.

4. **Evaluation** — precision/recall/ROC-AUC over accuracy (a model predicting
   "no failure" for everything would still score ~99.7% accuracy while being
   useless). Achieved 97% recall / 42% precision on the failure class — prioritizing
   recall, since missing a real failure (stranded cargo, safety risk) is far costlier
   than an unnecessary maintenance check.

5. **SQL validation** — loaded predictions into a local SQLite database and wrote
   aggregation queries (`GROUP BY unit_id`, `HAVING`) to independently verify model
   behavior: confirmed units. Confirmed genuine failures were flagged with peak risk
   scores near 1.0, and genuinely healthy units stayed at a risk score of exactly 0.0.

6. **Tableau dashboard** — built an interactive dashboard with a fleet risk
   leaderboard (units ranked by peak failure risk) and a per-unit risk trajectory
   chart, wired together with a click-to-filter interaction.

## Screenshots

<img width="717" height="425" alt="Screenshot 2026-07-09 at 23 50 27" src="https://github.com/user-attachments/assets/cf17a5d8-1b0f-4427-a0d7-ebd49d5af309" />
<img width="717" height="425" alt="Screenshot 2026-07-09 at 23 50 27" src="https://github.com/user-attachments/assets/15632c0f-0af9-4560-8968-9f2c04caf371" />



## Repo structure

```
thermoking_predictive_maintenance.ipynb   # main pipeline: data gen -> features -> model -> SQL
predict_new_units.ipynb                    # scoring template for new/unseen unit data
new_unseen_units.csv                       # sample unseen data for scoring demo
rf_model.joblib                            # saved trained model
thermoking.db                              # SQLite database with predictions table
tableau_unit_summary.csv                   # per-unit aggregated view (Tableau leaderboard)
tableau_daily_predictions.csv              # per-day detail view (Tableau trajectory chart)
screenshots/                               # dashboard + chart exports
```

## Key design decisions

- **Group-based train/test split** — prevents leakage from correlated consecutive
  days of the same physical unit.
- **class_weight="balanced"** — compensates for the ~0.26% positive rate so the
  model doesn't just learn to always predict "healthy."
- **Recall-first evaluation** — reflects the real business cost asymmetry between
  missed failures and false alarms.
- **Rolling window features over raw readings** — captures degradation trends,
  since a single noisy reading isn't a reliable failure signal on its own.

## Caveats

Results here (ROC-AUC ~0.999) are on synthetic data with a clean, deliberately
simulated degradation signal — real sensor data would be noisier, with more
confounding factors (route, weather, cargo load), so I'd expect meaningfully lower
performance in production. The pipeline and validation approach (group-based
splitting, imbalance handling, recall-first evaluation) would carry over directly.

## Stack

Python, pandas, scikit-learn (RandomForestClassifier, GroupShuffleSplit), SQLite,
Tableau Public.
