# ICU Mortality Prediction with Cross-Hospital Generalization

A machine learning pipeline for predicting in-hospital ICU mortality, trained on the eICU Collaborative Research Database and externally validated on MIMIC-III. The central challenge this project addresses is **cross-institutional generalization**: the two datasets come from different hospital systems with a 6.7× difference in baseline mortality rate (5% vs 34%).

---

## Results

### eICU (internal) — threshold 0.044

| Split | N | Mortality | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Validation | 504 | 5% | 0.853 | 0.408 | 76% | 81% |
| Test | 504 | 5% | 0.827 | 0.398 | 72% | 76% |

### MIMIC-III (external) — label-shift-corrected threshold 0.30

| Split | N | Mortality | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| External Test | 136 | 34% | **0.734** | 0.611 | **70%** | **69%** |

**Best model:** Random Forest with sigmoid calibration (`models/rf_calibrated.pkl`).

Two thresholds are used depending on deployment context — see [Inference](#inference) section.

---

## Project Structure

```
.
├── notebooks/
│   ├── 00_data_collection_eICU.ipynb      # Feature extraction from eICU
│   ├── 00_data_collection_MIMIC.ipynb     # Feature extraction from MIMIC-III
│   ├── 01_data_exploration.ipynb          # EDA: distributions, missingness, class imbalance
│   ├── 02_data_preprocessing.ipynb        # Full preprocessing pipeline (153 → 80 features)
│   ├── 03_baseline_model.ipynb            # LR, RF, XGBoost training + evaluation
│   ├── 04_model_extensions.ipynb          # Domain adaptation: IW, label shift, MMD
│   ├── 05_gap_analysis.ipynb              # MIMIC generalization gap root-cause analysis
│   └── 06_results_comparison.ipynb        # Full visual comparison: ROC/PR, calibration, SHAP, subgroups
├── data/
│   ├── eicu_raw/                          # Raw eICU database tables
│   ├── mimic_3_raw/                       # Raw MIMIC-III database tables
│   └── output_data/
│       ├── eicu_train/                    # eicu_features.csv (2,520 × 73)
│       ├── mimic_val/                     # mimic_features.csv (136 × 74)
│       └── preprocessed/                  # Final modeling splits (80 features each)
│           ├── X_train.csv / y_train.csv  # 1,512 patients
│           ├── X_val.csv / y_val.csv      # 504 patients
│           ├── X_test.csv / y_test.csv    # 504 patients
│           ├── X_mimic.csv / y_mimic.csv  # 136 patients (external)
│           ├── X_train_smote.csv          # SMOTE-resampled training set (2,872 × 80)
│           └── preprocessing_objects.pkl  # Scaler, imputer, clip bounds
├── models/
│   ├── rf_calibrated.pkl                  # Best model (5.8 MB)
│   ├── rf_final.pkl                       # Uncalibrated RF for reference
│   ├── threshold_config.pkl               # Threshold parameters for both contexts
│   ├── model_params.json                  # RF best params + thresholds (used by 04)
│   └── extensions/
│       └── extension_models.pkl           # IW model, MMD state, da_scaler (saved by 04)
├── documents/
│   └── NEXT_STEPS.md                      # Prioritised next steps
└── src/                                   # (reserved for shared utilities)
```

---

## Datasets

| Dataset | Source | ICU Stays | Features | Mortality |
|---|---|---|---|---|
| eICU | eICU Collaborative Research Database v2.0 | 2,520 | 73 | ~5% |
| MIMIC-III | MIMIC-III Clinical Database (demo) | 136 | 74 | ~34% |

Both datasets require credentialed access via PhysioNet. The raw tables are not included in this repository.

Features cover **demographics**, **first-24h lab values** (min/max per analyte), **vital signs** (min/mean/max), **admission source**, **care unit type**, and **ICD-9 diagnosis chapter**. Feature sets were harmonized across datasets before modeling.

---

## Preprocessing Pipeline

Implemented in `02_data_preprocessing.ipynb`. Key steps in order:

1. **Column harmonization** — eICU vital sign names aligned to MIMIC convention
2. **Leakage removal** — `stay_id`, `dataset`, `icu_los_days` dropped (LOS only known at discharge)
3. **High-missingness drop** — temperature features removed (>94% missing in both datasets)
4. **Redundant feature removal** — 14 `_mean` lab aggregates dropped (r > 0.99 with min/max); vitals min/mean/max retained (not correlated)
5. **Train/val/test split** — 60/20/20 stratified on mortality; MIMIC held out entirely
6. **Outlier capping** — winsorization at [1st, 99th] percentile, fit on train only
7. **Missingness indicators** — 7 binary flags for clinically important but frequently missing features (albumin, lactate, ALT, SBP, DBP, MAP, INR)
8. **Imputation** — median (numeric), fit on eICU training set only
9. **One-hot encoding** — care unit, admission source, ethnicity group, ICD-9 chapter; `drop_first=True`
10. **Scaling** — StandardScaler on 47 continuous features only; binary flags and OHE dummies not scaled
11. **SMOTE** — applied inside CV folds only, never pre-applied to prevent leakage
12. **Feature stability filtering** — 6 features removed based on KS test + Cohen's d + correlation delta across train/val

Final shape: **80 features** per patient.

---

## Models

### Baseline (`03_baseline_model.ipynb`)

| Model | Val ROC-AUC | Test ROC-AUC | MIMIC ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.741 | 0.809 | — |
| XGBoost | 0.810 | — | — |
| Random Forest (uncalibrated) | 0.843 | 0.833 | 0.681 |
| **Random Forest (calibrated)** | **0.853** | **0.827** | **0.734** |

RF hyperparameters (GridSearch): `n_estimators=200`, `max_depth=20`, `min_samples_leaf=5`.  
Calibration: sigmoid, 5-fold cross-validation on training set only.

### Domain Adaptation Extensions (`04_model_extensions.ipynb`)

| Method | MIMIC ROC-AUC | MIMIC Sensitivity | MIMIC Specificity | Notes |
|---|---|---|---|---|
| Baseline RF (calibrated) | 0.735 | 65% | 70% | Reference — threshold 0.044 |
| Importance Weighting (IW) | **0.747** | 59% | 77% | Domain classifier AUC = 0.984 — extreme shift detected |
| Label Shift Correction (LS) | 0.735 | **70%** | 70% | ROC unchanged by design; gain is in sensitivity |
| MMD Neural Network | 0.611 | 48% | 78% | Underperforms — too few target samples (136 patients) |

IW gives the best AUC (+0.012). Label Shift gives the best sensitivity (+5pp) with no retraining, using threshold 0.30 on log-odds-corrected probabilities. MMD degraded performance — insufficient target data for the domain alignment to converge. The root-cause analysis (`05_gap_analysis.ipynb`) explains why all methods plateau — see below.

---

## Inference

Two deployment contexts require different thresholds.

### eICU-like context (5% base rate)

```python
import pickle, pandas as pd

with open("models/rf_calibrated.pkl", "rb") as f:
    model = pickle.load(f)

X = pd.read_csv("data/output_data/preprocessed/X_val.csv")
probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.044).astype(int)
```

### MIMIC-like context (different base rate — use label shift correction)

```python
import pickle, numpy as np, pandas as pd

with open("models/rf_calibrated.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/threshold_config.pkl", "rb") as f:
    cfg = pickle.load(f)

X = pd.read_csv("data/output_data/preprocessed/X_mimic.csv")
probs = model.predict_proba(X)[:, 1]

# Adjust for base-rate shift (5% eICU → 34% MIMIC)
log_odds = np.log(np.clip(probs, 1e-9, 1-1e-9) / (1 - np.clip(probs, 1e-9, 1-1e-9)))
probs_adj = 1 / (1 + np.exp(-(log_odds + cfg["log_odds_shift"])))

preds = (probs_adj >= cfg["mimic_threshold"]).astype(int)  # threshold = 0.30
# Sensitivity 70%, Specificity 69% on MIMIC external test
```

The label shift correction adjusts the model's log-odds by the difference in log prior odds between training (5%) and deployment (34%). Threshold 0.30 on the corrected probabilities was chosen to maximise specificity subject to sensitivity ≥ 75%, tuned on the eICU validation set.

---

## MIMIC Generalization Gap — Key Findings

Full analysis in `05_gap_analysis.ipynb`. Summary:

**The 9-point AUC gap (0.827 eICU test → 0.734 MIMIC) has four root causes:**

1. **Missingness pattern mismatch** — SBP/DBP/MAP were 84% missing in eICU training data, so the model was trained on mostly-constant imputed values for those features and learned near-zero weights for them. In MIMIC, where BP is measured on 97% of patients, those features carry real signal the model cannot use. INR shows the same problem (74% missing in eICU, 20% in MIMIC).

2. **Case mix shift** — MIMIC has 8× more SICU patients (25% vs 3%), 17× more cancer cases (12% vs 0.7%), and 5× more infectious disease cases (19% vs 4%). These groups are underrepresented in training. Subgroup AUCs: neoplasms 0.691, SICU 0.711.

3. **Age is not predictive in MIMIC** — age is the model's 4th most important feature but does not significantly separate survivors from deaths in MIMIC (t-test p = 0.64 vs p < 0.0001 in eICU). The model over-relies on a feature that does not generalise.

4. **Young patients (<45) fail completely** — AUC 0.425 on 13 patients. Young ICU deaths have different pathophysiology (trauma, sepsis at atypical age) not represented in the elderly-skewed eICU training set.

**Experiments that did not help:**
- Dataset-specific MIMIC imputation: −0.004 AUC (the problem is model weights, not test imputation)
- Retraining without BP/INR features: sensitivity dropped from 63% → 50% on MIMIC (those features still carry some inference-time signal despite low importance)

---

## Reproducing the Results

```bash
# 1. Feature extraction
jupyter notebook notebooks/00_data_collection_eICU.ipynb
jupyter notebook notebooks/00_data_collection_MIMIC.ipynb

# 2. EDA
jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Preprocessing → data/output_data/preprocessed/
jupyter notebook notebooks/02_data_preprocessing.ipynb

# 4. Baseline models
jupyter notebook notebooks/03_baseline_model.ipynb

# 5. Domain adaptation
jupyter notebook notebooks/04_model_extensions.ipynb

# 6. Gap analysis
jupyter notebook notebooks/05_gap_analysis.ipynb

# 7. Full results comparison (ROC/PR, calibration, SHAP, subgroups)
jupyter notebook notebooks/06_results_comparison.ipynb
```

---

## Dependencies

```
python >= 3.8
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
torch
matplotlib
scipy
```

---

## Limitations

- **External validation set is small** (136 patients from MIMIC-III demo). Results on full MIMIC-III would have tighter confidence intervals.
- **BP/INR features are unreliable in training** — 74–84% missing in eICU means the model never learned to rely on blood pressure or INR. A model trained on data with complete BP coverage would likely generalise better.
- **Threshold is context-dependent** — the 0.044 threshold applies to eICU-distribution data; a different deployment context requires re-deriving the threshold or applying the label shift correction with the appropriate target prevalence.
- **Young patients (<45) are not well served** — AUC 0.425 on this subgroup. Do not use this model for young ICU patients without further validation.
- **No prospective validation.** All results are retrospective.

---

## License

Raw datasets (eICU, MIMIC-III) are subject to PhysioNet credentialed access agreements and may not be redistributed.
