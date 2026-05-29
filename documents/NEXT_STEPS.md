# Next Steps — ICU Mortality Project

*Last updated: 2026-05-29*

---

## Where We Are

The core pipeline is complete and working:

- Feature extraction from eICU (2,520 patients) and MIMIC-III (136 patients)
- Full preprocessing pipeline (153 → 80 features) with no data leakage
- Baseline models trained: LR, RF, XGBoost, SVM
- Best model: **Random Forest + sigmoid calibration** (ROC-AUC 0.835 internal, 0.712 external)
- Four domain adaptation techniques implemented: importance weighting, label shift correction, MMD neural net, feature pruning

The open questions are methodological, not implementation. Below are the priorities in order.

---

## Priority 1 — Understand the MIMIC Generalization Gap

**The problem:** Our best MIMIC ROC-AUC is 0.712, and none of the four domain adaptation methods pushed it past 0.739 — a marginal gain for the effort involved. We don't know why.

**What we know:**
- The domain classifier (eICU vs MIMIC) achieves AUC = 0.999, confirming extreme covariate shift
- But importance weighting, which directly corrects for covariate shift, only improved MIMIC AUC by +0.025
- This suggests the gap is not primarily covariate shift

**What to investigate:**

1. **Feature-level distribution comparison** — for the top 10 SHAP features, plot their distributions in eICU train vs MIMIC. Look for features that are systematically shifted in ways that don't reflect real clinical differences (e.g., different lab reference ranges, different charting frequencies).

2. **Missing pattern analysis** — check whether missingness patterns differ between datasets. A feature that is always measured in eICU but only measured when abnormal in MIMIC encodes information differently even after imputation.

3. **Subgroup breakdown** — break down MIMIC performance by ICD-9 chapter, age group, and care unit. If one subgroup drives the gap, that points to a specific data issue.

4. **Label definition audit** — confirm that "in-hospital mortality" is extracted the same way from both datasets. Even a subtle difference (e.g., 30-day vs ICU-stay mortality) would explain a persistent gap that domain adaptation can't fix.

**Deliverable:** A single notebook (`05_gap_analysis.ipynb`) with the above four analyses and a written conclusion on root cause.

---

## Priority 2 — Fix the Decision Threshold for MIMIC

**The problem:** Our decision threshold (0.044) was tuned on eICU validation data with a 5% mortality base rate. MIMIC has a 34% base rate. Applying the same threshold to MIMIC is not valid.

**What to do:**

The label shift correction in `04_extensions.ipynb` already adjusts the predicted probabilities for the prior shift. We should use those adjusted probabilities to derive a MIMIC-appropriate threshold using the same specificity ≥ 80% constraint.

This is a one-cell change but it meaningfully changes the reported sensitivity/specificity on MIMIC — those numbers currently use threshold=0.5 as a workaround, which isn't principled.

**Deliverable:** Updated threshold logic in `04_extensions.ipynb`; updated results table in README.

---

## Priority 3 — Expand the External Validation Set

**The problem:** 136 MIMIC patients is a small external test set. Confidence intervals on ROC-AUC 0.712 at n=136 are wide (roughly ±0.08). Any claim about generalization needs more data.

**Options:**
- Use the full MIMIC-III dataset instead of the demo subset (requires PhysioNet credentialed access — we may already have it)
- Add a third dataset (e.g., AmsterdamUMCdb or HiRID) as a second external validation site

**Minimum viable step:** Run `00_data_collection_MIMIC.ipynb` on the full MIMIC-III tables. The extraction code already handles the full schema — it's just a matter of pointing it at the complete files.

**Deliverable:** Updated `mimic_features.csv` with full MIMIC cohort; re-run `03_baseline_model.ipynb` and `04_extensions.ipynb` on new external set.

---

## Priority 4 — Refactor Shared Utilities into `src/`

**The problem:** The helper functions `evaluate_model()`, `best_threshold()`, and `plot_roc_pr()` are copy-pasted across four notebooks. Any fix needs to be applied in four places.

**What to do:** Create `src/evaluation.py` with these three functions and import it in each notebook. No behavior change — just moving code.

```
src/
└── evaluation.py   # evaluate_model, best_threshold, plot_roc_pr
```

This is low-priority but becomes important if we add more notebooks (e.g., `05_gap_analysis.ipynb`).

**Deliverable:** `src/evaluation.py`; updated import cells in all four modeling notebooks.

---

## Priority 5 — Clean Up `01_data_exploration.ipynb`

**The problem:** The EDA notebook is >25 MB because output cells were never cleared. It's slow to open and can't be reviewed on GitHub.

**What to do:** Kernel → Restart & Clear Output, then re-run. No content changes needed.

**Deliverable:** File size back under 1 MB.

---

## Out of Scope (for now)

These came up during review but are not worth pursuing until Priority 1 is resolved:

- **Fairness audit** (performance by race/gender/age) — valid concern but needs a larger external set first
- **Clinical comparison to SOFA/APACHE** — would strengthen a paper but requires additional score extraction
- **Deployment / REST API** — premature until we understand the generalization ceiling

---

## Summary Table

| # | Task | Effort | Impact | Depends On |
|---|---|---|---|---|
| 1 | Understand MIMIC gap | High | High | — |
| 2 | Fix MIMIC threshold | Low | Medium | — |
| 3 | Expand external validation | Medium | High | PhysioNet access |
| 4 | Refactor `src/` utilities | Low | Low | — |
| 5 | Clean EDA notebook | Trivial | Low | — |
