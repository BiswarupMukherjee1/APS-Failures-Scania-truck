# Predicting Air Pressure System Failures in Scania Trucks

Project for *Data Mining and Machine Learning* MSc IPCVai 25-27.
Author: Biswarup Mukherjee

The Air Pressure System (APS) on a heavy truck supplies compressed air to the brakes and the gear shift. The task is to
decide, from one snapshot of anonymised on-board readouts, whether a truck that came into the workshop has a failure
in the APS or a failure somewhere else. The two kinds of mistake do not cost the same: sending a healthy APS for
inspection costs a mechanic's time, while missing a faulty APS sends a truck back on the road with a braking
component that is about to fail.

| | |
|---|---|
| Final model | Stacking ensemble: Ridge classifier + CatBoost, logistic-regression meta-learner |
| Recall on failures (out-of-fold, 60,000 trucks) | **0.953** (1,035 of 1,086 failures caught, 51 missed) |
| False-alarm rate | 3.1 % (1,816 of 58,914 non-APS trucks flagged) |
| Balanced accuracy | **0.961** |
| ROC-AUC / PR-AUC | 0.990 / 0.768 |
| Cost with the challenge's weights (10 per false alarm, 500 per miss) | 43,660, against 543,000 for "never flag" |

All numbers above come from 5-fold out-of-fold predictions on the training set at a decision threshold of 0.65.
The 16,000-truck test set had hidden labels (course Kaggle competition); predictions are in `submission_final.csv`.

---

## 1. Data

The data is the Scania *APS Failure at Scania Trucks* dataset, released for the IDA 2016 Industrial Challenge and
redistributed for the course as an in-class Kaggle competition.

- **60,000 training trucks, 170 numeric features**, no duplicate rows.
- Features are anonymised. Some are single counters (`aa_000`), others are bins of the same histogram
  (`ag_000` … `ag_009`), for example the time spent in different pressure ranges.
- **Labels:** `pos` = the failure was in the APS, `neg` = the failure was in another component.
  1,086 positives, **1.81 %** of the trucks (about 1 in 55).

## 2. What the data looked like

Each finding below changed a design decision later in the pipeline.

**Class imbalance.** A model that always answers `neg` is 98.2 % accurate and useless. Accuracy was therefore never
used. Model selection used balanced accuracy (mean of recall on each class), and recall on failures was tracked
separately.

**Missing values are structured, not random.** 8.33 % of all cells are missing (849,509 cells). A handful of
features are missing for most trucks (`br_000` 82.1 %, `bq_000` 81.2 %, `bp_000` 79.5 %), and they go missing
together, in bands, which points to sensor groups that only exist on some truck configurations
(`figures/missingness_pattern.png`).

To check whether the missingness itself is informative, I compared the failure rate when a feature is recorded with
the rate when it is missing:

| Feature | P(APS failure \| recorded) | P(APS failure \| missing) | Ratio |
|---|---|---|---|
| `br_000` | 9.24 % | 0.19 % | ≈ 49× |
| `bq_000` | 8.85 % | 0.18 % | ≈ 49× |
| `bp_000` | 8.15 % | 0.17 % | ≈ 48× |
| `cc_000` | 1.86 % | 0.98 % | ≈ 1.9× |

Trucks on which these sensors report a value are about 49 times more likely to have an APS failure. Imputing the gap
and moving on would throw that away, so every feature with more than 5 % missing values got a binary
`*_is_missing` flag.

**Heavy tails.** Many counters are dominated by a few extreme trucks. `cf_000` has a skewness of 244.9. After
`log1p`, it drops to 3.5 (`co_000`: 244.9 → 0.9, `ad_000`: 244.9 → −0.6).

**Redundant histogram bins.** Bins of the same histogram are strongly correlated with each other, and a 2-D PCA
projection shows the two classes overlapping, so no single linear direction separates them.

## 3. Pipeline

| Step | What | Why |
|---|---|---|
| 1 | Constant-feature check (`VarianceThreshold(0)`) | None found; kept as a guard |
| 2 | Missingness flags for features with > 5 % missing | Section 2: presence of a sensor carries risk |
| 3 | Median imputation | The mean is pulled by the heavy tails |
| 4 | `log1p` on all numeric features | Skewness from ~245 to single digits |
| 5 | Histogram totals for 7 histogram families (`ag`, `ay`, `az`, `ba`, `cn`, `cs`, `ee`) | One dense "total activity" value per histogram |
| 6 | Distance to each of 5 K-Means centroids (on standardised features) | Lets every model see which operating profile a truck belongs to |
| 7 | Drop one of every pair of features with \|Pearson r\| > 0.95 | Removes duplicated bins |

Result: 60,000 × 142 features.

## 4. Model selection

Nine model families were tuned with **Optuna** (TPE sampler, seed 42, median pruner, 10–20 trials per model).
Each trial was scored by **balanced accuracy averaged over 3 stratified folds**. Every model used class weights
(`balanced`, or `scale_pos_weight` for XGBoost) instead of resampling.

| Model | Best balanced accuracy (3-fold) |
|---|---|
| CatBoost | **0.9607** |
| XGBoost | 0.9572 |
| LightGBM | 0.9563 |
| Ridge classifier | 0.9563 |
| Logistic regression | 0.9554 |
| Random forest | 0.9520 |
| Linear SVM (SGD, hinge) | 0.9429 |
| Gaussian Naive Bayes | 0.9158 |
| k-NN | 0.7971 |

Full parameters are in `final_tuning_report.csv`.

The interesting row is the Ridge classifier. With the engineered features, a regularised linear model finishes level
with LightGBM and only 0.4 points behind CatBoost. Most of the signal is captured by the preprocessing, and the
boosted trees add a little on top.

## 5. Final model

A `StackingClassifier` with two base learners chosen to be as different as possible:

- **Ridge classifier** (standardised inputs, balanced class weights): linear, stable.
- **CatBoost** (depth 4, 963 trees, learning rate 0.018, balanced class weights): non-linear.
- **Meta-learner:** logistic regression (balanced class weights), trained on 5-fold out-of-fold base predictions.
- **Decision threshold:** 0.65 on the stacked probability.

## 6. Results

Out-of-fold predictions on all 60,000 training trucks, threshold 0.65 (`figures/confusion_matrix_oof.png`):

| | Predicted non-APS | Predicted APS |
|---|---|---|
| **Actual non-APS** | 57,098 | 1,816 |
| **Actual APS failure** | 51 | 1,035 |

| Metric | Value |
|---|---|
| Recall (failures caught) | 0.953 |
| Specificity | 0.969 |
| Balanced accuracy | 0.961 |
| Precision | 0.363 |
| ROC-AUC | 0.990 |
| Average precision (PR-AUC) | 0.768 |
| Challenge cost (10 × FP + 500 × FN) | 43,660 |

For scale: flagging no truck costs 543,000, flagging every truck costs 589,140.

## 7. Analysis

**The two base models make different mistakes.** Using out-of-fold predictions at each model's default threshold:

- 11 failures were caught by CatBoost but missed by Ridge.
- 23 failures were caught by Ridge but missed by CatBoost.

On its own, CatBoost therefore catches 12 fewer failures than Ridge at the default operating point, even though it
scored higher on balanced accuracy during tuning. An exact binomial (McNemar) test on these 34
discordant failures gives p ≈ 0.058, so neither model is clearly better on recall. What the split does show is that
their errors overlap only partly, which is the situation where stacking can help.

**What the stack gained is small.** The stack reaches 0.961 balanced accuracy at threshold 0.65; tuned CatBoost alone
reached 0.961 on 3 folds at its default threshold. The two numbers come from different protocols (3 vs 5 folds,
different thresholds), so I cannot claim a gain from stacking. A paired comparison on the same folds is needed.

**The probabilities are badly calibrated.** The reliability diagram (`figures/calibration_oof.png`) sits far below
the diagonal: trucks scored at 0.6–0.8 have an actual failure rate of only 3–5 %, and even the top bin
(≈ 0.97) is right about 46 % of the time. This is the expected side effect of balanced class weights in both the base
learners and the meta-learner, which shift every score towards the minority class. The model ranks trucks well
(ROC-AUC 0.99), but its outputs should not be read as failure probabilities.

**What the threshold actually does.** Because the scores are inflated, raising the threshold from 0.5 to 0.65 removes
false alarms at the cost of some recall; it does not increase recall. With the challenge's 50:1 cost ratio, a
missed failure costs as much as 50 false alarms, so the threshold should be chosen by minimising expected cost on
held-out predictions rather than set by hand.

**What drives the predictions.** SHAP on the CatBoost model (`figures/shap_summary_catboost.png`): high values of
`aa_000`, `ah_000` and `ai_000` push towards failure, and three of the five K-Means distance features are in the top
six, so the operating-profile features did add information. The missingness flags do not appear among the top 20.
Their signal is probably carried by the cluster distances and the imputed values of the same sensors, but I did not
test this.

## 8. Limitations

- **Preprocessing was fitted on the full training set** before cross-validation (imputer medians, K-Means
  centroids, scaler for K-Means, correlation filter). None of these steps uses the labels, so there is no label
  leakage, but validation folds did influence the feature statistics. A strict version would fit them inside each
  fold.
- No uncertainty is reported on the cross-validated metrics (single seed, no repeated CV, no confidence intervals).
- The histogram totals are sums of `log1p` bins, not the log of the summed counts.
- The Ridge `alpha` in the final pipeline (1.05) differs from the Optuna optimum (0.011).
- Test-set performance depends on the hidden labels of the course competition and is not reported here.

## 9. What I would do next

1. Fit all preprocessing inside the folds and compare stack vs CatBoost on identical, repeated folds.
2. Calibrate the stacked scores (isotonic or Platt on held-out folds, or a prior correction for the class weights)
   and check with a reliability diagram and expected calibration error.
3. Choose the threshold by minimising `10·FP + 500·FN` on out-of-fold predictions.
4. Test the missingness flags with an ablation instead of relying on SHAP.

## 10. Repository

```
mukherjee-dlpgc5-edafeature.ipynb      EDA, missingness analysis, feature engineering
mukherjee-dlpgc5-optuna.ipynb          Optuna tuning of 9 model families
mukherjee-dlpgc5-training-post.ipynb   final stacking model, evaluation plots, SHAP, Ridge vs CatBoost analysis
final_tuning_report.csv                best score and parameters per model
submission_final.csv                   predictions on the 16,000-truck test set
APS Failure Scania Truck - Predictive ML report.pdf   full written report
figures/                               plots exported from the notebooks
```

The notebooks were run on Kaggle (Python 3.11, scikit-learn, CatBoost, XGBoost, LightGBM, Optuna, SHAP).
To run them elsewhere, change the `/kaggle/input/...` paths to the location of `X_train.csv`, `y_train.csv` and
`X_test.csv`.

## Data source

Scania CV AB, *APS Failure at Scania Trucks*, IDA 2016 Industrial Challenge. Available from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks).
The data is not included in this repository.
