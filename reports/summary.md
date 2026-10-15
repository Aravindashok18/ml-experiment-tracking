# Experiment summary

49 finished runs (48 model configurations plus a class-prior baseline), 5-fold stratified cross-validation on the training file, one evaluation on the held-out test file (16,281 rows).

Baseline (always predicts the class prior): ROC AUC 0.500, accuracy 0.764 (the positive class is 23.6% of the test rows).

## Families

Selection uses cross-validated ROC AUC only. The champion's test scores are what it achieved afterwards.

| family | runs | cv_auc_best | cv_auc_median | cv_auc_worst | champion_test_auc | champion_test_f1 | median_train_seconds |
|---|---|---|---|---|---|---|---|
| XGBoost | 18 | 0.9292 | 0.9256 | 0.9138 | 0.9277 | 0.7091 | 0.2 |
| Random Forest | 18 | 0.9192 | 0.9072 | 0.8975 | 0.9180 | 0.6991 | 0.4 |
| Logistic Regression | 12 | 0.9051 | 0.9047 | 0.8834 | 0.9032 | 0.6720 | 0.1 |


## Ten best runs by cross-validated ROC AUC

| family | config_id | cv_roc_auc_mean | cv_roc_auc_std | cv_f1_mean | test_roc_auc | test_f1 | train_seconds |
|---|---|---|---|---|---|---|---|
| XGBoost | 35fff94837 | 0.9292 | 0.0039 | 0.7165 | 0.9277 | 0.7091 | 0.2 |
| XGBoost | 0d7afd7508 | 0.9282 | 0.0039 | 0.7153 | 0.9269 | 0.7081 | 0.3 |
| XGBoost | 3c9b21f833 | 0.9278 | 0.0035 | 0.7142 | 0.9264 | 0.7075 | 0.2 |
| XGBoost | 77eda3d413 | 0.9278 | 0.0044 | 0.7056 | 0.9261 | 0.7035 | 0.1 |
| XGBoost | 8e7f689261 | 0.9278 | 0.0041 | 0.7128 | 0.9263 | 0.7101 | 0.3 |
| XGBoost | 3580a514a9 | 0.9274 | 0.0041 | 0.7119 | 0.9257 | 0.7064 | 0.2 |
| XGBoost | aeaeae84a2 | 0.9268 | 0.0039 | 0.7051 | 0.9255 | 0.7004 | 0.2 |
| XGBoost | baca85f6fa | 0.9266 | 0.0045 | 0.7111 | 0.9242 | 0.7063 | 0.1 |
| XGBoost | 0db248a750 | 0.9259 | 0.0044 | 0.7013 | 0.9255 | 0.7037 | 0.3 |
| XGBoost | 89e212d192 | 0.9252 | 0.0038 | 0.7000 | 0.9247 | 0.6995 | 0.1 |


Ranking all 48 runs by cross-validation or by test score agrees closely: Spearman rho = 0.993 (p = 4.3e-44).

## Do the family champions differ?

Paired bootstrap over the same test rows, a minus b; the interval is the 2.5th to 97.5th percentile of 1,000 resamples. An interval that contains 0 means the test set cannot separate the two.

| a | b | metric | diff | ci_low | ci_high | share_a_better |
|---|---|---|---|---|---|---|
| XGBoost | Random Forest | test ROC AUC | +0.0098 | +0.0079 | +0.0114 | 1.00 |
| XGBoost | Random Forest | test F1 | +0.0098 | -0.0004 | +0.0210 | 0.97 |
| XGBoost | Logistic Regression | test ROC AUC | +0.0244 | +0.0220 | +0.0271 | 1.00 |
| XGBoost | Logistic Regression | test F1 | +0.0369 | +0.0265 | +0.0481 | 1.00 |
| Random Forest | Logistic Regression | test ROC AUC | +0.0147 | +0.0121 | +0.0172 | 1.00 |
| Random Forest | Logistic Regression | test F1 | +0.0270 | +0.0200 | +0.0339 | 1.00 |


## Which hyperparameters mattered

Spearman correlation with cross-validated AUC inside each family (random search, so each hyperparameter varies independently of the others). 11 correlations were tested, so the last column applies a Bonferroni threshold of 0.05 / 11; a p-value just under 0.05 above it is not evidence on its own.

| family | hyperparameter | n | spearman_rho | p_value | significant_after_bonferroni |
|---|---|---|---|---|---|
| Random Forest | max_depth | 18 | +0.721 | 0.0007 | True |
| Logistic Regression | C | 12 | +0.762 | 0.0040 | True |
| XGBoost | max_depth | 18 | +0.480 | 0.0437 | False |
| Random Forest | min_samples_leaf | 18 | -0.292 | 0.2393 | False |
| XGBoost | n_estimators | 18 | +0.284 | 0.2533 | False |
| XGBoost | reg_lambda | 18 | +0.280 | 0.2610 | False |
| XGBoost | learning_rate | 18 | +0.162 | 0.5207 | False |
| XGBoost | colsample_bytree | 18 | -0.127 | 0.6157 | False |
| XGBoost | min_child_weight | 18 | -0.122 | 0.6304 | False |
| Random Forest | n_estimators | 18 | -0.020 | 0.9374 | False |
| XGBoost | subsample | 18 | +0.015 | 0.9514 | False |


## Categorical hyperparameters

Mean cross-validated AUC per level. Levels were drawn at random, so some have only a few runs (see `runs`) and the other hyperparameters differ between levels; read these as descriptions, not tests.

| family | hyperparameter | level | runs | mean_cv_auc |
|---|---|---|---|---|
| Logistic Regression | penalty | l1 | 6 | 0.8940 |
| Logistic Regression | penalty | l2 | 6 | 0.9049 |
| Logistic Regression | class_weight | None | 2 | 0.9048 |
| Logistic Regression | class_weight | balanced | 10 | 0.8984 |
| Random Forest | class_weight | None | 9 | 0.9046 |
| Random Forest | class_weight | balanced_subsample | 9 | 0.9093 |
| Random Forest | max_features | 0.3 | 4 | 0.9105 |
| Random Forest | max_features | log2 | 10 | 0.9041 |
| Random Forest | max_features | sqrt | 4 | 0.9103 |
