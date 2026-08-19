# XGBoost (Extreme Gradient Boosting)

Welcome to the **XGBoost** module! This folder contains practical implementations and explanations of Extreme Gradient Boosting, a highly optimized, distributed gradient boosting library designed to be highly efficient, flexible, and portable.

## What is XGBoost?

XGBoost is an advanced implementation of gradient boosting algorithms. It pushes the limits of computing power for boosted tree algorithms, being built for maximizing performance and speed. It has historically been a dominating algorithm in machine learning competitions (like Kaggle) due to its excellent execution speed and model performance.

While standard Gradient Boosting builds trees sequentially to correct the errors of previous trees, **XGBoost** takes this further by introducing:
- **System Optimization**: Parallelized tree building, cache awareness, and out-of-core computing.
- **Algorithmic Enhancements**: Regularization (L1 and L2) to prevent overfitting, sparsity awareness (handling missing values automatically), and weighted quantile sketch.

## Contents & Findings

### 1. [`XGBoost_Part-1.ipynb`](XGBoost_Part-1.ipynb)
**Goal:** Introduce the XGBoost API, train a basic `XGBClassifier`, and explore its internal components.
- **The Process:** We create a small synthetic dataset (`X` with 3 features and `y` for binary classification). We split the data into training and testing sets using `train_test_split`.
- **Model Implementation:** We initialize and fit an `xgb.XGBClassifier` using specific hyperparameters:
  - `n_estimators=50`, `learning_rate=0.1`, `max_depth=3`
  - Regularization: `reg_alpha=0.1` (L1), `reg_lambda=1.0` (L2)
  - Execution: `tree_method="hist"`, `device="cpu"`
- **Result:** We predict the labels on the test set, evaluate the model using `accuracy_score`, print a visual text dump of the very first decision tree in the ensemble, and extract `feature_importances_` to see which features contributed the most to the predictions.

## Key Hyperparameters Demonstrated

1. **`reg_alpha` & `reg_lambda`**: L1 (Lasso) and L2 (Ridge) regularization terms on the leaf weights to prevent overfitting.
2. **`tree_method`**: Set to `"hist"` to use histogram-based split finding, which is significantly faster for large datasets.
3. **`n_estimators` & `learning_rate`**: The standard gradient boosting parameters controlling the number of sequential trees and the step size of each tree's contribution.

## Getting Started
Launch the notebook to see the XGBoost implementation in action:
```bash
jupyter notebook XGBoost_Part-1.ipynb
```

---
*“XGBoost is the go-to algorithm for tabular data, blending algorithmic efficiency with unparalleled predictive power.”*
