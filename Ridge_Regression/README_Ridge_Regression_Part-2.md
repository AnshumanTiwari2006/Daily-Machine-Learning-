# 📉 Ridge Regression (Part 2): 1D Mathematical Implementation from Scratch

## 📂 Overview
In this notebook, we move away from `scikit-learn`'s black box and build a custom Ridge Regression class from scratch. This part specifically focuses on **Simple Ridge Regression (1D data)** to clearly demonstrate the underlying mathematics of the L2 penalty.

## 🧠 Key Concepts & Implementations
- **Dataset Generation:** We generate a synthetic dataset with exactly one feature using `make_regression` (`n_samples=100`, `n_features=1`, `noise=25`).
- **Sklearn Baseline:** We establish benchmark coefficients using `sklearn.linear_model.Ridge(alpha=10)` (resulting in `m = 33.04`, `b = -1.78`).
- **Custom `RidgeReg` Class:** We implement the closed-form mathematical formula for 1D Ridge Regression:
  - **Slope (m):** Calculated as `num / (den + alpha)` where `num` is the covariance of X and y, and `den` is the variance of X. The `alpha` penalty is directly added to the denominator to shrink the slope.
  - **Intercept (b):** Calculated using the standard equation `mean(y) - m * mean(X)`.
- **Conclusion:** By running our custom `RidgeReg(alpha=10)` class, we perfectly match the `scikit-learn` baseline, proving our mathematical derivation is correct!
