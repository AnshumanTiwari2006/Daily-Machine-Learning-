# 📉 Ridge Regression (Part 1): Introduction with Scikit-Learn

## 📂 Overview
This notebook serves as a gentle introduction to Ridge Regression (L2 Regularization) using `scikit-learn`. It demonstrates how Ridge Regression helps penalize large coefficients compared to standard Ordinary Least Squares (OLS) Linear Regression.

## 🧠 Key Concepts & Implementations
- **Dataset:** We use the `diabetes` dataset from `sklearn.datasets`, which contains multiple features, making it a great candidate for regularization.
- **Linear Regression Baseline:** We first train a standard `LinearRegression()` model to establish a baseline $R^2$ score and observe its unpenalized coefficients.
- **Ridge Regression:** We train a `Ridge(alpha=0.001)` model and compare it to the baseline.
- **Comparison:** The notebook prints both the $R^2$ scores and the exact coefficients arrays for both models side-by-side, allowing us to visually inspect how the L2 penalty `alpha` (or $\lambda$) shrinks the coefficients towards zero to reduce overfitting.
