# 📉 Lasso Regression: L1 Regularization & Feature Selection

Welcome to the **Lasso Regression** module! This folder explores how L1 Regularization (Lasso) differs from standard Linear Regression and L2 Regularization (Ridge).

## 📂 Contents & Findings

### [`Lasso_Regression_Part-1.ipynb`](Lasso_Regression_Part-1.ipynb)
**Goal:** Understand how the L1 penalty shrinks coefficients, and crucially, how it performs automatic **Feature Selection**.

* **Visualizing the Alpha Penalty (1D Data):** 
  * We generate a simple 1D synthetic dataset (`noise=20`).
  * We fit `Lasso` models using an array of different `alpha` values (`[0, 5, 10, 100]`) and plot their regression lines. 
  * *Takeaway:* As `alpha` increases, you can visually see the slope of the line getting squashed closer and closer to a flat horizontal line (slope = 0).

* **The Core Superpower of Lasso (n-Dimensional Data):**
  * We load the multi-variable `diabetes` dataset to truly see Lasso in action.
  * We train a `Lasso(alpha=0.1)` model and print its coefficient array. 
  * *The Magic:* If you look closely at the Lasso coefficients, you'll see exact `0.` values! (`[0., -113.9, ..., -0., ..., 0., ...]`). Because the L1 penalty uses absolute values instead of squares, it aggressively drives less-important feature coefficients to exactly zero, effectively removing them from the model.
  * *The Comparison:* Finally, we train a `Ridge(alpha=0.1)` model on the exact same data. In Ridge, the coefficients get smaller, but none of them are driven to exactly `0.0`. This beautifully demonstrates why Lasso is the go-to tool for automatic feature selection!

## 🚀 Getting Started
Launch the notebook to see the plots and the feature-selection magic happen:
```bash
jupyter notebook
```
