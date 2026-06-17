# 📉 Gradient Descent for Regression

Welcome to the **Gradient Descent** module! This folder contains three Jupyter Notebooks where we break down the mathematics of Gradient Descent and build custom Python classes to replicate the behavior of `scikit-learn`'s Linear Regression models.

## 📂 Contents & Findings

### 1. [`Gradient_Descent.ipynb`](Gradient_Descent.ipynb)
**Goal:** Build a full Gradient Descent optimizer from scratch to find both the slope (`m`) and intercept (`b`).
* **The Process:** We first generate a synthetic dataset (`noise=50`) and fit it using `sklearn.linear_model.LinearRegression` to get our benchmark coefficients (`m ≈ 28.16`, `b ≈ -5.74`). 
* **Custom Implementation:** We then build a `GradientRegressor` class that calculates the partial derivatives (`loss_slope_wrt_b` and `loss_slope_wrt_m`) to update both parameters simultaneously.
* **Result:** After just 50 epochs with a learning rate of `0.001`, our custom model perfectly converges to the exact same values as the `sklearn` model!

### 2. [`Gradient_Descent_Mathematically_where_m_or_b_given.ipynb`](<Gradient_Descent_Mathematically_where_m_or_b_given.ipynb>)
**Goal:** Understand the calculus and cost function behavior when one parameter is fixed.
* **The Process:** Uses `make_regression` with a much higher noise level (`noise=90`).
* **Focus:** This notebook mathematically breaks down how the gradient behaves and how the loss curve looks when we already know either the slope or the intercept, reducing the optimization problem from 3D to 2D. 

### 3. [`Gradient_Descent_Own_Class.ipynb`](Gradient_Descent_Own_Class.ipynb)
**Goal:** Step-by-step optimization tracking for a single parameter.
* **The Process:** We generate a cleaner dataset (`noise=20`) which gives an OLS baseline of `m = 53.15` and `b = 1.013`.
* **Custom Implementation:** We build another `GradientRegressor` class, but this time we **hardcode the slope** (`m = 53.15`) and use Gradient Descent to find *only* the intercept `b`.
* **Result:** Starting from a terrible initial guess of `b = -100`, the notebook prints the loss slope and the updated `b` value at every single step for 100 epochs, demonstrating exactly how the steps get smaller as `b` converges beautifully to `1.01313`.

## 🚀 Getting Started
Launch the notebooks to see the custom classes and the math in action:
```bash
jupyter notebook
```
