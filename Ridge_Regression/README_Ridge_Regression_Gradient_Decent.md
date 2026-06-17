# 📉 Ridge Regression: Gradient Descent Implementation

## 📂 Overview
While Part 3 utilized the closed-form Normal Equation to solve n-dimensional Ridge Regression, this notebook takes a completely different optimization path: **Gradient Descent**. 

## 🧠 Key Concepts & Implementations
- **Dataset:** We utilize the `diabetes` dataset for a multi-variable regression problem.
- **Sklearn Baselines:** 
  - We first evaluate an iterative solver using `SGDRegressor(penalty='l2')` (Stochastic Gradient Descent).
  - We also test `Ridge(solver='sparse_cg')` (Conjugate Gradient) to observe how iterative approaches perform.
- **Custom `MainRidge` Class (Gradient Descent Edition):** 
  - Instead of computing matrix inverses (which can be slow for massive datasets), we initialize a weight vector $\theta$ of ones.
  - We loop for a set number of `epochs`, computing the gradient derivative of the Ridge Cost Function:
    `thetha_der = (X^T \cdot X \cdot \theta) - (X^T \cdot y) - \alpha \cdot \theta`
  - We update the weights iteratively by subtracting the learning rate multiplied by the derivative.
- **Conclusion:** By using an `alpha=0.001`, `epochs=500`, and `learning_rate=0.005`, our custom Gradient Descent algorithm successfully converges, providing a high-quality $R^2$ score comparable to `scikit-learn`'s solvers without requiring an expensive matrix inversion!
