# 📉 Ridge Regression (Part 3): n-Dimensional Matrix Implementation from Scratch

## 📂 Overview
Building upon Part 2, this notebook tackles the much more complex scenario of **Multiple Ridge Regression (n-dimensional data)**. Instead of a simple slope formula, we utilize Linear Algebra and Matrix Operations to find the closed-form (Normal Equation) solution.

## 🧠 Key Concepts & Implementations
- **Dataset:** We return to the n-dimensional `diabetes` dataset.
- **Sklearn Baseline:** We use `Ridge(alpha=0.1, solver='cholesky')` to get our target coefficients and $R^2$ score (`0.4693`).
- **Custom `MainRidge` Class:** We build an n-dimensional Ridge Regression solver using NumPy matrix multiplication (`np.dot`) and inverses (`np.linalg.inv`).
- **The Core Formula:** 
  We implement the Ridge Normal Equation: 
  $$\theta = (X^T X + \alpha I)^{-1} X^T y$$
  where $I$ is the identity matrix (`np.identity`) which applies the L2 penalty uniformly across all feature dimensions.
- **Conclusion:** Our custom matrix-based class calculates the exact same $R^2$ score and multi-dimensional coefficient array as the optimized `scikit-learn` Cholesky solver.
