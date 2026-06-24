# 📉 Gradient Descent Optimization Algorithms

Welcome to the **Gradient Descent** module! This directory explores the core optimization engine that powers almost all modern machine learning and deep learning models.

## 🎯 Overview

Gradient Descent is a first-order iterative optimization algorithm used to minimize a cost function (e.g., Mean Squared Error) and discover the optimal parameters (weights and biases) for a model. It achieves this by taking iterative steps in the opposite direction of the mathematical gradient.

**The Core Update Rule:**
$$ \theta_{new} = \theta_{old} - \alpha \nabla J(\theta) $$
Where:
- $\theta$ represents the parameters (weights/coefficients and intercept).
- $\alpha$ is the **Learning Rate**, controlling the size of the step.
- $\nabla J(\theta)$ is the gradient (derivative) of the cost function $J$ with respect to the parameters.

## 📂 Contents & Findings

### 1. [`Batch_&__Mini_Batch_&_Stochastic_GD.ipynb`](Batch_%26__Mini_Batch_%26_Stochastic_GD.ipynb)
This comprehensive notebook builds a custom `UniversalGD` class from scratch, elegantly unifying the three major variations of Gradient Descent into a single framework.

* **Batch Gradient Descent:**
  * Uses the **entire training dataset** to compute the gradient at each step.
  * *Formula Focus:* $\frac{\partial J}{\partial w} = -\frac{2}{N} \sum_{i=1}^{N} x_i (y_i - \hat{y}_i)$
  * *Interpretation:* Very stable and smooth convergence, but extremely slow and memory-intensive for large datasets.

* **Stochastic Gradient Descent (SGD):**
  * Uses a **single, randomly chosen training example** to compute the gradient at each step.
  * *Formula Focus:* $\frac{\partial J}{\partial w} = -2 x_i (y_i - \hat{y}_i)$
  * *Interpretation:* Extremely fast and memory-efficient. The noisy updates can help escape shallow local minima, but it fluctuates wildly and rarely settles exactly at the absolute minimum.

* **Mini-Batch Gradient Descent:**
  * The sweet spot: Uses a **small random subset** (e.g., 32 samples) of the training data at each step.
  * *Formula Focus:* $\frac{\partial J}{\partial w} = -\frac{2}{B} \sum_{i=1}^{B} x_i (y_i - \hat{y}_i)$
  * *Interpretation:* Balances the speed of SGD with the stability of Batch GD. This is the defacto standard for training neural networks.

* **Performance & Comparison:**
  * We train our custom `UniversalGD` on the `diabetes` dataset across all three modes to observe their convergence.
  * We validate our math by comparing our custom implementation directly against Scikit-Learn's highly optimized `LinearRegression` (exact solver) and `SGDRegressor`.

### 2. [`batch-gradient-descent.ipynb`](batch-gradient-descent.ipynb)
**Goal:** A focused, stripped-down introduction strictly for **Batch Gradient Descent**.
* Implements a standalone `GDRegressor` class.
* Unpacks the manual calculation of the intercept derivative `(-2 * np.mean(y_train - y_hat))` and coefficient derivatives `(-2 * np.dot((y_train - y_hat), X_train) / N)`.
* Provides a baseline $R^2$ score comparison to prove the raw mathematics align with production-level libraries.

## 🚀 Getting Started
Launch the unified notebook to explore the algorithms side-by-side:
```bash
jupyter notebook "Batch_&__Mini_Batch_&_Stochastic_GD.ipynb"
```
