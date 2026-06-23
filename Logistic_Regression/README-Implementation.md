# 💻 Logistic Regression: Implementation & Gradient Descent

Welcome to the **Logistic Regression** implementation module! This file focuses on the transition from rigid Step Functions to probability-based classification using the Sigmoid curve and Gradient Descent.

## 📂 Contents & Findings

### [`Logistic_Regression-Part-3_Gradient_Descent.ipynb`](Logistic_Regression-Part-3_Gradient_Descent.ipynb)

**Goal:** Understand how to train a fully-fledged Logistic Regression model by implementing Gradient Descent from scratch with a continuous Sigmoid activation function.

* **The Sigmoid Function:** 
  * Unlike the binary Step Function (which outputs strictly $0$ or $1$), the Sigmoid function squashes predictions smoothly into a probability curve between $0$ and $1$: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  * We implement this mathematically to evaluate continuous outputs: `def sigmoid(z): return 1 / (1 + np.exp(-z))`.

* **Custom Gradient Descent Implementation:**
  * We build a custom Gradient Descent loop that iterates over randomly sampled data points.
  * For each sample, we calculate the continuous probability: `y_hat = sigmoid(np.dot(X[j], weights))`.
  * We then update the weights using the derivative of the Log Loss (Cross-Entropy Loss) function: `weights = weights + lr * (y[j] - y_hat) * X[j]`.
  * *Takeaway:* This loop beautifully demonstrates how the model learns smoothly and continuously over time, adjusting its boundary dynamically to minimize the error.

* **Scikit-Learn Verification:**
  * We instantiate a production-level `sklearn.linear_model.LogisticRegression` model on the exact same dataset.
  * By calculating and plotting the boundary for both models side-by-side (`m = -(coef_[0]/coef_[1])`), we prove that our custom Gradient Descent algorithm closely approximates the optimized math running under the hood in modern libraries.

## 🚀 Getting Started
Launch the notebook to visualize the Gradient Descent boundary convergence:
```bash
jupyter notebook
```
