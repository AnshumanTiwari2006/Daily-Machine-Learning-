# 📚 Logistic Regression: Theory & Fundamentals

Welcome to the **Logistic Regression** theoretical module! This file focuses on the mathematical bedrock of linear classification: the Perceptron and Decision Boundaries.

## 📂 Contents & Findings

### [`Logistic_Regression-Part-1.ipynb`](Logistic_Regression-Part-1.ipynb) & [`Logistic_Regression-Part-2.ipynb`](Logistic_Regression-Part-2.ipynb)

**Goal:** Understand the fundamental building block of linear classification—the Perceptron—and how simple step functions can divide a feature space.

* **The Perceptron Algorithm from Scratch:**
  * Instead of relying on probability curves, we build a custom Perceptron using a rigid **Step Function**: `def step(z): return 1 if z > 0 else 0`.
  * We randomly sample data points, calculate the predicted label using the step function, and update the weights iteratively if the prediction is wrong: `weights = weights + lr * (y[j] - y_hat) * X[j]`.
  
* **Mathematical Theory & Formula:** 
  * The core theory here is that a linear boundary defines a decision threshold in $n$-dimensional space. 
  * In 2D, the condition is: If $w_1x_1 + w_2x_2 + b > 0$, we classify as Class 1. Otherwise, Class 0.

* **Visualizing the Boundary:**
  * We use `make_classification` from `sklearn` to generate synthetic clustered data.
  * We extract the custom model's intercept and coefficients to compute the slope (`m = -(coef[0]/coef[1])`) and y-intercept (`b = -(intercept/coef[1])`).
  * *Takeaway:* By plotting the line $y = mx + b$ over the scatter plot, we visually confirm that our raw Perceptron logic successfully discovers a hyperplane separating the two distinct classes.

## 🚀 Getting Started
Launch the notebooks to see the raw Perceptron logic in action:
```bash
jupyter notebook
```
