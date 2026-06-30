# ⚖️ Feature Scaling: Standardization & Normalization

Welcome to the **Feature Scaling** module! In machine learning, algorithms that calculate geometric distances (like KNN or K-Means) or rely on gradient descent (like Linear/Logistic Regression and Neural Networks) are extremely sensitive to the raw scale of the data. 

If one feature is measured in hundreds of thousands (e.g., Salary) and another in single digits (e.g., Age), the larger feature will mathematically dominate the model's cost function. This directory covers how to scale features to put them all on a level playing field.

## 🎯 Overview

Feature scaling ensures that no single feature artificially influences the model simply because of its unit of measurement. We focus on the two most critical scaling techniques:

1. **Standardization (Z-score Normalization):** Centers the data around a mean of 0 with a standard deviation of 1.
2. **Normalization (Min-Max Scaling):** Squeezes all data into a fixed mathematical boundary, usually strictly between 0 and 1.

## 📂 Contents & Findings

### 1. Standardization
* [`Standardization .ipynb`](Standardization%20.ipynb) / [`Standardization(Qwen).ipynb`](Standardization(Qwen).ipynb) / [`Z-score Normalization - scikit Learn.ipynb`](Z-score%20Normalization%20-%20scikit%20Learn.ipynb)
* **The Goal:** Transform the dataset so that its distribution has a mean ($\mu$) of $0$ and a standard deviation ($\sigma$) of $1$.
* **The Formula:** $z = \frac{x - \mu}{\sigma}$
* **When to use:** This is the default, go-to technique for most machine learning algorithms. It handles outliers slightly better than Normalization and is exceptionally necessary for algorithms like Principal Component Analysis (PCA), Support Vector Machines (SVM), and Gradient Descent optimization.
* **Implementation:** We explore the raw mathematics of the Z-score and implement it cleanly using Scikit-Learn's highly optimized `StandardScaler` class, visualizing the before-and-after distributions using KDE (Kernel Density Estimate) plots.

### 2. Normalization
* [`Normalization.ipynb`](Normalization.ipynb)
* **The Goal:** Scale all the features so they strictly fall within a specific boundary—typically exactly between 0 and 1.
* **The Formula:** $X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$
* **When to use:** Highly useful when you know the distribution of your data does *not* follow a normal (Gaussian) curve. It is also practically mandatory for algorithms like Neural Networks handling image pixel data (which inherently ranges from 0 to 255).
* **Implementation:** We utilize Scikit-Learn's `MinMaxScaler` to lock our data into the $[0, 1]$ range and visualize the exact geometric shift of the data points before and after scaling using scatter plots.

## 🚀 Getting Started
Launch any of the notebooks to explore how to effectively scale your datasets before training:
```bash
jupyter notebook "Standardization .ipynb"
```
