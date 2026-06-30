# 🔄 Data Transformers (Scikit-Learn)

Welcome to the **Transformers** module! In real-world datasets, raw data is rarely perfect. Features often exhibit heavy skewed distributions (right or left skew) rather than a clean normal distribution, and different columns inherently require entirely different preprocessing steps.

This directory explores how to mathematically transform data to make it Gaussian (normally distributed) and how to architect clean, scalable preprocessing pipelines using Scikit-Learn's transformation ecosystem.

## 🎯 Overview

Mathematical transformations are deeply crucial because many parametric machine learning algorithms (like Linear Regression, Logistic Regression, and Artificial Neural Networks) inherently assume that the input data follows a normal (Gaussian) distribution. If the data violates this assumption, model performance degrades.

In this module, we cover:
1. **Function Transformers:** Applying foundational mathematical functions (like logarithms) to fix right-skewed data.
2. **Power Transformers:** Using advanced, parameterized algorithmic transformations (like Box-Cox and Yeo-Johnson) to aggressively force stubborn data into a normal distribution.
3. **Column Transformers:** Designing robust, modular pipelines that map distinct preprocessing techniques to specific columns concurrently.

## 📂 Contents & Findings

### 1. [`Function Transformer.ipynb`](Function%20Transformer.ipynb)
**Goal:** Manually apply simple, explicit mathematical functions to correct minor to moderate skewness.
* **Techniques Explored:**
  * **Log Transform ($log(x)$):** Excellent for highly right-skewed data (e.g., income, house prices, or population).
  * **Reciprocal Transform ($1/x$):** Inverts the data; useful for specific ratios.
  * **Square Root ($x^{1/2}$) & Square ($x^2$) Transforms:** Used to moderately adjust left or right skews.
* **Key Finding:** We leverage QQ-plots (Quantile-Quantile plots) and KDE distributions to visually prove how the raw data morphs closer to a perfect bell curve after passing through a `FunctionTransformer`.

### 2. [`Power Transformer.ipynb`](Power%20Transformer.ipynb)
**Goal:** Apply sophisticated, algorithmic transformations to force highly skewed, complex data into a normal distribution.
* **Techniques Explored:**
  * **Box-Cox Transform:** A highly effective historical method, but it strictly requires all data points to be strictly positive ($>0$).
  * **Yeo-Johnson Transform:** The modern, versatile successor to Box-Cox that mathematically supports zero and negative values effortlessly.
* **Implementation:** We utilize Scikit-Learn's `PowerTransformer` to automatically calculate the optimal lambda ($\lambda$) parameter that mathematically maximizes normality across the distribution.

### 3. [`Column_Transformer_Cleaned.ipynb`](Column_Transformer_Cleaned.ipynb)
**Goal:** Elevate preprocessing from messy scripts to clean, modular, production-ready pipelines.
* **The Problem:** In a standard dataset, an `Age` column might require SimpleImputer + StandardScaler, a `Gender` column might require OneHotEncoder, and a `Salary` column might require a PowerTransformer. Doing this sequentially in Pandas is messy and leads to data leakage.
* **The Solution:** We implement Scikit-Learn's `ColumnTransformer` to route distinct columns through specific, parallel transformation pipelines, ultimately concatenating the outputs into a single, model-ready matrix array.

## 🚀 Getting Started
Launch the notebooks to master data transformations and pipeline architecture:
```bash
jupyter notebook "Column_Transformer_Cleaned.ipynb"
```
