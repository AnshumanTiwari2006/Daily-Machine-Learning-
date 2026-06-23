# 📉 Polynomial Logistic Regression: Handling Non-Linear Boundaries

Welcome to the **Polynomial Logistic Regression** module! This notebook introduces the concept of applying polynomial transformations to classification problems.

## 📂 Contents & Concepts

### [`Polynomial_Logistics_Regression.ipynb`](Polynomial_Logistics_Regression.ipynb)
**Goal:** Understand why standard Logistic Regression fails on complex datasets and how we set up the problem to solve it.

* **The U-Shape Problem:**
  * The notebook begins by loading a custom dataset (`ushape.csv`) which contains two features (`X1`, `X2`) and a binary target label (`Y`).
  * When plotted on a scatter plot with colors representing the classes, the data exhibits a "U-shape" or non-linear separation boundary.
  
* **The Limitation of Standard Logistic Regression:**
  * Standard `LogisticRegression()` draws a **straight line** to separate classes. On a U-shaped dataset, a straight line will inevitably misclassify many points.
  * The notebook sets up the standard Logistic Regression estimator to demonstrate the baseline approach.
  
* **The Solution (Conceptual):**
  * To solve this, we must use the exact same trick used in Polynomial Regression: we wrap our input data in a `PolynomialFeatures` transformer before passing it to the Logistic Regression model. This allows the model to draw curves, circles, or complex U-shaped boundaries to perfectly separate the classes!

## 🚀 Getting Started
Launch the notebook to see the U-shaped dataset and the model setup:
```bash
jupyter notebook
```
