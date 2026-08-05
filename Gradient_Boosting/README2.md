# Gradient Boosting (Part 1)

This notebook demonstrates the fundamental working of the **Gradient Boosting Regressor** by implementing the algorithm step-by-step using **Decision Tree Regressors**. Instead of directly using Scikit-learn's `GradientBoostingRegressor`, the notebook builds the boosting process manually to illustrate how each successive tree learns from the residual errors of the previous model.

---

## Objective

The notebook aims to:

- Understand the intuition behind Gradient Boosting.
- Generate a synthetic regression dataset.
- Train the first model using a constant prediction.
- Compute residuals after each iteration.
- Train Decision Tree Regressors on the residuals.
- Update predictions sequentially.
- Visualize how the model improves after every boosting stage.
- Implement a recursive Gradient Boosting algorithm from scratch.

---

## Dataset

The notebook creates a synthetic regression dataset.

### Dataset Generation

```python
np.random.seed(42)

X = np.random.rand(100, 1) - 0.5
y = 4 * X[:, 0]**2 + 0.05 * np.random.randn(100)
```

### Dataset Characteristics

- Samples: 100
- Features: 1
- Task: Regression
- Target Function:

\[
y = 4x^2 + \text{noise}
\]

---

## Libraries Used

```python
numpy
pandas
matplotlib
scikit-learn
```

Main Scikit-learn module:

- DecisionTreeRegressor

---

## Workflow

### 1. Generate the Dataset

A synthetic regression dataset is generated using NumPy.

---

### 2. Visualize the Data

The relationship between the feature and target variable is visualized using a scatter plot.

---

### 3. Initialize the First Prediction

The first prediction is simply the mean of the target variable.

```python
prediction = y.mean()
```

This serves as the starting point for Gradient Boosting.

---

### 4. Compute the Residuals

Residuals are calculated as:

```python
Residual = Actual − Prediction
```

These residuals represent the errors that the next model attempts to learn.

---

### 5. Train the First Decision Tree

A Decision Tree Regressor is trained on the residuals.

```python
DecisionTreeRegressor(max_leaf_nodes=8)
```

The tree learns the remaining error instead of the original target values.

---

### 6. Update Predictions

The predictions from the tree are added to the previous prediction.

```python
New Prediction = Previous Prediction + Tree Prediction
```

This produces a more accurate approximation of the target function.

---

### 7. Repeat the Process

The notebook repeats the following steps:

- Compute new residuals.
- Train another Decision Tree Regressor.
- Update predictions.
- Visualize the improved regression curve.

Each new tree focuses only on correcting the mistakes made by the previous ensemble.

---

### 8. Recursive Gradient Boosting Implementation

The notebook defines a custom recursive function:

```python
GB(X, y, num, lr)
```

The function:

- Trains multiple Decision Trees sequentially.
- Updates residuals after each iteration.
- Applies the specified learning rate.
- Plots the regression curve after every boosting stage.

This provides a clear visualization of how Gradient Boosting gradually improves the model.

---

## What is Gradient Boosting?

Gradient Boosting is an ensemble learning technique that builds a strong predictive model by sequentially combining multiple weak learners.

Unlike Random Forest, where trees are trained independently, Gradient Boosting trains each new tree to predict the residual errors made by the existing ensemble.

Each iteration reduces the overall prediction error, resulting in progressively better performance.

---

## Concepts Covered

- Ensemble Learning
- Boosting
- Gradient Boosting
- Decision Tree Regressor
- Residual Learning
- Sequential Model Training
- Recursive Algorithm Implementation
- Learning Rate
- Function Approximation

---

## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install the required libraries

```bash
pip install numpy pandas matplotlib scikit-learn
```

3. Open the notebook

```bash
jupyter notebook Gradient_Boosting-Part-1.ipynb
```

---

## Expected Output

The notebook produces:

- A synthetic regression dataset.
- Scatter plot of the generated data.
- Residual calculations after each boosting stage.
- Decision Tree models trained on residuals.
- Progressive improvement of the regression curve.
- A recursive implementation of Gradient Boosting from scratch with visualizations after every iteration.

---

## License

This project is intended for educational purposes as part of a Machine Learning learning series.