# 📊 Regression Evaluation Metrics

Welcome to the **Regression Metrics** directory! Building a machine learning model is only half the battle; knowing how to accurately evaluate its performance is what makes it useful.

## 🎯 Overview
In regression tasks, we predict continuous values. To understand how well our model's predictions match the actual data, we rely on specific mathematical metrics. This section covers the fundamental error and accuracy metrics used to quantify model performance.

### 🔑 Key Metrics Covered
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual values. It is easy to interpret and robust to outliers.
- **MSE (Mean Squared Error):** Averages the squared differences. It heavily penalizes larger errors compared to MAE.
- **RMSE (Root Mean Squared Error):** The square root of MSE. It brings the error metric back to the original units of the target variable.
- **R² Score (Coefficient of Determination):** Represents the proportion of variance in the dependent variable that is predictable from the independent variable(s). Ranges from 0 to 1 (higher is better).
- **Adjusted R² Score:** A modified version of R² that adjusts for the number of predictors in the model, penalizing unnecessary complexity.

## 📂 Contents
- **[`MAE, MSE, RMSE, R2 and Adjusted R2 Score, .ipynb`](<MAE, MSE, RMSE, R2 and Adjusted R2 Score, .ipynb>)**: An interactive Jupyter Notebook detailing the mathematical formulas, intuition, and practical Python implementation of each metric.

## 🚀 Getting Started
Launch the notebook to see these metrics in action:

```bash
jupyter notebook "MAE, MSE, RMSE, R2 and Adjusted R2 Score, .ipynb"
```

Mastering these metrics will significantly improve your ability to select and tune the best models! 💡
