# Gradient Boosting Classification - Mathematical Intuition (Part 3)

This notebook dives into the mathematical intuition behind using **Gradient Boosting** for Classification tasks. While Gradient Boosting is naturally suited for regression, applying it to classification involves predicting log-odds and using the sigmoid function to convert these log-odds into probabilities, which are then used to compute residuals.

---

## Objective

The notebook aims to:

- Illustrate how Gradient Boosting works under the hood for classification.
- Manually compute the initial predictions using log-odds.
- Convert log-odds into probabilities.
- Compute residuals based on the difference between actual target values and predicted probabilities.
- Train a `DecisionTreeRegressor` to predict these residuals.
- Update the log-odds and iterate the process.

---

## Dataset

The notebook uses a small, manually created dataset to trace the math easily.

### Dataset Structure

```python
df = pd.DataFrame({
    "Soil_pH": [1.20, 1.80, 2.50, 2.90, 3.40, 3.80, 4.10, 4.50, 4.80, 5.00],
    "Rainfall_mm": [102, 108, 115, 121, 110, 118, 123, 125, 116, 105],
    "Plant_Grown": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})
```

- Features: `Soil_pH`, `Rainfall_mm`
- Target: `Plant_Grown` (Binary Classification: 0 or 1)

---

## Libraries Used

```python
numpy
pandas
matplotlib
scikit-learn
```

Main Scikit-learn module:
- `DecisionTreeRegressor`

---

## Workflow

### 1. The Initial Prediction (Log-Odds)

In classification, we don't start by predicting the mean of the target. Instead, we start by calculating the initial log-odds of the positive class.

```python
log_odds = np.log(Positive_Count / Negative_Count)
```
For our dataset (6 positive, 4 negative), the initial log-odds prediction is `np.log(6/4) = 0.405465`.

### 2. Convert Log-Odds to Probability

To find our residual errors, we need a probability between 0 and 1. We apply the sigmoid function:

```python
Probability = 1 / (1 + np.exp(-log_odds))
```
In this step, `0.405465` converts back to a probability of `0.6`.

### 3. Compute the Residuals

The residuals (pseudo-residuals) for classification are simply the actual label minus the predicted probability:

```python
Residual = Actual - Probability
```
If the actual label is 1, the residual is `1 - 0.6 = 0.4`. If 0, it's `0 - 0.6 = -0.6`.

### 4. Train a Decision Tree on Residuals

Just like in regression, we train a Decision Tree, but we train a **DecisionTreeRegressor** (not a Classifier) to predict the continuous *residual* values.

### 5. Calculate Output Values for Tree Leaves

Once the tree is built, the raw prediction of the tree (which is the average of the residuals in each leaf) needs to be transformed back into a log-odds update. The formula for the update value ($\gamma$) in a leaf is:

$$ \gamma = \frac{\sum Residual_i}{\sum (Probability_i \times (1 - Probability_i))} $$

### 6. Update Predictions

The new log-odds for each instance are updated using a learning rate:

```python
New Log-Odds = Old Log-Odds + (Learning Rate * Leaf Update Value)
```

This updated log-odds value is then passed back through the sigmoid function to get the new probability, and the process repeats!

---

## Concepts Covered

- Gradient Boosting for Classification
- Log-Odds and the Sigmoid Function
- Pseudo-Residuals
- Decision Tree Regressor (applied to classification residuals)
- Leaf Output Transformation
- Mathematical Step-by-Step Execution

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
jupyter notebook Gradient_Boosting_Classsification_Maths_Part-3.ipynb
```

---

## Expected Output

The notebook produces:

- A clear, step-by-step DataFrame transformation.
- Initial log-odds calculation and probability conversion.
- Residual calculations for the dataset.
- Scatter plots visualizing the dataset.

---

## License

This project is intended for educational purposes as part of a Machine Learning learning series.