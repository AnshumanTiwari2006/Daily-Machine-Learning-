# 🚀 Gradient Boosting (Gradient Boosting Machines - GBM)

Welcome to the **Gradient Boosting** module! This folder contains theoretical explanations and practical implementations of one of the most powerful and widely used ensemble learning algorithms in modern machine learning.

## 🎯 What is Gradient Boosting?

Gradient Boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

Unlike **AdaBoost**, which adjusts the *weights* of training instances based on previous errors, **Gradient Boosting** trains each new model directly on the **residual errors** (differences between the predicted and actual values) of the previous models.

## 🧠 The Mathematical Intuition

The core idea of Gradient Boosting is to minimize a loss function $L(y, F(x))$ by iteratively adding weak learners $h(x)$.

### 1. Initialize the Model
We start with a constant prediction $F_0(x)$, which is typically the mean of the target variable for regression or the log-odds for classification.
$$ F_0(x) = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, \gamma) $$

### 2. Iterative Learning (For $m = 1$ to $M$)

For each iteration $m$:

**A. Calculate Pseudo-Residuals**
We compute the negative gradient of the loss function with respect to the previous model's predictions. These are called pseudo-residuals $r_{im}$:
$$ r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)} $$
*For Mean Squared Error (MSE), the pseudo-residual is exactly the true residual: $y_i - F_{m-1}(x_i)$.*

**B. Fit a Weak Learner**
We train a weak learner (usually a shallow Decision Tree) $h_m(x)$ to predict these residuals $r_{im}$.

**C. Compute the Multiplier (Step Size)**
We find the optimal multiplier $\gamma_m$ that minimizes the loss when this new tree is added to the ensemble:
$$ \gamma_m = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i)) $$

**D. Update the Model**
We update our main model by adding the new tree, scaled by a **learning rate** $\eta$ (shrinkage parameter) to prevent overfitting:
$$ F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m h_m(x) $$

### 3. Final Prediction
After $M$ iterations, the final model is:
$$ F_M(x) = F_0(x) + \eta \sum_{m=1}^M \gamma_m h_m(x) $$

## ⚖️ AdaBoost vs. Gradient Boosting

| Feature | AdaBoost | Gradient Boosting |
| :--- | :--- | :--- |
| **Handling Errors** | Increases weights of misclassified data points. | Predicts the residual errors of the previous model directly. |
| **Weak Learner** | Usually Decision Stumps (Trees with max depth 1). | Usually shallow Decision Trees (Trees with depth 3-5). |
| **Loss Function** | Exponential Loss. | Any differentiable loss function (MSE, Log-Loss, etc.). |
| **Robustness to Outliers** | Highly sensitive to outliers (exponential loss). | Can be robust to outliers if a robust loss function (like Huber) is used. |

## ⚙️ Key Hyperparameters

1. **`n_estimators`**: The number of boosting stages to perform (number of trees).
2. **`learning_rate` ($\eta$)**: Shrinks the contribution of each tree. There is a trade-off between `learning_rate` and `n_estimators`.
3. **`max_depth`**: Maximum depth of the individual regression estimators. Limits the number of nodes in the tree.
4. **`subsample`**: The fraction of samples to be used for fitting the individual base learners (Stochastic Gradient Boosting).

## 📂 Implementation Details

In the accompanying notebook `Gradient_Boosting.ipynb`, you will find:
1. **Mathematical Validation**: A step-by-step custom implementation of Gradient Boosting for Regression using raw Decision Trees to predict residuals.
2. **Scikit-Learn Implementation**: Using `GradientBoostingRegressor` and `GradientBoostingClassifier` on real datasets.
3. **Hyperparameter Tuning**: Visualizing the effect of the learning rate and number of estimators on model fit.

---
*“Gradient Boosting gives you the power to optimize any differentiable loss function, turning a series of weak learners into an incredibly powerful predictive engine.”*
