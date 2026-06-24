# 🌸 Softmax Regression (Multinomial Logistic Regression)

Welcome to the **Softmax Regression** module! This folder explores how to extend binary classification to handle multiple distinct classes simultaneously without resorting to multiple independent binary models.

## 📂 Contents & Findings

### [`Softmax_Regression.ipynb`](Softmax_Regression.ipynb)

**Goal:** Understand how the Softmax function generalizes Logistic Regression to cleanly classify data into more than two mutually exclusive classes.

* **Theoretical Foundation (The Softmax Function):**
  * Standard Logistic Regression uses the **Sigmoid** function to output a probability for a binary (0/1) outcome.
  * When faced with $K$ classes (like 3 distinct flower species), we instead use the **Softmax** function: 
    $$ P(y=j|x) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}} $$
  * *Takeaway:* The Softmax function converts the raw scores (logits) for all classes into a proper probability distribution where the predicted probabilities across all $K$ classes sum to exactly $1.0$.

* **The Iris Dataset Implementation:**
  * We use the classic multi-class `iris` dataset, utilizing two continuous features (`sepal_length` and `petal_length`) to predict the three `species`.
  * We train a `sklearn.linear_model.LogisticRegression` model, which intelligently detects the multi-class nature of the labels and natively applies multinomial loss (Softmax) under the hood using the `lbfgs` solver.
  * When evaluating new data with `predict_proba()`, the model returns an array of probabilities (e.g., `[0.725, 0.273, 0.0004]`), clearly showing its confidence across all 3 classes simultaneously rather than just a single binary probability.

* **Visualizing Multi-Class Boundaries:**
  * We encode the string labels to integers and utilize the `mlxtend` library's `plot_decision_regions` to vividly map out the classification zones.
  * *Takeaway:* The resulting plot demonstrates how the algorithm maps out intersecting linear decision boundaries, effectively carving out distinct territorial regions for each of the three species in the feature space.

## 🚀 Getting Started
To run the notebook and see the multi-class decision boundaries, ensure you have the required libraries installed:
```bash
pip install mlxtend scikit-learn pandas numpy matplotlib seaborn
jupyter notebook Softmax_Regression.ipynb
```
