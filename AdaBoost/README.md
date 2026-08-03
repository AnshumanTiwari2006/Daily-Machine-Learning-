````markdown
# AdaBoost from Scratch

This notebook demonstrates how the **AdaBoost (Adaptive Boosting)** algorithm works by implementing it **from scratch** using Python and comparing its performance on a synthetic binary classification dataset.

The implementation builds AdaBoost without relying on Scikit-learn's `AdaBoostClassifier`, allowing you to understand the complete mathematics and working mechanism behind the algorithm.

---

## Objective

The notebook aims to:

- Implement AdaBoost from scratch.
- Train multiple Decision Stumps sequentially.
- Update sample weights after every iteration.
- Compute the alpha (model weight) for each weak learner.
- Combine all weak learners into a strong classifier.
- Evaluate the final model using standard classification metrics.

---

## Dataset

The notebook uses a **synthetic binary classification dataset** generated with Scikit-learn.

### Dataset Generation

```python
make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)
```

The class labels are converted from:

```text
0, 1
```

to

```text
-1, +1
```

which is the standard formulation used in the AdaBoost algorithm.

---

## Libraries Used

```python
numpy
pandas
scikit-learn
```

Main Scikit-learn modules:

- DecisionTreeClassifier
- make_classification
- train_test_split
- accuracy_score
- precision_score
- recall_score
- f1_score
- roc_auc_score
- confusion_matrix

---

## Workflow

### 1. Generate the Dataset

A synthetic binary classification dataset is created using `make_classification()`.

---

### 2. Split the Dataset

The data is divided into:

- 80% Training
- 20% Testing

---

### 3. Implement AdaBoost from Scratch

A custom `Custom_AdaBoost` class is implemented.

The implementation includes:

- Sample weight initialization
- Training Decision Stumps
- Weighted error calculation
- Alpha computation
- Sample weight updates
- Prediction using weighted voting

---

### 4. Train Multiple Weak Learners

Each iteration:

- Trains one Decision Stump.
- Computes its weighted classification error.
- Calculates the learner's importance (alpha).
- Updates sample weights to emphasize previously misclassified samples.

This process repeats for the specified number of estimators.

---

### 5. Make Predictions

Predictions from all weak learners are combined using a weighted majority vote.

The final prediction is determined by:

```text
sign(Σ αᵢ × hᵢ(x))
```

where:

- αᵢ = Weight of the i-th weak learner
- hᵢ(x) = Prediction of the i-th weak learner

---

### 6. Evaluate the Model

The notebook evaluates the trained AdaBoost model using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## What is AdaBoost?

AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines several weak learners into a single strong classifier.

Instead of training all models independently, AdaBoost trains them **sequentially**.

After every iteration:

- Misclassified samples receive higher weights.
- Correctly classified samples receive lower weights.

As a result, each new weak learner focuses more on the mistakes made by previous learners.

---

## Mathematical Concepts

### Weighted Error

\[
\epsilon = \sum w_i \cdot I(y_i \neq \hat{y}_i)
\]

where:

- \(w_i\) = Sample weight
- \(I\) = Indicator function

---

### Alpha (Weak Learner Weight)

\[
\alpha=\frac{1}{2}\ln\left(\frac{1-\epsilon}{\epsilon}\right)
\]

A lower error produces a higher alpha, meaning that the weak learner contributes more to the final prediction.

---

### Sample Weight Update

\[
w_i = w_i \times e^{-\alpha y_i h_i(x_i)}
\]

The weights are then normalized so that their sum equals 1.

---

## Concepts Covered

- Ensemble Learning
- Boosting
- AdaBoost Algorithm
- Decision Stumps
- Weighted Error
- Alpha Calculation
- Sample Weight Updates
- Sequential Learning
- Binary Classification
- Model Evaluation Metrics

---

## 📂 Additional Modules

### `AdaBoost_Part-2.ipynb`: Visualizing the Mathematics
While the primary notebook builds the AdaBoost algorithm from scratch on a larger synthetic dataset, **Part 2** focuses on a highly visual, micro-dataset (only 10 data points) to trace the exact mathematical updates row-by-row.

**Key Highlights:**
- **Step-by-Step Visualization:** We use `mlxtend.plotting` to draw the exact 2D decision boundary of the first `DecisionTreeClassifier(max_depth=1)` (the "stump").
- **Transparent Calculations:** We explicitly calculate the mathematical error of the stump, compute the `alpha` (model weight) using $\alpha = \frac{1}{2}\ln\left(\frac{1-\epsilon}{\epsilon}\right)$, and manually update the weight of every single row in the Pandas DataFrame using $e^{\alpha}$ (for misclassified points) or $e^{-\alpha}$ (for correctly classified points).
- **Intuition over Code:** This notebook is designed to prove the exact mathematical mechanics behind AdaBoost in a completely transparent, step-by-step DataFrame execution, leaving no "black box" mystery.

### `AdaBoost_Part-3.ipynb`: Training via Weighted Resampling
In this notebook, we implement AdaBoost from scratch using a purely procedural, functional approach, but with one massive architectural difference: **Weighted Sampling**.

**Key Highlights:**
- **The Resampling Technique:** Not all machine learning algorithms natively support `sample_weight` parameters. To get around this, AdaBoost can physically alter the dataset before training the next learner. 
- **Bootstrapping with Probabilities:** Instead of just passing weights to the model, we use `np.random.choice(replace=True, p=weights)` to actively pull a new, bootstrapped dataset where misclassified points have a mathematically higher chance of being drawn.
- **The Result:** The next Decision Stump is forced to focus on the hard-to-predict data points simply because there are now multiple physical copies of them in the newly sampled training set!

---

## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install numpy pandas scikit-learn
```

3. Open the notebook

```bash
jupyter notebook AdaBoost.ipynb
```

---

## Expected Output

The notebook produces:

- A complete implementation of AdaBoost from scratch.
- Trained weak learners (Decision Stumps).
- Alpha values for each estimator.
- Final predictions using weighted voting.
- Accuracy, Precision, Recall, F1 Score, ROC-AUC Score, and Confusion Matrix.

---

## License

This project is intended for educational purposes as part of a Machine Learning learning series.
````
