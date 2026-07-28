# Feature Importance in Random Forest (Part 2)

This notebook demonstrates two approaches for calculating **Feature Importance** in a Random Forest model using the **Iris Dataset**.

The first approach uses Scikit-learn's built-in implementation, while the second manually computes feature importance by traversing every decision tree and calculating impurity reduction at each split.

---

## Objective

The notebook aims to:

- Train a Random Forest Classifier on the Iris dataset.
- Calculate feature importance using Scikit-learn.
- Manually compute feature importance from every decision tree.
- Compare the manually calculated values with Scikit-learn's output.
- Understand how Random Forest internally measures feature importance.

---

## Dataset

**Dataset Used**

- Iris Dataset (loaded from Scikit-learn)

### Features

- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

### Target Classes

- Setosa
- Versicolor
- Virginica

---

## Libraries Used

```python
numpy
pandas
scikit-learn
```

Main Scikit-learn modules:

- RandomForestClassifier
- load_iris

---

## Workflow

### 1. Load the Iris Dataset

```python
from sklearn.datasets import load_iris
```

---

### 2. Train the Random Forest Classifier

```python
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf.fit(X, y)
```

---

### 3. Calculate Feature Importance (Scikit-learn)

The built-in feature importance is obtained using:

```python
clf.feature_importances_
```

The values are displayed in descending order of importance.

---

### 4. Manually Calculate Feature Importance

The notebook manually computes feature importance by:

- Iterating through every decision tree in the forest.
- Visiting each internal node.
- Computing impurity reduction at every split.
- Weighting the reduction by the number of samples reaching the node.
- Summing contributions for every feature.
- Averaging across all trees.
- Normalizing the final importance values.

---

### 5. Compare Results

Finally, the notebook compares:

- Scikit-learn Feature Importance
- Manual Feature Importance

Both methods produce nearly identical results, demonstrating how Scikit-learn internally calculates feature importance.

---

## What is Feature Importance?

Feature Importance measures how much each feature contributes to improving the predictions of a Random Forest model.

Whenever a feature is used to split a node, it reduces impurity. Features that consistently produce larger impurity reductions across many trees receive higher importance scores.

The final importance values are normalized so that their sum equals **1.0**.

---

## Mathematical Formula

For each split:

\[
Importance =
\frac{N_t}{N}
\left(
I_t
-
\frac{N_L}{N_t}I_L
-
\frac{N_R}{N_t}I_R
\right)
\]

Where:

- **N** = Total number of training samples
- **Nt** = Samples reaching the current node
- **NL** = Samples in the left child
- **NR** = Samples in the right child
- **It** = Parent node impurity
- **IL** = Left child impurity
- **IR** = Right child impurity

The impurity reductions from every split of every tree are summed and normalized to obtain the final feature importance.

---

## Concepts Covered

- Decision Trees
- Random Forest
- Gini Impurity
- Impurity Reduction
- Feature Importance
- Manual Feature Importance Calculation
- Scikit-learn Feature Importance
- Tree Traversal

---

## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install the required libraries

```bash
pip install numpy pandas scikit-learn
```

3. Open the notebook

```bash
jupyter notebook Feature_Importance_Part-2.ipynb
```

---

## Expected Output

The notebook generates:

- Feature importance using Scikit-learn
- Manually calculated feature importance
- A comparison showing that both methods produce nearly identical results

---

## License

This project is intended for educational purposes as part of a Machine Learning learning series.