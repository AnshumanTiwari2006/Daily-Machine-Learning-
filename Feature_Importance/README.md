# Feature Importance in Random Forest

This notebook demonstrates how **Feature Importance** is calculated in a **Random Forest Classifier**. It covers both the built-in Scikit-learn implementation and the manual computation of feature importance using impurity reduction.

## Objective

The notebook aims to:

- Train a Random Forest Classifier
- Compute feature importance using `feature_importances_`
- Manually calculate feature importance from impurity reduction
- Compare manual results with Scikit-learn's implementation
- Understand how Random Forest determines the importance of each feature

---

## Dataset

Dataset used:

- Heart Failure Clinical Records Dataset

**Target Variable**

- `DEATH_EVENT`
  - `0` → Patient survived
  - `1` → Patient died

### Features

- age
- anaemia
- creatinine_phosphokinase
- diabetes
- ejection_fraction
- high_blood_pressure
- platelets
- serum_creatinine
- serum_sodium
- sex
- smoking
- time

---

## Libraries Used

```python
numpy
pandas
scikit-learn
matplotlib
```

Main Scikit-learn modules:

- RandomForestClassifier
- DecisionTreeClassifier
- StandardScaler
- train_test_split

---

## Workflow

### 1. Load Dataset

```python
df = pd.read_csv("heart.csv")
```

### 2. Separate Features and Target

```python
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]
```

### 3. Standardize Continuous Features

Continuous numerical features are standardized using `StandardScaler`.

### 4. Split the Dataset

```python
80% Training
20% Testing
```

### 5. Train Random Forest

Train the model using `RandomForestClassifier`.

### 6. Compute Feature Importance

The notebook demonstrates two approaches:

#### Method 1: Built-in Scikit-learn

```python
clf.feature_importances_
```

#### Method 2: Manual Calculation

The notebook manually computes feature importance by:

- Traversing every tree in the forest
- Calculating impurity reduction at each split
- Weighting impurity decrease by the number of samples reaching each node
- Aggregating contributions across all trees
- Normalizing the final importance values

---

## What is Feature Importance?

Feature Importance measures how much each feature contributes to reducing prediction error across all trees in a Random Forest.

Features that consistently produce large reductions in impurity receive higher importance scores.

The importance values are normalized so that their sum equals **1.0**.

---

## Mathematical Formula

For each split:

\[
Importance = \frac{N_t}{N}
\left(
I_t -
\frac{N_L}{N_t}I_L -
\frac{N_R}{N_t}I_R
\right)
\]

Where:

- \(N\) = Total training samples
- \(N_t\) = Samples at the current node
- \(N_L\) = Samples in the left child
- \(N_R\) = Samples in the right child
- \(I_t\) = Impurity of the parent node
- \(I_L\) = Impurity of the left child
- \(I_R\) = Impurity of the right child

The impurity reductions from all splits and all trees are summed and normalized to produce the final feature importance.

---

## Why Feature Importance?

Feature importance helps to:

- Identify the most influential features
- Improve model interpretability
- Perform feature selection
- Remove less useful features
- Understand how the model makes decisions

---

## Concepts Covered

- Decision Trees
- Random Forest
- Gini Impurity
- Information Gain
- Impurity Reduction
- Feature Importance
- Manual Feature Importance Calculation
- Scikit-learn Feature Importance

---

## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

3. Place the dataset

```
Dataset/
    heart.csv
```

4. Run the notebook

```bash
jupyter notebook Feature_Importance.ipynb
```

---

## Expected Output

The notebook produces:

- Feature importance using `feature_importances_`
- Manually computed feature importance
- Comparison between both approaches
- Ranking of features based on their importance

---

## License

This project is intended for educational purposes as part of a Machine Learning learning series.