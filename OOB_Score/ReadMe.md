# Out-of-Bag (OOB) Evaluation in Random Forest

This notebook demonstrates how to use **Out-of-Bag (OOB) Evaluation** in a Random Forest Classifier using the **Heart Failure Clinical Records Dataset**.

## Objective

The goal of this notebook is to:

- Train a Random Forest Classifier
- Enable Out-of-Bag (OOB) scoring
- Compare the OOB Score with the Test Accuracy
- Understand how OOB evaluation works without requiring a separate validation set

---

## Dataset

Dataset used:

- **Heart Failure Clinical Records Dataset**
- Target Variable:
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

## 🛠 Libraries Used

```python
numpy
pandas
scikit-learn
```

Main sklearn modules:

- RandomForestClassifier
- StandardScaler
- train_test_split
- accuracy_score

---

## ⚙ Workflow

### 1. Load Dataset

```python
df = pd.read_csv("heart.csv")
```

---

### 2. Separate Features and Target

```python
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]
```

---

### 3. Standardize Continuous Features

Continuous columns are standardized using **StandardScaler**.

```python
age
creatinine_phosphokinase
ejection_fraction
platelets
serum_creatinine
serum_sodium
time
```

---

### 4. Train-Test Split

```python
80% Training
20% Testing
```

Random state:

```python
42
```

---

### 5. Train Random Forest

OOB evaluation is enabled using

```python
RandomForestClassifier(oob_score=True)
```

---

### 6. Evaluate Model

The notebook computes:

- OOB Score
- Test Accuracy

```python
rf.oob_score_

accuracy_score(y_test, y_pred)
```

---

# What is Out-of-Bag (OOB) Evaluation?

Random Forest uses **Bootstrap Sampling**, meaning every decision tree is trained using a random sample (with replacement) from the training data.

Since sampling is done with replacement:

- Some observations are selected multiple times.
- Some observations are never selected.

Those unselected observations are called **Out-of-Bag (OOB) samples**.

These OOB samples act as an **internal validation set** for each tree.

After training:

- Every sample is predicted using only the trees that never saw it.
- These predictions are combined.
- The resulting accuracy is called the **OOB Score**.

---

# Why Use OOB Score?

Advantages:

- No need for an additional validation dataset
- Makes better use of available training data
- Provides an unbiased estimate of model performance
- Especially useful for small datasets

---

# Machine Learning Concepts Covered

- Bootstrap Sampling
- Random Forest Classifier
- Out-of-Bag Evaluation
- Feature Scaling
- Train-Test Split
- Model Accuracy

---

# How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install numpy pandas scikit-learn
```

3. Place the dataset

```
Dataset/
    heart.csv
```

4. Run the notebook

```bash
jupyter notebook OOB_Evaluation_in_Random_Forest.ipynb
```

---

# Expected Output

The notebook displays:

- OOB Score
- Test Accuracy

These values can be compared to understand how closely the OOB estimate matches the model's performance on unseen test data.

---

# License

This project is intended for educational purposes and machine learning practice.