# Stacking Classifier — Heart Disease Dataset

## 1. Import Libraries

```python
import numpy as np
import pandas as pd
```

---

## 2. Load the Dataset

```python
df = pd.read_csv('heart.csv')
```

View the first few rows:

```python
df.head()
```

The dataset contains **303 rows** and **14 columns**:

- 13 input features
- 1 target column

The target is:

```text
target
```

---

## 3. Separate X and y

```python
X = df.drop(columns=['target'])
y = df['target']
```

### X — Input Features

`X` contains the 13 features:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

Shape:

```text
303 rows × 13 columns
```

### y — Target

```text
target = 0 or 1
```

This is a **binary classification problem**.

---

## 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=8
)
```

We keep:

```text
80% → Training data
20% → Testing data
```

Since the dataset has 303 records:

```text
Training → 242 samples
Testing  → 61 samples
```

Check:

```python
print(X_train.shape)
```

Output:

```text
(242, 13)
```

The **61 test samples remain unseen** during model training.

---

# 5. Import the Base Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
```

We will use 3 base models:

1. Random Forest
2. KNN
3. Gradient Boosting

And one meta-model:

```text
Logistic Regression
```

---

# 6. Create the Base Models

```python
estimators = [
    (
        'rf',
        RandomForestClassifier(
            n_estimators=10,
            random_state=42
        )
    ),

    (
        'knn',
        KNeighborsClassifier(
            n_neighbors=10
        )
    ),

    (
        'gbdt',
        GradientBoostingClassifier()
    )
]
```

Conceptually:

```text
                    Training Data
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     Random Forest      KNN      Gradient Boosting
```

---

# 7. Create the Stacking Classifier

```python
from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=10
)
```

Here:

```text
estimators
    ↓
Base Models

final_estimator
    ↓
Meta Model

cv=10
    ↓
10-Fold Cross-Validation
```

---

# 8. What Happens Internally?

We have:

```text
242 Training Samples
```

Because:

```python
cv=10
```

the training data is divided into **10 folds** internally.

Approximately:

```text
242 / 10 ≈ 24 samples per fold
```

For each base model, StackingClassifier generates **out-of-fold (OOF) predictions**.

For example:

```text
                 10-Fold CV
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
         RF         KNN        GBDT
          │          │          │
          ↓          ↓          ↓
       OOF Pred    OOF Pred   OOF Pred
          │          │          │
          └──────────┼──────────┘
                     ↓
              Meta Model
          Logistic Regression
```

The meta-model learns from the predictions of the three base models.

In simplified form:

```text
RF Prediction
KNN Prediction
GBDT Prediction
       ↓
[RF, KNN, GBDT]
       ↓
Logistic Regression
       ↓
Final Prediction
```

### Why use Cross-Validation?

The meta-model should not learn from predictions made on samples that the corresponding base model has already trained on.

Cross-validation allows us to generate **out-of-fold predictions**, reducing this data-leakage problem.

---

# 9. Train the Stacking Model

```python
clf.fit(X_train, y_train)
```

At this point, Scikit-learn handles the stacking process internally:

1. Performs 10-fold CV on the training data.
2. Generates OOF predictions from the base models.
3. Uses those predictions to train the meta-model.
4. Retrains the base models using the full training dataset.

---

# 10. Predict on Test Data

```python
y_pred = clf.predict(X_test)
```

The 61 previously unseen test samples are passed through the trained stacking system:

```text
X_test
   │
   ├── Random Forest
   ├── KNN
   └── Gradient Boosting
          │
          ↓
     Meta Model
     (Logistic Regression)
          │
          ↓
     Final Prediction
```

---

# 11. Evaluate the Model

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

Output:

```text
0.8688524590163934
```

Therefore:

```text
Accuracy ≈ 86.89%
```

The model correctly classified approximately **86.89% of the 61 test samples**.

---

# 12. Complete Code

```python
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('heart.csv')

# Separate input and target
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=8
)

# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Base models
estimators = [
    (
        'rf',
        RandomForestClassifier(
            n_estimators=10,
            random_state=42
        )
    ),
    (
        'knn',
        KNeighborsClassifier(
            n_neighbors=10
        )
    ),
    (
        'gbdt',
        GradientBoostingClassifier()
    )
]

# Stacking classifier
from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=10
)

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

---

# 13. Final Workflow

```text
303 Total Samples
       │
       ├───────────────┐
       ↓               ↓
242 Training        61 Testing
       │               │
       │               │
    CV = 10             │
       │               │
       ↓               │
 ┌─────┼─────┐         │
 ↓     ↓     ↓          │
 RF   KNN   GBDT        │
 └─────┼─────┘         │
       ↓               │
  OOF Predictions      │
       ↓               │
 Logistic Regression   │
    (Meta Model)       │
       │               │
       └───────┐       │
               ↓       ↓
             Final Test Prediction
                    │
                    ↓
               Accuracy
                    │
                 86.89%
```

## Key Takeaway

> **Stacking combines multiple base models by using their predictions as inputs to a meta-model. Cross-validation (`cv=10`) is used to generate out-of-fold predictions for training the meta-model, while the final evaluation is performed on the completely held-out test set.**
