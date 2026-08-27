# XGBoost Classification — Mathematical Implementation

## Overview

This project demonstrates the mathematical working of **XGBoost for binary classification** using a small dataset containing CGPA as the input feature and Placement as the binary target.

The objective is not simply to train an `XGBClassifier`, but to understand what happens mathematically during the construction of the boosting trees.

The implementation covers:

- Initial probability
- Initial raw score / log-odds
- Sigmoid function
- Gradient calculation
- Hessian calculation
- Root node score
- Candidate split generation
- Split gain calculation
- Best split selection
- Leaf output calculation
- Learning-rate update
- Updated probability
- Final class prediction
- Actual XGBoost model training
- XGBoost tree inspection

---

## Dataset

The example uses five observations:

| CGPA | Placement |
|------|-----------|
| 6.5 | 0 |
| 6.7 | 0 |
| 7.0 | 0 |
| 7.5 | 1 |
| 8.0 | 1 |

Where:

- `CGPA` is the input feature.
- `Placement = 0` represents not placed.
- `Placement = 1` represents placed.

---

## Mathematical Workflow

The classification process follows the general sequence:

```text
Dataset
   ↓
Initial Probability
   ↓
Initial Raw Score
   ↓
Sigmoid
   ↓
Probability
   ↓
Gradient (G)
   ↓
Hessian (H)
   ↓
Root Score
   ↓
Candidate Splits
   ↓
Gain Calculation
   ↓
Best Split
   ↓
Leaf Output
   ↓
Learning Rate
   ↓
Updated Raw Score
   ↓
Updated Probability
   ↓
Next Boosting Tree