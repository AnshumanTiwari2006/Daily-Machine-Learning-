# 📈 ROC Curve & AUC Score

Welcome to the **ROC Curve** module! When evaluating a classification model, especially one that outputs probabilities, looking at a single decision threshold (like 0.5) rarely gives you the full picture. 

This directory focuses on evaluating how well a model can distinguish between classes across *all* possible classification thresholds.

## 🎯 Overview

The ROC (Receiver Operating Characteristic) Curve and the AUC (Area Under the Curve) score are essential graphical and numerical tools for evaluating binary classification models.

* **ROC Curve:** A graph showing the performance of a classification model at all possible thresholds. It plots two key parameters:
  * **True Positive Rate (TPR):** Also known as Sensitivity or Recall. It measures the proportion of actual positives correctly identified ($\frac{TP}{TP + FN}$).
  * **False Positive Rate (FPR):** It measures the proportion of actual negatives incorrectly identified as positive ($\frac{FP}{FP + TN}$).
* **AUC Score:** Measures the entire two-dimensional area underneath the ROC curve from $(0,0)$ to $(1,1)$. It provides an aggregate measure of performance. Conceptually, it represents the probability that the model ranks a random positive example more highly than a random negative example. A perfect model has an AUC of $1.0$, while a random-guessing model has an AUC of $0.5$.

## 📂 Contents & Findings

### [`ROC-AUC.ipynb`](ROC-AUC.ipynb)
**Goal:** Understand how to compute, plot, and interpret the ROC Curve and calculate the AUC score using a real-world dataset.

* **The Dataset:** We use the `breast_cancer` dataset from Scikit-Learn to predict whether a tumor is benign or malignant.
* **Model Training:** We train a `LogisticRegression` model, which naturally outputs probabilities for its predictions.
* **Generating the Curve:** 
  * Instead of predicting hard classes (`0` or `1`), we use `predict_proba()` to get the raw probabilities of the positive class.
  * We use Scikit-Learn's `roc_curve` to calculate the TPR and FPR at various threshold cutoffs.
  * We use `RocCurveDisplay` to plot the curve, visualizing the critical trade-off between capturing True Positives and falsely triggering False Positives.
* **Calculating AUC:** We compute the `roc_auc_score` to get a single, unified metric of the model's diagnostic ability.

## 🚀 Getting Started
Launch the notebook to see the exact code for generating and interpreting ROC curves in Python:
```bash
jupyter notebook ROC-AUC.ipynb
```
