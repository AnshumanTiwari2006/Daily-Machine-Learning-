# 📊 Classification Metrics & Evaluation

Welcome to the **Classification Metrics** module! Building a machine learning model is only half the battle; knowing how to properly evaluate its real-world performance is equally critical. 

This directory focuses on the foundational techniques used to measure the success of classification models, especially when distinguishing between different types of errors.

## 🎯 Overview

When predicting discrete categories (like classifying an email as spam or a tumor as malignant), relying solely on "Accuracy" can be dangerously misleading—especially if your dataset is imbalanced or if one type of error is far more costly than the other. 

This module breaks down the essential tools required to look past raw accuracy:
1. **Accuracy Score**
2. **The Confusion Matrix**
3. **Type I Errors (False Positives) vs. Type II Errors (False Negatives)**

## 📂 Contents & Findings

### [`Accuracy_Confusion_Matrix_Type1_Type2.ipynb`](Accuracy_Confusion_Matrix_Type1_Type2.ipynb)
**Goal:** Understand the 4 quadrants of the Confusion Matrix and analyze how different algorithms balance Type I and Type II errors.

* **The Dataset:** We use the classic `breast_cancer` dataset from Scikit-Learn. This medical context is the perfect real-world example of why understanding the *type* of error matters far more than overall accuracy.

* **Accuracy Score:**
  * The percentage of total predictions that the model got correct. 
  * *Formula:* $\frac{TP + TN}{TP + TN + FP + FN}$

* **The Confusion Matrix:**
  * A $2 \times 2$ grid that maps out exactly where the model succeeded and where it failed.
  * **True Positives (TP):** Model predicted positive, and it actually was positive.
  * **True Negatives (TN):** Model predicted negative, and it actually was negative.
  
* **Type I vs. Type II Errors:**
  * **Type I Error (False Positive - FP):** The model predicted a condition exists when it doesn't (e.g., diagnosing a healthy patient with cancer). 
  * **Type II Error (False Negative - FN):** The model completely missed the condition (e.g., telling a sick patient they are healthy). In medical datasets, minimizing Type II errors is usually the absolute top priority.

* **Model Comparison:** 
  * We train both a **Logistic Regression** model and a **Decision Tree Classifier**.
  * By printing their respective Confusion Matrices, we analyze not just which model boasts a higher accuracy, but which model makes safer, less costly mistakes.

## 🚀 Getting Started
Launch the notebook to see how to generate and interpret these critical metrics in Python:
```bash
jupyter notebook Accuracy_Confusion_Matrix_Type1_Type2.ipynb
```
