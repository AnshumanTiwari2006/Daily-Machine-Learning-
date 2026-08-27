# Feature Construction (Feature Engineering)

Welcome to the **Feature Construction** module! Sometimes, the raw data provided in a dataset isn't enough for a machine learning model to easily find meaningful patterns. However, hidden within that raw data are mathematical relationships that *can* be highly predictive.

This directory focuses on the art of **Feature Construction**—the process of manually creating brand new, highly predictive features by mathematically combining, splitting, or transforming existing columns.

## Overview

Machine learning models (especially Linear and Logistic regression models) often struggle to infer complex, multi-column relationships on their own. By explicitly constructing these relationships for the model, we can drastically boost baseline accuracy.

* **Combining Features:** Adding `Siblings` and `Parents` together to create a single `Family_Size` feature.
* **Deconstructing Features:** Splitting a complex `Date_of_Birth` string into separate `Day`, `Month`, and `Year` numerical columns so the model can capture seasonality.
* **Mathematical Ratios:** Dividing `Total_Revenue` by `Total_Users` to construct an `Average_Revenue_Per_User (ARPU)` feature.

## Contents & Findings

### [`Feature Construction.ipynb`](Feature%20Construction.ipynb)
**Goal:** Mathematically prove how manually engineered features can directly improve a model's predictive power.

* **The Problem:** We use a classic dataset (like Titanic). The dataset provides separate columns for `SibSp` (Siblings/Spouses aboard) and `Parch` (Parents/Children aboard). On their own, these features provide weak, fragmented predictive signals.
* **The Construction Step:** 
  * We mathematically add `SibSp + Parch + 1` (accounting for the passenger themselves) to construct a completely new feature called `Family_Size`.
  * We then take it a step further by categorizing this numeric `Family_Size` into logical groups: *Alone*, *Small Family*, and *Large Family*.
* **The Result:** We train classification models before and after feature construction. The model trained on our manually constructed `Family_Size` categories achieves tangibly higher accuracy because we explicitly fed it the concept of "Family Dynamics," a human concept that heavily affected survival rates but wasn't obvious in the raw data.

## Getting Started
Launch the notebook to learn how to actively create your own predictive features from raw data:
```bash
jupyter notebook "Feature Construction.ipynb"
```
