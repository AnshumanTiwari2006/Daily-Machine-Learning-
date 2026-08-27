# Encoders: Handling Categorical Data

Welcome to the **Encoders** module! Machine learning models are fundamentally mathematical algorithms—they only understand numbers. When working with real-world datasets, we frequently encounter categorical (text-based) data like "Red", "Blue", or "High", "Medium", "Low".

This directory focuses on exactly how to mathematically translate these text-based categories into numeric formats that machine learning algorithms can successfully process, without accidentally introducing false numerical relationships.

## Overview

Categorical data generally falls into two distinct types, and each requires a specific encoding strategy:

1. **Ordinal Data:** Categories that have a strict, logical order or ranking (e.g., *Poor < Average < Excellent*). We use **Ordinal Encoding** to assign increasing integers that preserve this mathematical hierarchy.
2. **Nominal Data:** Categories with absolutely no inherent mathematical order (e.g., *Red, Green, Blue* or *New York, London, Tokyo*). We use **One-Hot Encoding** to prevent the model from assuming that *Tokyo (3)* is mathematically larger or "better" than *New York (1)*.

## Contents & Findings

### 1. [`(Ordinal)Encoding Categorical Data.ipynb`](%28Ordinal%29Encoding%20Categorical%20Data.ipynb)
**Goal:** Translate ordered text features into ordered integers.
* **The Logic:** For features where order inherently matters (like education level, income brackets, or customer satisfaction), we explicitly map strings to integers so the model can grasp the hierarchy.
* **Scikit-Learn Tools:**
  * **`OrdinalEncoder`:** Used for encoding independent variables ($X$). We map categories to integers (e.g., 0, 1, 2) while explicitly defining the expected rank order.
  * **`LabelEncoder`:** A specific utility strictly designed for encoding the target variable ($y$). It automatically maps text-based classes (like "Yes"/"No" or "Malignant"/"Benign") to numeric IDs.

### 2. [`One Hot Encoding.ipynb`](One%20Hot%20Encoding.ipynb)
**Goal:** Translate unordered text features into binary vectors to prevent false numerical relationships.
* **The Problem:** If we Ordinal-Encode nominal data like car brands (*Ford=1, Honda=2, Toyota=3*), algorithms that use weights (Linear Regression) or distances (KNN) will falsely assume that *Toyota* is mathematically three times as impactful as *Ford*.
* **The Solution:** One-Hot Encoding creates a new, independent binary column (0 or 1) for every single unique category.
* **The Dummy Variable Trap:** Creating a column for every single category introduces perfect multicollinearity (e.g., if a coin is not Heads, it *must* be Tails; we don't need two columns to represent this). We learn how to mathematically resolve this by dropping the first column (`drop='first'`).
* **Implementation:** We compare the quick, exploratory Pandas approach (`pd.get_dummies()`) against Scikit-Learn's highly robust, pipeline-ready `OneHotEncoder`.

## Getting Started
Launch the notebooks to master how to feed categorical text data into mathematical models:
```bash
jupyter notebook "One Hot Encoding.ipynb"
```
