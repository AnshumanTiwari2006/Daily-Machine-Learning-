# 📉 Elastic Net Regression: The Best of Both Worlds

Welcome to the **Elastic Net Regression** module! This folder brings together everything we've learned so far by combining the concepts of L1 Regularization (Lasso) and L2 Regularization (Ridge) into a single, powerful model.

## 📂 Contents & Findings

### [`ElasticNet_Regression.ipynb`](ElasticNet_Regression.ipynb)
**Goal:** Compare all four major regression techniques side-by-side to see how Elastic Net performs.

* **The Experiment:**
  * We use the `diabetes` dataset to keep our benchmark consistent.
  * We train four different models on the exact same train/test split to directly compare their $R^2$ scores:
    1. **Ordinary Linear Regression:** No penalty. (Score: `0.4399`)
    2. **Lasso Regression:** L1 penalty only. (Score: `0.4411`)
    3. **Ridge Regression:** L2 penalty only. (Score: `0.4519`)
    4. **Elastic Net Regression:** Combined L1 and L2 penalty. (Score: `0.4531` 🏆)

* **Understanding Elastic Net Parameters:**
  * We configure our Elastic Net model with `alpha=0.005` (the overall penalty strength) and `l1_ratio=0.9`. 
  * *The Magic:* The `l1_ratio=0.9` tells the model to use 90% Lasso penalty and 10% Ridge penalty. This allows the model to perform aggressive feature selection (like Lasso) while still maintaining the stability and grouping effects of Ridge Regression. 
  * *The Result:* Elastic Net slightly edges out all other models, proving that combining regularization techniques often yields the most robust predictions!

## 🚀 Getting Started
Open the notebook to see the ultimate regression showdown:
```bash
jupyter notebook
```
