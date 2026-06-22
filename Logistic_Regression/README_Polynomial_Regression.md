# 📉 Polynomial Regression: Fitting Curves and 3D Surfaces

Welcome to the **Polynomial Regression** module! This notebook demonstrates how to capture complex, non-linear relationships by combining `PolynomialFeatures` with standard `LinearRegression`.

## 📂 Contents & Findings

### [`Polynomial_Regression.ipynb`](Polynomial_Regression.ipynb)
**Goal:** Learn how to transform linear models to fit quadratic curves (2D) and complex surfaces (3D).

* **2D Polynomial Regression (Fitting a Curve):**
  * **Data Generation:** We generate synthetic data following a quadratic equation: $y = 0.5x^2 + 0.6x + 2$ with added random noise.
  * **The Polynomial Transformation:** Standard Linear Regression can only draw straight lines. To fix this, we use `PolynomialFeatures(degree=2, include_bias=True)`. This takes our single feature `x` and transforms it into three features: `[1, x, x^2]`.
  * **Training & Visualization:** We fit a `LinearRegression` model on these new features. The notebook plots the final predictions, showing a beautiful red curve perfectly tracking the blue training points and green testing points!

* **3D Polynomial Regression (Fitting a Surface):**
  * **Data Generation:** We level up to 3D by generating $x$ and $y$, and setting $z = x^2 + y^2 + 0.2x + 0.2y + 0.1xy + 2$ plus noise.
  * **The Multi-Variable Transformation:** We pass `[x, y]` into `PolynomialFeatures(degree=2)`. The notebook outputs exactly how the features expand from 2 inputs to 6 outputs: `[1, x, y, x^2, xy, y^2]`.
  * **Interactive 3D Visualization:** We train the model on these 6 features to predict $z$. Using `plotly.express` and `plotly.graph_objects`, we render an interactive 3D scatter plot of our data overlayed with the curved 3D surface (`go.Surface`) our model learned!

## 🚀 Getting Started
Launch the notebook to interact with the 3D surface plot:
```bash
jupyter notebook
```
