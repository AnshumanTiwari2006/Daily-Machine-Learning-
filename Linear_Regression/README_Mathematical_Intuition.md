# Linear Regression: Mathematical Intuition from Scratch 🧮

This module walks through the underlying mathematical formulation of **Simple 1D Linear Regression** and implements it from first principles using standard Python, NumPy, and Pandas (without relying on `scikit-learn` for the model).

---

## 📐 Mathematical Formulation

Simple Linear Regression models the relationship between a single independent variable $x$ and a dependent variable $y$ by fitting a linear equation to the observed data:

$$y = mx + b$$

Where:
*   **$y$**: Dependent Variable (Target)
*   **$x$**: Independent Variable (Feature)
*   **$m$**: Slope of the line (Coefficient)
*   **$b$**: Y-intercept

### 1. Estimating the Slope ($m$)
Using the **Ordinary Least Squares (OLS)** method, we find the line that minimizes the sum of squared residuals. The slope $m$ is derived as:

$$m = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

*   $\bar{x}$ represents the mean of $X$.
*   $\bar{y}$ represents the mean of $Y$.
*   The numerator calculates the covariance between $x$ and $y$.
*   The denominator calculates the variance of $x$.

### 2. Estimating the Intercept ($b$)
Once the slope $m$ is obtained, the intercept $b$ is computed such that the best fit line passes through the centroid $(\bar{x}, \bar{y})$ of the data:

$$b = \bar{y} - m\bar{x}$$

---

## 💻 Class Implementation: `Algo_of_LR`

The custom algorithm is implemented in Python as follows:

```python
class Algo_of_LR:
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self, X_train, y_train):
        num = 0
        den = 0
        # Iterate over all training points to compute covariance and variance
        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))
        
        # Calculate optimal parameters
        self.m = num / den
        self.b = y_train.mean() - (self.m * X_train.mean())
        
        print(f"Slope (m): {self.m}")
        print(f"Intercept (b): {self.b}")
        
    def predict(self, X_test):        
        return self.m * X_test + self.b
```

---

## 📊 Dataset & Model Evaluation

The model was trained on the weather dataset:
*   **Independent Feature ($X$)**: `Temperature (C)`
*   **Dependent Target ($y$)**: `Humidity`

### Custom Fit Parameters:
*   **Slope ($m$)**: `-1.2187856150797678` (An increase of $1^\circ\text{C}$ in temperature corresponds to approximately $1.22\%$ drop in humidity)
*   **Intercept ($b$)**: `100.72745838686504`

### Sample Prediction:
*   **Input Temperature**: `23.89 °C`
*   **Mathematical Calculation**: 
    $$y = (-1.21878 \times 23.89) + 100.72746 \approx 71.61\%$$
*   **Predicted Humidity**: `71.61%`
