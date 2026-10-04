# K-Means Clustering — Complete Explanation with Iris Dataset

## 1. What is Clustering?

Clustering is an **unsupervised machine learning** technique.

In supervised learning, we have:
- Features (`X`)
- Target/label (`y`)

In clustering, we generally only have:
- Features (`X`)

The algorithm tries to discover natural groups in the data.

Example:

```text
● ● ●          ▲ ▲ ▲          ■ ■ ■
 ● ●            ▲ ▲            ■ ■
```

The algorithm can discover that these points form three groups without being told their labels.

---

# 2. What is K-Means?

**K-Means** is an unsupervised clustering algorithm that divides data into **K clusters**.

`K` = number of clusters.

For example:

```python
KMeans(n_clusters=3)
```

means:

> Try to divide the dataset into 3 clusters.

The basic idea is:

1. Choose K.
2. Choose K initial centroids.
3. Assign every point to its nearest centroid.
4. Recalculate the centroids.
5. Repeat until the centroids stop changing significantly.

---

# 3. What is a Centroid?

A **centroid** is the center of a cluster.

Suppose a cluster contains:

```text
(1, 1)
(2, 1)
(1, 2)
```

The centroid is the mean of the coordinates:

```text
x = (1 + 2 + 1) / 3 = 1.33
y = (1 + 1 + 2) / 3 = 1.33
```

So:

```text
Centroid = (1.33, 1.33)
```

K-Means repeatedly moves these centroids until a stable solution is reached.

---

# 4. How K-Means Works

Suppose:

```text
K = 3
```

### Step 1 — Initialize 3 centroids

The algorithm starts with 3 initial centroids.

```text
C1     C2          C3
×       ×           ×
```

Modern Scikit-Learn uses **K-Means++** by default to choose good initial centroids.

---

### Step 2 — Calculate distances

For every data point, calculate its distance from each centroid.

The common distance is **Euclidean distance**:

```text
distance = √((x1-c1)² + (x2-c2)²)
```

For many features:

\[
d(x,c)=\sqrt{\sum_{j=1}^{n}(x_j-c_j)^2}
\]

The point is assigned to the nearest centroid.

---

### Step 3 — Recalculate centroids

After assigning all points:

```text
Cluster 1 → calculate its mean
Cluster 2 → calculate its mean
Cluster 3 → calculate its mean
```

These means become the new centroids.

---

### Step 4 — Repeat

```text
Assign points
     ↓
Calculate new centroids
     ↓
Assign points again
     ↓
Calculate new centroids
     ↓
Repeat
```

The algorithm stops when the centroids no longer change significantly.

This is called **convergence**.

---

# 5. Why is it called K-Means?

The name comes from:

- **K** → number of clusters
- **Means** → each centroid is the mean of the points assigned to that cluster

So:

> K-Means = K clusters whose centers are calculated using means.

---

# 6. K-Means Objective Function

K-Means tries to minimize the total squared distance between every point and its assigned centroid.

This is called **Within-Cluster Sum of Squares (WCSS)**.

\[
WCSS =
\sum_{k=1}^{K}
\sum_{x_i \in C_k}
||x_i-\mu_k||^2
\]

Where:

- `K` = number of clusters
- `Ck` = cluster k
- `xi` = data point
- `μk` = centroid of cluster k
- `||xi - μk||²` = squared distance from point to centroid

### Simple meaning

> K-Means tries to keep points as close as possible to the center of their own cluster.

Lower WCSS generally means tighter clusters.

---

# 7. What is the Iris Dataset?

We will use the built-in **Iris dataset** from Scikit-Learn.

It contains:

- 150 samples
- 4 numerical features
- 3 actual species

The four features are:

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

The three species are:

- Setosa
- Versicolor
- Virginica

For clustering, we will **not use the species as the target**.

We only use the four features.

---

# 8. Importing Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

### `pandas`

Used for working with tabular data.

```python
import pandas as pd
```

We will use it to create a DataFrame.

### `matplotlib`

Used for visualization.

```python
import matplotlib.pyplot as plt
```

We use it to plot the Elbow curve.

### `load_iris`

Loads the built-in Iris dataset.

```python
from sklearn.datasets import load_iris
```

### `KMeans`

Imports the K-Means algorithm.

```python
from sklearn.cluster import KMeans
```

### `StandardScaler`

Used for feature scaling.

```python
from sklearn.preprocessing import StandardScaler
```

---

# 9. Loading the Dataset

```python
iris = load_iris()
```

This loads the Iris dataset into the variable `iris`.

Now:

```python
X = iris.data
```

stores the four numerical features.

We can check its shape:

```python
print(X.shape)
```

Output:

```text
(150, 4)
```

Meaning:

```text
150 rows
4 features
```

---

# 10. Why Do We Scale the Data?

K-Means is **distance-based**.

Suppose we have:

```text
Age       → 18 to 60
Salary    → 20,000 to 2,00,000
```

Salary has much larger numerical values.

Therefore, it can dominate the distance calculation.

Scaling puts features on a comparable scale.

---

# 11. StandardScaler

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

`StandardScaler` standardizes every feature.

The formula is:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- `x` = original value
- `μ` = mean of the feature
- `σ` = standard deviation

After scaling:

- Mean ≈ 0
- Standard deviation ≈ 1

---

# 12. `fit_transform()`

This line:

```python
X_scaled = scaler.fit_transform(X)
```

does two things.

### `fit()`

Learns the mean and standard deviation from the data.

### `transform()`

Uses those learned values to scale the data.

Therefore:

```python
fit_transform()
```

means:

> Learn the scaling parameters and transform the data.

---

# 13. What is the Elbow Method?

The biggest question in K-Means is:

> How do we decide the value of K?

We usually don't know the correct number of clusters beforehand.

The **Elbow Method** helps us choose K.

We train K-Means with different values:

```text
K = 1
K = 2
K = 3
K = 4
...
K = 10
```

For each K, we calculate WCSS.

Then we plot:

```text
K → WCSS
```

---

# 14. Why Does WCSS Decrease When K Increases?

Imagine:

```text
K = 1
```

All points belong to one cluster.

The centroid is far from many points.

So WCSS is large.

Now:

```text
K = 2
```

There are two centroids.

Points can be closer to their assigned centroid.

WCSS decreases.

With:

```text
K = 3
```

it decreases again.

This continues as K increases.

Eventually, adding more clusters provides only a small improvement.

That is where we look for the **elbow**.

---

# 15. Creating the WCSS List

```python
wcss = []
```

This creates an empty list.

We will store the WCSS value for every K.

For example:

```text
K = 1 → WCSS = 500
K = 2 → WCSS = 300
K = 3 → WCSS = 180
...
```

---

# 16. The `for` Loop

```python
for i in range(1, 11):
```

This generates:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

So we test K from 1 to 10.

---

# 17. Creating the K-Means Model

```python
kmeans = KMeans(
    n_clusters=i,
    random_state=42,
    n_init=10
)
```

### `n_clusters=i`

This tells K-Means how many clusters to create.

For example:

```python
i = 3
```

means:

```python
n_clusters=3
```

---

## `random_state=42`

K-Means involves random initialization.

Setting:

```python
random_state=42
```

makes the result reproducible.

The number `42` itself has no special mathematical meaning.

You could use another fixed number.

---

## `n_init=10`

K-Means can produce different results depending on its initial centroids.

`n_init=10` means the algorithm tries multiple initializations and keeps the best result.

This makes the solution more reliable.

---

# 18. Fitting K-Means

```python
kmeans.fit(X_scaled)
```

This trains the K-Means algorithm on the scaled data.

Internally, it performs the process:

```text
Initialize centroids
       ↓
Calculate distances
       ↓
Assign points
       ↓
Calculate new centroids
       ↓
Repeat until convergence
```

---

# 19. What is `inertia_`?

After fitting:

```python
kmeans.inertia_
```

contains the **inertia**.

In K-Means, inertia is the sum of squared distances from each point to its nearest centroid.

It is essentially the WCSS objective.

Therefore:

```python
wcss.append(kmeans.inertia_)
```

stores the WCSS for the current K.

---

# 20. Complete Elbow Loop

```python
wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)
```

At the end:

```python
wcss
```

contains 10 WCSS values.

One value for each:

```text
K = 1 → WCSS
K = 2 → WCSS
...
K = 10 → WCSS
```

---

# 21. Plotting the Elbow Curve

```python
plt.plot(range(1, 11), wcss, marker='o')

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()
```

### `range(1, 11)`

Provides the K values:

```text
1 through 10
```

### `wcss`

Provides the corresponding WCSS values.

### `marker='o'`

Places a circle on each data point.

### `xlabel()`

Names the X-axis.

### `ylabel()`

Names the Y-axis.

### `title()`

Adds the graph title.

### `show()`

Displays the graph.

---

# 22. Understanding the Elbow

The graph will generally look something like:

```text
WCSS
 ↑
 |\
 | \
 |  \
 |   \
 |    \__
 |       \___
 |
 +----------------→ K
       2  3  4  5
```

The curve falls quickly at first.

Then it starts flattening.

The point where this change happens is the **elbow**.

For Iris, K = 3 is a reasonable choice.

Therefore:

```python
K = 3
```

---

# 23. Applying K-Means with K = 3

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

Now we have selected our final number of clusters.

We fit the model and obtain cluster labels:

```python
clusters = kmeans.fit_predict(X_scaled)
```

---

# 24. What is `fit_predict()`?

`fit_predict()` performs two operations:

### Step 1

Fits the K-Means model.

### Step 2

Returns the cluster assigned to every sample.

Conceptually:

```python
kmeans.fit(X_scaled)

clusters = kmeans.predict(X_scaled)
```

is similar to:

```python
clusters = kmeans.fit_predict(X_scaled)
```

---

# 25. What are Cluster Labels?

The result may look like:

```text
[1, 1, 1, 1, 1, ..., 0, 0, ..., 2, 2, ...]
```

The labels:

```text
0
1
2
```

represent the three clusters.

Important:

> Cluster `0` does not inherently mean Iris Setosa, and cluster `1` does not inherently mean Versicolor.

Cluster numbers are simply identifiers assigned by the algorithm.

---

# 26. Creating a DataFrame

```python
df = pd.DataFrame(
    X,
    columns=iris.feature_names
)
```

This converts the feature matrix into a Pandas DataFrame.

The columns become:

```text
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
```

---

# 27. Adding the Cluster Column

```python
df["Cluster"] = clusters
```

This adds the cluster assigned to every sample.

The DataFrame now looks conceptually like:

```text
sepal length | sepal width | petal length | petal width | Cluster
-------------------------------------------------------------------
5.1          | 3.5         | 1.4          | 0.2         | 1
4.9          | 3.0         | 1.4          | 0.2         | 1
...
```

---

# 28. Complete Code

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load Iris dataset
iris = load_iris()

X = iris.data


# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Elbow Method
wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)


# Plot Elbow Curve
plt.plot(range(1, 11), wcss, marker='o')

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()


# Apply K-Means with K = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)


# Create DataFrame
df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["Cluster"] = clusters

print(df.head())
```

---

# 29. The Complete Mental Model

Think of the entire process as:

```text
Iris Dataset
     ↓
Get 4 features
     ↓
Scale features
     ↓
Try K = 1, 2, 3, ... 10
     ↓
Calculate WCSS for every K
     ↓
Plot WCSS vs K
     ↓
Find the Elbow
     ↓
Choose K ≈ 3
     ↓
Train final K-Means
     ↓
Assign every point to a cluster
```

---

# 30. Important Parameters

## `n_clusters`

Number of clusters.

```python
KMeans(n_clusters=3)
```

---

## `init`

Controls how initial centroids are selected.

```python
init="k-means++"
```

K-Means++ is the standard choice in Scikit-Learn.

---

## `n_init`

Number of different centroid initializations tried.

```python
n_init=10
```

The best result is retained.

---

## `max_iter`

Maximum number of iterations for one K-Means run.

Example:

```python
KMeans(
    n_clusters=3,
    max_iter=300
)
```

If convergence doesn't happen before this limit, the algorithm stops.

---

## `random_state`

Controls randomness so results can be reproduced.

```python
random_state=42
```

---

# 31. Important K-Means Problems

## 1. Choosing K

The algorithm doesn't automatically know the correct K.

Common methods:

- Elbow Method
- Silhouette Score

---

## 2. Scaling

Because K-Means uses distance, features with very different scales can distort the result.

Standardization is often useful.

---

## 3. Outliers

Extreme points can pull centroids away from the main data.

---

## 4. Cluster Shape

K-Means generally works best when clusters are relatively compact and roughly spherical.

It may struggle with complicated shapes.

---

# 32. K-Means vs KNN

Do not confuse these algorithms.

| K-Means | KNN |
|---|---|
| Unsupervised | Supervised |
| Clustering | Classification / Regression |
| No target required | Target required |
| Finds groups | Predicts output |
| Learns centroids | Uses neighboring training points |
| K = number of clusters | K = number of neighbors |

The only similarity is that both use distance in common implementations.

---

# 33. Interview Definition

If asked:

**"What is K-Means?"**

Answer:

> K-Means is an unsupervised clustering algorithm that partitions data into K clusters by repeatedly assigning each point to its nearest centroid and then updating each centroid as the mean of the points assigned to it.

If asked:

**"What is the Elbow Method?"**

Answer:

> The Elbow Method is used to select a suitable number of clusters by plotting WCSS against different values of K and choosing the point where the reduction in WCSS starts becoming significantly smaller.

---

# 34. One-Line Summary

> **K-Means finds K groups by minimizing the squared distance between data points and their assigned cluster centroids.**
