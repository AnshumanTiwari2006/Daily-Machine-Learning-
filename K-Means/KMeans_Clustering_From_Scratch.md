# K-Means Clustering — From Scratch

This notebook/project implements **K-Means Clustering from scratch** using Python, NumPy, Pandas, and Matplotlib.

The example uses the `student_clustering.csv` dataset and follows the same flow as the CampusX implementation.

---

## 1. What is K-Means?

K-Means is an **unsupervised machine learning algorithm** used to divide data into `K` groups called **clusters**.

The algorithm tries to put:

- Similar data points → into the same cluster
- Different data points → into different clusters

For example, if we have student data with two features, K-Means can discover groups of students having similar feature values.

---

# 2. Import Libraries

```python
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Why these libraries?

- `random` → used to randomly select initial centroids
- `numpy` → numerical calculations and arrays
- `pandas` → reading and manipulating CSV data
- `matplotlib` → plotting the clusters

---

# 3. Load the Dataset

Make sure `student_clustering.csv` is in the same folder as your Python file/notebook.

```python
df = pd.read_csv('student_clustering.csv')

df.head()
```

We can inspect the dataset:

```python
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nData types:")
print(df.dtypes)
```

---

# 4. Convert DataFrame to NumPy Array

K-Means will work with the numerical feature matrix.

```python
X = df.iloc[:, :].values

print("X shape:", X.shape)
print(X[:5])
```

### Understanding this line

```python
df.iloc[:, :]
```

means:

- `:` before the comma → select all rows
- `:` after the comma → select all columns

Then:

```python
.values
```

converts the Pandas DataFrame into a NumPy array.

---

# 5. K-Means Algorithm

K-Means follows these basic steps:

1. Choose the number of clusters `K`.
2. Randomly select `K` points as initial centroids.
3. Calculate the distance between every point and every centroid.
4. Assign each point to its nearest centroid.
5. Calculate the mean of every cluster.
6. Move the centroids to these new means.
7. Repeat the process.
8. Stop when the centroids stop changing or the maximum number of iterations is reached.

---

# 6. Euclidean Distance

K-Means commonly uses **Euclidean distance**.

The formula is:

$$
d(x,c) = \sqrt{\sum_i (x_i-c_i)^2}
$$

For two-dimensional points:

$$
d = \sqrt{(x_1-c_1)^2 + (x_2-c_2)^2}
$$

For example:

```text
Point      = (2, 3)
Centroid   = (5, 7)
```

Then:

$$
d = \sqrt{(2-5)^2 + (3-7)^2}
$$

$$
d = \sqrt{9 + 16}
$$

$$
d = 5
$$

The point is assigned to whichever centroid has the smallest distance.

---

# 7. Implement K-Means From Scratch

```python
class KMeans:

    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):

        # Step 1:
        # Randomly select K data points as initial centroids
        random_index = random.sample(
            range(0, X.shape[0]),
            self.n_clusters
        )

        self.centroids = X[random_index]

        # Repeat the K-Means process
        for i in range(self.max_iter):

            # Step 2 & 3:
            # Assign every point to its nearest centroid
            cluster_group = self.assign_clusters(X)

            # Store old centroids
            old_centroids = self.centroids.copy()

            # Step 4:
            # Move centroids to the mean of their clusters
            self.centroids = self.move_centroids(
                X,
                cluster_group
            )

            # Step 5:
            # Stop if centroids no longer change
            if np.array_equal(old_centroids, self.centroids):
                break

        return cluster_group

    def assign_clusters(self, X):

        cluster_group = []

        for row in X:

            distances = []

            # Calculate distance from current point
            # to every centroid
            for centroid in self.centroids:

                distance = np.sqrt(
                    np.dot(
                        row - centroid,
                        row - centroid
                    )
                )

                distances.append(distance)

            # Find the closest centroid
            min_distance = min(distances)

            index_pos = distances.index(min_distance)

            # Store the cluster number
            cluster_group.append(index_pos)

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):

        new_centroids = []

        # Find unique cluster numbers
        cluster_type = np.unique(cluster_group)

        for cluster in cluster_type:

            # Select points belonging to this cluster
            cluster_points = X[cluster_group == cluster]

            # Calculate the mean of the cluster
            cluster_mean = cluster_points.mean(axis=0)

            new_centroids.append(cluster_mean)

        return np.array(new_centroids)
```

---

# 8. Understanding `__init__()`

```python
def __init__(self, n_clusters=2, max_iter=100):
```

The constructor receives two parameters:

### `n_clusters`

Number of clusters we want.

For example:

```python
KMeans(n_clusters=4)
```

means:

> Divide the dataset into 4 clusters.

### `max_iter`

Maximum number of iterations.

For example:

```python
max_iter=500
```

means K-Means can perform at most 500 iterations.

---

# 9. Selecting Initial Centroids

Inside `fit_predict()`:

```python
random_index = random.sample(
    range(0, X.shape[0]),
    self.n_clusters
)
```

This randomly selects `K` row indices from the dataset.

Then:

```python
self.centroids = X[random_index]
```

uses those selected data points as the initial centroids.

For example, if:

```text
K = 4
```

then four random points are initially selected as:

```text
Centroid 0
Centroid 1
Centroid 2
Centroid 3
```

---

# 10. Assigning Clusters

This happens here:

```python
cluster_group = self.assign_clusters(X)
```

For every data point, we calculate its distance from every centroid.

Example:

```text
Point P

Distance to Centroid 0 = 4.5
Distance to Centroid 1 = 2.1
Distance to Centroid 2 = 8.3
Distance to Centroid 3 = 5.2
```

The smallest distance is:

```text
2.1
```

Therefore:

```text
Point P → Cluster 1
```

---

# 11. The `assign_clusters()` Function

```python
def assign_clusters(self, X):

    cluster_group = []

    for row in X:

        distances = []

        for centroid in self.centroids:

            distance = np.sqrt(
                np.dot(
                    row - centroid,
                    row - centroid
                )
            )

            distances.append(distance)

        min_distance = min(distances)

        index_pos = distances.index(min_distance)

        cluster_group.append(index_pos)

    return np.array(cluster_group)
```

### Important variables

`row`:

> Current data point.

`centroid`:

> One of the current cluster centers.

`distances`:

> Stores the distance from the current point to every centroid.

`index_pos`:

> Index of the nearest centroid.

`cluster_group`:

> Stores the cluster assignment of every data point.

---

# 12. Moving the Centroids

After assigning the points to clusters, we calculate the mean of every cluster.

```python
self.centroids = self.move_centroids(
    X,
    cluster_group
)
```

The basic idea is:

```text
Cluster 0
   ↓
Find all its points
   ↓
Calculate mean
   ↓
New Centroid 0
```

The same happens for every cluster.

---

# 13. `move_centroids()`

```python
def move_centroids(self, X, cluster_group):

    new_centroids = []

    cluster_type = np.unique(cluster_group)

    for cluster in cluster_type:

        cluster_points = X[cluster_group == cluster]

        cluster_mean = cluster_points.mean(axis=0)

        new_centroids.append(cluster_mean)

    return np.array(new_centroids)
```

This line:

```python
np.unique(cluster_group)
```

finds the cluster IDs.

For four clusters, it could return:

```text
[0, 1, 2, 3]
```

Then:

```python
X[cluster_group == cluster]
```

selects all points belonging to that cluster.

Finally:

```python
cluster_points.mean(axis=0)
```

calculates the new centroid.

---

# 14. Convergence

After calculating new centroids, we compare them with the previous centroids.

```python
if np.array_equal(old_centroids, self.centroids):
    break
```

If they are identical:

> The centroids have stopped moving.

Therefore, the algorithm has converged and we can stop.

If they are different:

> Continue to the next iteration.

---

# 15. Create the K-Means Model

Now we create our model.

```python
km = KMeans(
    n_clusters=4,
    max_iter=500
)
```

Here:

```text
Number of clusters = 4
Maximum iterations = 500
```

---

# 16. Train the Model

```python
y_means = km.fit_predict(X)
```

This does two things:

### `fit`

Learns the cluster centroids.

### `predict`

Assigns each data point to a cluster.

The result is stored in:

```python
y_means
```

For example:

```text
[0, 2, 1, 3, 0, 1, 2, ...]
```

Each number represents a cluster.

---

# 17. Check Cluster Sizes

We can check how many data points are present in every cluster.

```python
unique, counts = np.unique(
    y_means,
    return_counts=True
)

for cluster, count in zip(unique, counts):

    print(
        f"Cluster {cluster}: {count} students"
    )
```

---

# 18. View Final Centroids

The final centroids are stored in:

```python
km.centroids
```

We can print them:

```python
print("Final centroids:")
print(km.centroids)
```

These are the final centers of the four clusters.

---

# 19. Visualize the Clusters

Because this dataset contains two features, we can visualize the clusters in 2D.

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    X[y_means == 0, 0],
    X[y_means == 0, 1],
    color='red',
    label='Cluster 0'
)

plt.scatter(
    X[y_means == 1, 0],
    X[y_means == 1, 1],
    color='blue',
    label='Cluster 1'
)

plt.scatter(
    X[y_means == 2, 0],
    X[y_means == 2, 1],
    color='green',
    label='Cluster 2'
)

plt.scatter(
    X[y_means == 3, 0],
    X[y_means == 3, 1],
    color='yellow',
    label='Cluster 3'
)

# Plot centroids
plt.scatter(
    km.centroids[:, 0],
    km.centroids[:, 1],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])

plt.title('K-Means Clustering — Student Dataset')

plt.legend()

plt.show()
```

---

# 20. Understanding the Plot

The four colors represent the four clusters.

```text
Red       → Cluster 0
Blue      → Cluster 1
Green     → Cluster 2
Yellow    → Cluster 3
```

The black `X` marks represent the final centroids.

A centroid is essentially the **center point of a cluster**.

---

# 21. Complete K-Means Flow

The complete algorithm can be remembered as:

```text
                Dataset
                   ↓
             Choose K = 4
                   ↓
       Randomly select 4 centroids
                   ↓
        Calculate distances
                   ↓
       Assign nearest centroid
                   ↓
       Calculate cluster means
                   ↓
          Move centroids
                   ↓
      Did centroids change?
             ↙          ↘
           YES           NO
            ↓             ↓
         Repeat          Stop
                          ↓
                   Final clusters
```

---

# 22. Why Is K-Means Unsupervised?

K-Means does not require a target/output column.

Suppose we have:

```text
Student   Feature 1   Feature 2
A            20          50
B            22          52
C            80          90
D            82          88
```

We don't tell the algorithm:

```text
A → Cluster 0
B → Cluster 0
C → Cluster 1
D → Cluster 1
```

Instead, K-Means discovers the groups itself based on similarity/distance.

Therefore:

> **K-Means is an unsupervised learning algorithm.**

---

# 23. Important Point About Cluster Labels

Cluster numbers are arbitrary.

For example, one run could produce:

```text
Student A → Cluster 0
Student B → Cluster 0
Student C → Cluster 1
```

Another run could produce:

```text
Student A → Cluster 2
Student B → Cluster 2
Student C → Cluster 3
```

The actual grouping may be the same.

Only the cluster numbers changed.

This happens because the initial centroids are randomly selected.

---

# 24. Optional: Synthetic Data Using `make_blobs`

The original example also used Scikit-Learn's `make_blobs`.

```python
from sklearn.datasets import make_blobs
```

We can create artificial clusters:

```python
centroids = [
    (-5, -5),
    (5, 5),
    (-2.5, 2.5),
    (2.5, -2.5)
]

cluster_std = [1, 1, 1, 1]

X_blob, y_blob = make_blobs(
    n_samples=100,
    cluster_std=cluster_std,
    centers=centroids,
    n_features=2,
    random_state=2
)
```

Then visualize them:

```python
plt.figure(figsize=(7, 5))

plt.scatter(
    X_blob[:, 0],
    X_blob[:, 1]
)

plt.title("Synthetic Data Using make_blobs")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
```

---

# 25. Why Does `make_blobs` Have `y_blob`?

`make_blobs` generates artificial data.

Therefore, it already knows which artificial cluster each point came from.

```python
X_blob
```

contains the features.

```python
y_blob
```

contains the generated cluster labels.

For example:

```text
X_blob → input data
y_blob → known generated cluster
```

But in a real K-Means clustering problem, we generally don't have `y`.

That's why K-Means is unsupervised.

---

# 26. One Important Limitation of This Custom Implementation

This implementation is useful for **learning how K-Means works internally**, but it is not as robust as Scikit-Learn's implementation.

For real projects, you would normally use:

```python
from sklearn.cluster import KMeans
```

The custom implementation is mainly valuable because it helps you understand:

- Centroid initialization
- Distance calculation
- Cluster assignment
- Centroid update
- Iterations
- Convergence

---

# 27. Final Mental Model

Think of K-Means as four repeating actions:

```text
1. Place centroids
       ↓
2. Assign points
       ↓
3. Move centroids
       ↓
4. Repeat
```

Or even shorter:

> **Assign → Average → Move → Repeat**

That is the core idea behind K-Means.

---

# 28. Key Terms to Remember

| Term | Meaning |
|---|---|
| Cluster | Group of similar data points |
| K | Number of clusters |
| Centroid | Center of a cluster |
| Distance | Measures how far a point is from a centroid |
| Euclidean Distance | Common distance metric used by K-Means |
| Assignment | Giving each point its nearest cluster |
| Convergence | When centroids stop changing |
| `max_iter` | Maximum number of iterations |
| Unsupervised Learning | Learning without target labels |

---

# 29. Minimal Complete Code

If you want the entire implementation without explanations:

```python
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):

        random_index = random.sample(
            range(0, X.shape[0]),
            self.n_clusters
        )

        self.centroids = X[random_index]

        for i in range(self.max_iter):

            cluster_group = self.assign_clusters(X)

            old_centroids = self.centroids.copy()

            self.centroids = self.move_centroids(
                X,
                cluster_group
            )

            if np.array_equal(
                old_centroids,
                self.centroids
            ):
                break

        return cluster_group

    def assign_clusters(self, X):

        cluster_group = []

        for row in X:

            distances = []

            for centroid in self.centroids:

                distance = np.sqrt(
                    np.dot(
                        row - centroid,
                        row - centroid
                    )
                )

                distances.append(distance)

            min_distance = min(distances)

            index_pos = distances.index(min_distance)

            cluster_group.append(index_pos)

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):

        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for cluster in cluster_type:

            cluster_points = X[
                cluster_group == cluster
            ]

            cluster_mean = cluster_points.mean(
                axis=0
            )

            new_centroids.append(cluster_mean)

        return np.array(new_centroids)


# Load data
df = pd.read_csv('student_clustering.csv')

X = df.iloc[:, :].values


# Create model
km = KMeans(
    n_clusters=4,
    max_iter=500
)


# Train and predict
y_means = km.fit_predict(X)


# Visualize clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X[y_means == 0, 0],
    X[y_means == 0, 1],
    color='red',
    label='Cluster 0'
)

plt.scatter(
    X[y_means == 1, 0],
    X[y_means == 1, 1],
    color='blue',
    label='Cluster 1'
)

plt.scatter(
    X[y_means == 2, 0],
    X[y_means == 2, 1],
    color='green',
    label='Cluster 2'
)

plt.scatter(
    X[y_means == 3, 0],
    X[y_means == 3, 1],
    color='yellow',
    label='Cluster 3'
)

plt.scatter(
    km.centroids[:, 0],
    km.centroids[:, 1],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('K-Means Clustering')
plt.legend()
plt.show()
```

---

## Final Takeaway

K-Means repeatedly performs:

```text
Initialize Centroids
        ↓
Calculate Distances
        ↓
Assign Clusters
        ↓
Calculate Means
        ↓
Move Centroids
        ↓
Check Convergence
        ↓
Repeat
```

The most important concept to understand is:

> **Every point is assigned to the nearest centroid, and every centroid is then moved to the mean of the points assigned to it.**
