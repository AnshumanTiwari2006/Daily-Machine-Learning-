# K-Means Clustering — Part 2

This README documents the work performed in `K-Means_Part-2.ipynb`.

## Dataset

The notebook uses `student_clustering.csv`, containing **200 rows and 2 features**:

- `cgpa`
- `iq`

The notebook first inspects the dataset and visualizes `cgpa` against `iq`.

## 1. Import Libraries

```python
import numpy as np
import pandas as pd
```

- `numpy` is imported for numerical operations.
- `pandas` is used to load and manipulate the tabular dataset.

## 2. Load the Dataset

```python
df = pd.read_csv('student_clustering.csv')

print("The shape of data is", df.shape)

df.head()
```

The notebook reports the shape as `(200, 2)`. The first rows contain `cgpa` and `iq`.

## 3. Visualize the Data

```python
import matplotlib.pyplot as plt

plt.scatter(df['cgpa'], df['iq'])
```

A scatter plot is used because there are two numerical features. Each point represents one student, with CGPA on the x-axis and IQ on the y-axis.

## 4. Import K-Means

```python
from sklearn.cluster import KMeans
```

`KMeans` from Scikit-Learn provides the clustering algorithm.

## 5. WCSS and the Elbow Method

The notebook tests different values of `K` from 1 to 10:

```python
wcss = []

for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    wcss.append(km.inertia_)
```

### What is WCSS?

**WCSS (Within-Cluster Sum of Squares)** measures how close the data points are to the centroid of their assigned cluster.

For each value of `K`, the notebook stores:

```python
km.inertia_
```

In Scikit-Learn, K-Means inertia is the sum of squared distances of samples to their closest cluster center.

The notebook obtains these WCSS values:

| K | WCSS |
|---:|---:|
| 1 | 29957.8983 |
| 2 | 4184.1413 |
| 3 | 2362.7133 |
| 4 | 681.9697 |
| 5 | 514.1617 |
| 6 | 388.8524 |
| 7 | 295.4392 |
| 8 | 234.4869 |
| 9 | 199.9912 |
| 10 | 171.4059 |

### Elbow Plot

```python
plt.plot(range(1, 11), wcss)
```

The plot places the number of clusters on the x-axis and WCSS on the y-axis. The purpose is to look for the point where increasing K begins to give much smaller reductions in WCSS.

## 6. K-Means with a Chosen Number of Clusters

The notebook then creates a K-Means model with four clusters:

```python
X = df.iloc[:, :].values

km = KMeans(n_clusters=4)

y_means = km.fit_predict(X)
```

### `X = df.iloc[:, :].values`

This selects all rows and all columns from the DataFrame and converts them into a NumPy array.

The features are:

- CGPA
- IQ

### `n_clusters=4`

This tells K-Means to create four clusters.

### `fit_predict(X)`

`fit_predict()` fits the K-Means model and returns the cluster label assigned to each observation.

The notebook stores those labels in `y_means`.

The output contains cluster identifiers:

```text
0, 1, 2, 3
```

## 7. Selecting Points from a Particular Cluster

The notebook demonstrates Boolean indexing:

```python
X[y_means == 3, 1]
```

Here:

- `y_means == 3` creates a Boolean mask selecting observations assigned to cluster `3`.
- `X[..., 1]` selects the second feature, which is the IQ column because the features are ordered as `cgpa`, then `iq`.

## 8. 3D Visualization

The notebook later creates a 3D scatter visualization using Plotly:

```python
import plotly.express as px

fig = px.scatter_3d(
    x=X[:, 0],
    y=X[:, 1],
    z=X[:, 2]
)

fig.show()
```

This code expects at least three columns in `X`, because it accesses columns `0`, `1`, and `2`.

## 9. Another WCSS Experiment

The notebook also contains another WCSS loop testing `K` from 1 through 20:

```python
wcss = []

for i in range(1, 21):
    km = KMeans(n_clusters=i)
    km.fit_predict(X)
    wcss.append(km.inertia_)
```

This repeats the Elbow-method idea over a larger range of K values.

## Important Concepts

### K-Means

An unsupervised clustering algorithm that divides observations into a specified number of groups using cluster centroids.

### Cluster

A group of observations that K-Means considers similar according to the distance measure used.

### Centroid

The center of a cluster. K-Means updates centroids during the clustering process.

### WCSS / Inertia

A measure of the total squared distance of observations from their assigned cluster centers. K-Means attempts to minimize this quantity.

### Elbow Method

A method for choosing K by examining how WCSS changes as K increases. WCSS decreases as more clusters are added, so the goal is to identify a point after which additional clusters provide relatively smaller improvements.

### Cluster Labels

The integers returned by K-Means, such as `0`, `1`, `2`, and `3`, identify clusters. The numeric labels themselves do not have an inherent meaning.

## Notebook Workflow

```text
student_clustering.csv
        ↓
Load with Pandas
        ↓
Inspect shape and head
        ↓
Scatter plot: CGPA vs IQ
        ↓
Run K-Means for multiple K values
        ↓
Collect inertia / WCSS
        ↓
Plot Elbow curve
        ↓
Choose a K
        ↓
Fit K-Means
        ↓
Get cluster labels
        ↓
Visualize / inspect clusters
```

## Notes from the Notebook

- Dataset: `student_clustering.csv`
- Dataset shape: `(200, 2)`
- Features: `cgpa`, `iq`
- First WCSS experiment: `K = 1` through `10`
- Later clustering experiment: `n_clusters=4`
- Cluster labels stored in `y_means`
- Additional WCSS experiment: `K = 1` through `20`
- Plotly 3D visualization is also present in the notebook.

## Source

Based on the accompanying notebook: `K-Means_Part-2.ipynb`.
