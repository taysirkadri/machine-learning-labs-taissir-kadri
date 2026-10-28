import numpy as np
import pandas as pd

data = {
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics": [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms": [13, 15, 9, 17, 10, 14],
}
students = ["S1", "S2", "S3", "S4", "S5", "S6"]
#dataframe 
df = pd.DataFrame(data, index=students)
print("Dataset:")
print(df)

print("Dimensions:", df.shape)
print(df.describe().round(3))

# Standardize each subject.
X = df.values.astype(float)
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

# Correlation matrix of the standardized data.
R = np.corrcoef(Z, rowvar=False)
print("Standardized data:\n", pd.DataFrame(
    Z, index=df.index, columns=df.columns
).round(3))
print("Correlation matrix:\n", pd.DataFrame(
    R, index=df.columns, columns=df.columns
).round(3))


# each column of eigenvectors is the eigenvector associated with one eigenvalue.
eigenvalues, eigenvectors = np.linalg.eig(R)

# Sort eigenvalues in descending order and reorder the corresponding columns.
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices].real
eigenvectors = eigenvectors[:, sorted_indices].real

print("Sorted eigenvalues:\n", pd.DataFrame(
    eigenvalues, index=["PC1", "PC2", "PC3", "PC4"], columns=["Eigenvalue"]
).round(3))
print("Sorted eigenvectors:\n", pd.DataFrame(
    eigenvectors, index=df.columns,
    columns=["PC1", "PC2", "PC3", "PC4"]
).round(3))

# Select the two eigenvectors associated with the two largest eigenvalues.
selected_eigenvectors = eigenvectors[:, :2]

# Project the standardized data onto the two selected components.
Z_pca = Z @ selected_eigenvectors
print("Projected data:\n", pd.DataFrame(
    Z_pca, index=df.index, columns=["PC1", "PC2"]
).round(3))

# Explained variance ratios and cumulative explained variance.
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print("Explained variance ratio:\n", explained_variance_ratio)
print("Cumulative explained variance:\n",
      cumulative_explained_variance)
