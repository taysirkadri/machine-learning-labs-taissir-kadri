import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Dataset
data = {
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics": [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms": [13, 15, 9, 17, 10, 14],
}

students = ["S1", "S2", "S3", "S4", "S5", "S6"]
df = pd.DataFrame(data, index=students)

# Use the same StandardScaler convention as PCA-scikit.py.
scaler = StandardScaler()
Z = scaler.fit_transform(df)

# PCA with Scikit-learn.
pca = PCA(n_components=2)
principal_components = pca.fit_transform(Z)
transformed_df = pd.DataFrame(
    principal_components,
    index=df.index,
    columns=["PC1", "PC2"],
)

# Manual PCA from the correlation matrix.
correlation_matrix = np.corrcoef(Z, rowvar=False)
manual_eigenvalues, manual_eigenvectors = np.linalg.eigh(correlation_matrix)

order = np.argsort(manual_eigenvalues)[::-1]
manual_eigenvalues = manual_eigenvalues[order]
manual_eigenvectors = manual_eigenvectors[:, order]
manual_coordinates = Z @ manual_eigenvectors[:, :2]

manual_coordinates_df = pd.DataFrame(
    manual_coordinates,
    index=df.index,
    columns=["PC1", "PC2"],
)

print("Part C: Comparison and interpretation")

# 14. Compare the raw coordinates without changing their signs.
print("\n14. Manual PCA coordinates (raw):")
print(manual_coordinates_df.round(6))

print("\nScikit-learn PCA coordinates (raw):")
print(transformed_df.round(6))

print("\nSigned difference: manual - Scikit-learn:")
print((manual_coordinates_df - transformed_df).round(6))

print("\nAbsolute difference:")
print((manual_coordinates_df - transformed_df).abs().round(6))

# 15. Compare eigenvalues and explained variance ratios.
# The manual correlation matrix uses 1 / n, while sklearn PCA uses 1 / (n - 1).
# Consequently, sklearn eigenvalues are manual eigenvalues multiplied by n / (n - 1).
manual_ratios = manual_eigenvalues / manual_eigenvalues.sum()
scikit_eigenvalues = pca.explained_variance_

print("\n15. Manual eigenvalues:")
print(manual_eigenvalues.round(6))

print("\nScikit-learn eigenvalues:")
print(scikit_eigenvalues.round(6))

print("\nManual explained variance ratios:")
print(manual_ratios.round(6))

print("\nScikit-learn explained variance ratios:")
print(pca.explained_variance_ratio_.round(6))

print("\nDifference between the first two explained variance ratios:")
print(
    np.abs(manual_ratios[:2] - pca.explained_variance_ratio_)
    .round(10)
)

# 16. Plot both raw coordinate sets.
plt.figure(figsize=(9, 6))
plt.scatter(
    manual_coordinates[:, 0],
    manual_coordinates[:, 1],
    marker="o",
    label="Manual PCA",
)
plt.scatter(
    principal_components[:, 0],
    principal_components[:, 1],
    marker="x",
    label="Scikit-learn PCA",
)

for i, student in enumerate(students):
    plt.annotate(
        student,
        (manual_coordinates[i, 0], manual_coordinates[i, 1]),
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Raw student coordinates: Manual PCA vs Scikit-learn PCA")
plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(0, color="gray", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("pca_comparison_raw.png", dpi=150)
plt.close()

print("\n16. Plot saved as: pca_comparison_raw.png")

# 17. Explain the raw comparison.
print(
    "\n17. Explanation:\n"
    "The raw coordinates can differ by the sign of a principal component.\n"
    "This is not an error: if v is an eigenvector, -v is also an eigenvector.\n"
    "Therefore, a component and all its scores may be multiplied by -1.\n"
    "The variance, distances, and relationships between students do not change.\n"
    "The eigenvalue difference comes from using 1/n in the correlation matrix\n"
    "and 1/(n-1) in Scikit-learn's sample covariance.\n"
)

# 18. Retained variance must use the first two components, not the last value.
variance_retained = pca.explained_variance_ratio_.sum()
print(
    "18. Variance retained by the first two components:",
    f"{variance_retained * 100:.4f}%",
)
