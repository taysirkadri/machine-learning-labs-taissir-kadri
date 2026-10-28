import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = {
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics": [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms": [13, 15, 9, 17, 10, 14],
}

students = ["S1", "S2", "S3", "S4", "S5", "S6"]

df = pd.DataFrame(data, index=students)

print("Dataset:")
print(df)

print("\nDimensions:")
print(df.shape)

#standardize variables with sklearn
scaler = StandardScaler()
Z = scaler.fit_transform(df)

standardized_df = pd.DataFrame(
    Z,
    index=df.index,
    columns=df.columns,
)

print("\nStandardized data:")
print(standardized_df.round(3))

#PCA and keep two principal components

pca = PCA(n_components=2)
principal_components = pca.fit_transform(Z)

transformed_df = pd.DataFrame(
    principal_components,
    index=df.index,
    columns=["PC1", "PC2"],
)


#transformed coordinates

print("\nTransformed coordinates:")
print(transformed_df.round(3))

#explained variance

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print("\nExplained variance ratio:")
print(explained_variance_ratio.round(6))

print("\nCumulative explained variance:")
print(cumulative_explained_variance.round(6))


# Part C: Comparison and interpretation
