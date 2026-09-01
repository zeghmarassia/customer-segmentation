import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# 1. Load ML features
# ============================================================

FEATURES_PATH = "data/features/ml_features.csv"
CLUSTERS_PATH = "results/clusters.csv"

df = pd.read_csv(FEATURES_PATH)
clusters = pd.read_csv(CLUSTERS_PATH)


# ============================================================
# 2. Clean column names if necessary
# ============================================================

df.columns = df.columns.str.replace("*", "", regex=False).str.strip()


# ============================================================
# 3. Prepare feature matrix
# ============================================================

# business_id is an identifier, not a feature
feature_columns = [col for col in df.columns if col != "business_id"]

X = df[feature_columns]


# ============================================================
# 4. Apply PCA
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)


# ============================================================
# 5. Create PCA results dataframe
# ============================================================

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["business_id"] = df["business_id"]


# Add cluster labels
if "business_id" in clusters.columns:
    pca_df = pca_df.merge(
        clusters,
        on="business_id",
        how="left"
    )
else:
    # If clusters.csv only contains a cluster column
    cluster_column = clusters.columns[-1]
    pca_df["cluster"] = clusters[cluster_column].values


# ============================================================
# 6. Explained variance
# ============================================================

explained_variance = pca.explained_variance_ratio_

print("\nPCA explained variance:")
print(f"PC1: {explained_variance[0] * 100:.2f}%")
print(f"PC2: {explained_variance[1] * 100:.2f}%")
print(
    f"Total: {explained_variance.sum() * 100:.2f}%"
)


# ============================================================
# 7. Save PCA coordinates
# ============================================================

pca_df.to_csv(
    "results/pca_coordinates.csv",
    index=False
)


# ============================================================
# 8. PCA visualization
# ============================================================

plt.figure(figsize=(10, 7))

for cluster in sorted(pca_df["cluster"].dropna().unique()):

    cluster_data = pca_df[pca_df["cluster"] == cluster]

    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        label=f"Cluster {int(cluster)}",
        alpha=0.7
    )


plt.xlabel(
    f"PC1 ({explained_variance[0] * 100:.1f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1] * 100:.1f}% variance)"
)

plt.title("PCA Visualization of Business Segments")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/figures/pca_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()