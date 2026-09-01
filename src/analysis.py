"""Analysis utilities for customer segmentation results."""


def summarize_clusters(df):
    """Return per-cluster summary statistics."""
    return df.groupby("cluster_label").size().to_dict()


if __name__ == "__main__":
    print("Analyzing results...")
