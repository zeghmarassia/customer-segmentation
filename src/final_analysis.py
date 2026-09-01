"""
UFMEEG - Final Analysis

Combines:
    1. Business data
    2. Clustering results
    3. Prospect scoring results

Produces:
    - Overall project statistics
    - Cluster analysis
    - Prospect analysis
    - Top recommended clients
    - Product-specific opportunities
"""

import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUSINESSES_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "businesses_clean.csv"
)

CLUSTERS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "clusters.csv"
)

PROSPECTS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "prospect_scores.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading project results...")

    businesses = pd.read_csv(BUSINESSES_FILE)
    clusters = pd.read_csv(CLUSTERS_FILE)
    prospects = pd.read_csv(PROSPECTS_FILE)

    # Clean column names
    for df in [businesses, clusters, prospects]:

        df.columns = (
            df.columns
            .str.replace("**", "", regex=False)
            .str.strip()
        )

    print("Data loaded successfully.")

    return businesses, clusters, prospects


# ============================================================
# GENERAL STATISTICS
# ============================================================

def general_statistics(businesses):

    print("\n" + "=" * 75)
    print("                    GENERAL STATISTICS")
    print("=" * 75)

    print(f"\nTotal businesses analyzed : {len(businesses)}")

    print(
        f"Businesses with website  : "
        f"{businesses['has_website'].sum()}"
    )

    print(
        f"Businesses with phone    : "
        f"{businesses['has_phone'].sum()}"
    )

    print(
        f"Businesses with social media : "
        f"{businesses['has_social_media'].sum()}"
    )

    print(
        f"Electrical-related businesses : "
        f"{businesses['electrical_related'].sum()}"
    )

    print(
        f"Industrial-related businesses : "
        f"{businesses['industrial_related'].sum()}"
    )


# ============================================================
# CLUSTER ANALYSIS
# ============================================================

def cluster_analysis(clusters, prospects):

    print("\n" + "=" * 75)
    print("                       CLUSTER ANALYSIS")
    print("=" * 75)

    if "cluster" not in clusters.columns:

        print("\nCluster column not found.")

        return

    cluster_counts = (
        clusters["cluster"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nNumber of clusters : "
        f"{clusters['cluster'].nunique()}"
    )

    print("\nBusinesses per cluster:")

    for cluster_id, count in cluster_counts.items():

        print(
            f"  Cluster {cluster_id}: "
            f"{count} businesses"
        )

    # --------------------------------------------------------
    # Prospect quality by cluster
    # --------------------------------------------------------

    if "cluster" in prospects.columns:

        cluster_scores = (
            prospects
            .groupby("cluster")["prospect_score"]
            .agg(["count", "mean", "max"])
            .round(2)
        )

        print("\nProspect quality by cluster:")

        print(
            cluster_scores.to_string()
        )

        best_cluster = (
            cluster_scores["mean"]
            .idxmax()
        )

        print(
            f"\nCluster with highest average "
            f"prospect score: {best_cluster}"
        )


# ============================================================
# PROSPECT ANALYSIS
# ============================================================

def prospect_analysis(prospects):

    print("\n" + "=" * 75)
    print("                     PROSPECT ANALYSIS")
    print("=" * 75)

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print("\nProspect levels:")

    distribution = (
        prospects["prospect_level"]
        .value_counts()
    )

    for level in ["HIGH", "MEDIUM", "LOW"]:

        count = distribution.get(level, 0)

        percentage = (
            count / len(prospects) * 100
        )

        print(
            f"  {level:<8}: "
            f"{count:>3} "
            f"({percentage:.1f}%)"
        )

    # --------------------------------------------------------
    # Top 10
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print("TOP 10 RECOMMENDED PROSPECTS")
    print("-" * 75)

    top_columns = [
        "rank",
        "business_name",
        "category",
        "prospect_score",
        "prospect_level",
        "best_product"
    ]

    available_columns = [
        column
        for column in top_columns
        if column in prospects.columns
    ]

    print(
        prospects[
            available_columns
        ]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

def product_analysis(prospects):

    print("\n" + "=" * 75)
    print("                      PRODUCT ANALYSIS")
    print("=" * 75)

    products = {
        "Electricity meters": "electricity_meter_score",
        "Gas meters": "gas_meter_score",
        "Gas leak detectors": "gas_detector_score"
    }

    for product, column in products.items():

        if column not in prospects.columns:
            continue

        strong = (
            prospects[column] >= 70
        ).sum()

        average = (
            prospects[column].mean()
        )

        print(
            f"\n{product}:"
        )

        print(
            f"  Strong prospects : {strong}"
        )

        print(
            f"  Average fit     : {average:.2f}/100"
        )

        # Top businesses for this product

        top = (
            prospects
            .sort_values(
                column,
                ascending=False
            )
            .head(5)
        )

        print("  Top candidates:")

        for _, row in top.iterrows():

            print(
                f"    - {row['business_name']} "
                f"({row[column]:.1f}/100)"
            )


# ============================================================
# WHOLESALE ANALYSIS
# ============================================================

def wholesale_analysis(prospects):

    print("\n" + "=" * 75)
    print("                 WHOLESALE / DISTRIBUTION ANALYSIS")
    print("=" * 75)

    if "wholesale_score" not in prospects.columns:
        return

    strong = (
        prospects["wholesale_score"] >= 70
    ).sum()

    print(
        f"\nStrong wholesale/distribution prospects: "
        f"{strong}"
    )

    print("\nTop potential distributors:")

    top = (
        prospects
        .sort_values(
            "wholesale_score",
            ascending=False
        )
        .head(10)
    )

    for _, row in top.iterrows():

        print(
            f"  {row['business_name']:<45} "
            f"{row['wholesale_score']:.1f}/100"
        )


# ============================================================
# FINAL RECOMMENDATIONS
# ============================================================

def final_recommendations(prospects):

    print("\n" + "=" * 75)
    print("                    FINAL RECOMMENDATIONS")
    print("=" * 75)

    high = prospects[
        prospects["prospect_level"] == "HIGH"
    ]

    print(
        f"\nNumber of HIGH-priority prospects: "
        f"{len(high)}"
    )

    print("\nRecommended businesses for first contact:")

    for _, row in high.head(10).iterrows():

        print(
            f"\n  {row['business_name']}"
        )

        print(
            f"    Category     : {row['category']}"
        )

        print(
            f"    Score        : "
            f"{row['prospect_score']:.2f}"
        )

        print(
            f"    Best product : "
            f"{row['best_product']}"
        )

        if "reasons" in prospects.columns:

            print(
                f"    Reasons      : "
                f"{row['reasons']}"
            )

    # --------------------------------------------------------
    # Strategic conclusion
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print("STRATEGIC CONCLUSION")
    print("-" * 75)

    print(
        """
The analysis identifies businesses that are more likely to be
valuable B2B prospects for UFMEEG.

Priority should be given to businesses that combine:

    - Strong product relevance
    - Wholesale or distribution potential
    - Electrical or industrial activity
    - Good commercial reputation
    - Sufficient business/contact presence

The prospect score provides a ranking that can help the sales
team prioritize potential clients instead of contacting all
businesses with the same priority.
"""
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("                  UFMEEG FINAL ANALYSIS")
    print("=" * 75)

    try:

        businesses, clusters, prospects = load_data()

    except FileNotFoundError as e:

        print("\nERROR: Required file not found.")
        print(f"\nDetails: {e}")

        return

    # --------------------------------------------------------
    # Run analyses
    # --------------------------------------------------------

    general_statistics(
        businesses
    )

    cluster_analysis(
        clusters,
        prospects
    )

    prospect_analysis(
        prospects
    )

    product_analysis(
        prospects
    )

    wholesale_analysis(
        prospects
    )

    final_recommendations(
        prospects
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("                    ANALYSIS COMPLETED")
    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()