from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
#                    CONFIGURATION
# ============================================================

# Project root = folder containing src/, data/, results/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BUSINESS_FILE = PROJECT_ROOT / "data" / "processed" / "businesses_clean.csv"
CLUSTER_FILE = PROJECT_ROOT / "results" / "clusters.csv"


# ============================================================
#                    HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):
    """
    Remove accidental characters such as ** from column names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("*", "", regex=False)
        .str.replace(" ", "_")
    )
    return df


def yes_no(value):
    """
    Convert boolean values to Yes/No for display.
    """
    if pd.isna(value):
        return "No"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if str(value).lower() in ["true", "1", "yes"]:
        return "Yes"

    return "No"


def percentage(series):
    """
    Calculate percentage of True/1 values.
    """
    if len(series) == 0:
        return 0.0

    values = series.astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    return values.mean() * 100


def safe_mean(series):
    """
    Calculate mean while safely handling missing values.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.dropna().empty:
        return 0.0

    return numeric.mean()


def get_main_category(cluster):
    """
    Return the most common category in a cluster.
    """
    if "category" not in cluster.columns:
        return "Unknown"

    categories = cluster["category"].dropna()

    if categories.empty:
        return "Unknown"

    return categories.mode().iloc[0]


def get_main_city(cluster):
    """
    Return the most common city in a cluster.
    """
    if "city" not in cluster.columns:
        return "Unknown"

    cities = cluster["city"].dropna()

    if cities.empty:
        return "Unknown"

    return cities.mode().iloc[0]


def interpret_cluster(cluster):
    """
    Automatically generate a simple interpretation
    based on digital presence, reviews, rating and sector.
    """

    interpretations = []

    website_pct = percentage(cluster["has_website"])
    phone_pct = percentage(cluster["has_phone"])

    if "has_social_media" in cluster.columns:
        social_pct = percentage(cluster["has_social_media"])
    else:
        social_pct = 0

    avg_reviews = safe_mean(cluster["review_count"])
    avg_rating = safe_mean(cluster["rating"])

    electrical_pct = percentage(cluster["electrical_related"])
    industrial_pct = percentage(cluster["industrial_related"])

    # --------------------------------------------------------
    # Digital presence
    # --------------------------------------------------------

    digital_score = (
        website_pct +
        phone_pct +
        social_pct
    ) / 3

    if digital_score >= 70:
        interpretations.append("High digital presence")

    elif digital_score >= 35:
        interpretations.append("Moderate digital presence")

    else:
        interpretations.append("Low digital presence")

    # --------------------------------------------------------
    # Customer engagement
    # --------------------------------------------------------

    if avg_reviews >= 20:
        interpretations.append("High customer engagement")

    elif avg_reviews >= 5:
        interpretations.append("Moderate customer engagement")

    else:
        interpretations.append("Low customer engagement")

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    if avg_rating >= 4.5:
        interpretations.append("Strong average rating")

    elif avg_rating >= 3.5:
        interpretations.append("Moderate average rating")

    elif avg_rating > 0:
        interpretations.append("Lower average rating")

    # --------------------------------------------------------
    # Sector
    # --------------------------------------------------------

    if electrical_pct >= 50:
        interpretations.append("Electrical-focused")

    elif industrial_pct >= 50:
        interpretations.append("Industrial-focused")

    else:
        interpretations.append("Construction/general services")

    return " | ".join(interpretations)


# ============================================================
#                    LOAD DATA
# ============================================================

print("\nLoading data...")

try:
    businesses = pd.read_csv(BUSINESS_FILE)
    clusters = pd.read_csv(CLUSTER_FILE)

except FileNotFoundError as e:
    print("\nERROR: Could not find one of the required files.")
    print("Make sure these files are in the same folder as display_clusters.py:")
    print(f"  - {BUSINESS_FILE}")
    print(f"  - {CLUSTER_FILE}")
    print(f"\nDetails: {e}")
    exit()


# ============================================================
#                    CLEAN COLUMNS
# ============================================================

businesses = clean_column_names(businesses)
clusters = clean_column_names(clusters)


# ============================================================
#                    VALIDATE DATA
# ============================================================

if "business_id" not in businesses.columns:
    print("\nERROR: 'business_id' column not found in business_clean.csv.")
    print("Available columns:")
    print(businesses.columns.tolist())
    exit()

if "business_id" not in clusters.columns:
    print("\nERROR: 'business_id' column not found in clusters.csv.")
    print("Available columns:")
    print(clusters.columns.tolist())
    exit()

if "cluster" not in clusters.columns:
    print("\nERROR: 'cluster' column not found in clusters.csv.")
    print("Available columns:")
    print(clusters.columns.tolist())
    exit()


# ============================================================
#                    MERGE DATA
# ============================================================

data = businesses.merge(
    clusters[["business_id", "cluster"]],
    on="business_id",
    how="inner"
)


if data.empty:
    print("\nERROR: No businesses could be matched between the two files.")
    exit()


# ============================================================
#                    FIX DATA TYPES
# ============================================================

numeric_columns = [
    "rating",
    "review_count",
    "social_platform_count"
]

for column in numeric_columns:
    if column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )


# ============================================================
#                    MAIN HEADER
# ============================================================

print("\n" + "=" * 70)
print("                    UFMEEG CLUSTER ANALYSIS")
print("=" * 70)

print(f"\nTotal businesses analyzed: {len(data)}")
print(f"Number of clusters: {data['cluster'].nunique()}")

print("\nCluster distribution:")
print("-" * 40)

distribution = data["cluster"].value_counts().sort_index()

for cluster_id, count in distribution.items():

    percentage_value = (count / len(data)) * 100

    print(
        f"Cluster {cluster_id}: "
        f"{count:3d} businesses "
        f"({percentage_value:5.1f}%)"
    )


# ============================================================
#                    CLUSTER ANALYSIS
# ============================================================

for cluster_id in sorted(data["cluster"].unique()):

    cluster = data[data["cluster"] == cluster_id].copy()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    number_businesses = len(cluster)

    avg_rating = safe_mean(cluster["rating"])

    avg_reviews = safe_mean(cluster["review_count"])

    website_pct = percentage(cluster["has_website"])

    phone_pct = percentage(cluster["has_phone"])

    if "has_social_media" in cluster.columns:
        social_pct = percentage(cluster["has_social_media"])
    else:
        social_pct = 0

    main_category = get_main_category(cluster)

    main_city = get_main_city(cluster)

    interpretation = interpret_cluster(cluster)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(f"CLUSTER {cluster_id}")
    print("=" * 70)

    print(f"\nNumber of businesses: {number_businesses}")

    print(f"Main city: {main_city}")

    print(f"Main category: {main_category}")

    print(f"\nAverage rating: {avg_rating:.2f}")

    print(f"Average review count: {avg_reviews:.2f}")

    print(f"Website presence: {website_pct:.1f}%")

    print(f"Phone presence: {phone_pct:.1f}%")

    print(f"Social media presence: {social_pct:.1f}%")

    print("\nInterpretation:")
    print(f"  {interpretation}")

    # --------------------------------------------------------
    # Businesses
    # --------------------------------------------------------

    print("\nBusinesses:")
    print("-" * 70)

    # Sort businesses by review count
    if "review_count" in cluster.columns:
        cluster = cluster.sort_values(
            by="review_count",
            ascending=False
        )

    for _, business in cluster.iterrows():

        business_id = business.get(
            "business_id",
            "Unknown"
        )

        business_name = business.get(
            "business_name",
            "Unknown"
        )

        category = business.get(
            "category",
            "Unknown"
        )

        city = business.get(
            "city",
            "Unknown"
        )

        rating = business.get(
            "rating",
            np.nan
        )

        reviews = business.get(
            "review_count",
            np.nan
        )

        website = yes_no(
            business.get("has_website", False)
        )

        phone = yes_no(
            business.get("has_phone", False)
        )

        social = yes_no(
            business.get("has_social_media", False)
        )

        # Format rating
        if pd.isna(rating):
            rating_text = "N/A"
        else:
            rating_text = f"{rating:.1f}"

        # Format reviews
        if pd.isna(reviews):
            reviews_text = "N/A"
        else:
            reviews_text = f"{int(reviews)}"

        print(f"\n{business_id} | {business_name}")

        print(f"  Category : {category}")

        print(f"  City     : {city}")

        print(
            f"  Rating   : {rating_text} "
            f"| Reviews: {reviews_text}"
        )

        print(
            f"  Website  : {website} "
            f"| Phone: {phone} "
            f"| Social: {social}"
        )


# ============================================================
#                    FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("                         FINAL SUMMARY")
print("=" * 70)

print(
    f"\nThe dataset contains {len(data)} businesses "
    f"distributed across {data['cluster'].nunique()} clusters."
)

print("\nCluster sizes:")

for cluster_id, count in distribution.items():

    print(
        f"  Cluster {cluster_id}: "
        f"{count} businesses"
    )


# ------------------------------------------------------------
# Largest cluster
# ------------------------------------------------------------

largest_cluster = distribution.idxmax()
largest_count = distribution.max()

print(
    f"\nLargest cluster: Cluster {largest_cluster} "
    f"({largest_count} businesses)"
)


# ------------------------------------------------------------
# Smallest cluster
# ------------------------------------------------------------

smallest_cluster = distribution.idxmin()
smallest_count = distribution.min()

print(
    f"Smallest cluster: Cluster {smallest_cluster} "
    f"({smallest_count} businesses)"
)


# ------------------------------------------------------------
# Best digital presence
# ------------------------------------------------------------

digital_scores = {}

for cluster_id in sorted(data["cluster"].unique()):

    cluster = data[data["cluster"] == cluster_id]

    website = percentage(cluster["has_website"])
    phone = percentage(cluster["has_phone"])

    if "has_social_media" in cluster.columns:
        social = percentage(cluster["has_social_media"])
    else:
        social = 0

    digital_scores[cluster_id] = (
        website + phone + social
    ) / 3


best_digital_cluster = max(
    digital_scores,
    key=digital_scores.get
)

print(
    f"\nStrongest digital presence: "
    f"Cluster {best_digital_cluster} "
    f"({digital_scores[best_digital_cluster]:.1f}% average)"
)


# ------------------------------------------------------------
# Highest customer engagement
# ------------------------------------------------------------

review_scores = {}

for cluster_id in sorted(data["cluster"].unique()):

    cluster = data[data["cluster"] == cluster_id]

    review_scores[cluster_id] = safe_mean(
        cluster["review_count"]
    )


best_review_cluster = max(
    review_scores,
    key=review_scores.get
)

print(
    f"Highest average review count: "
    f"Cluster {best_review_cluster} "
    f"({review_scores[best_review_cluster]:.2f} reviews)"
)


# ============================================================
#                    END
# ============================================================

print("\n" + "=" * 70)
print("                    ANALYSIS COMPLETED")
print("=" * 70)
print()