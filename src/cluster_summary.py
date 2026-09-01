import pandas as pd
from pathlib import Path

# ============================================================
# UFMEEG - CLUSTER SUMMARY
# ============================================================

print("=" * 70)
print("                    UFMEEG CLUSTER SUMMARY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Locate project root
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

BUSINESS_FILE = BASE_DIR / "data" / "processed" / "businesses_clean.csv"
CLUSTER_FILE = BASE_DIR / "results" / "clusters.csv"

# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------

print("\nLoading data...")

try:
    businesses = pd.read_csv(BUSINESS_FILE)
    clusters = pd.read_csv(CLUSTER_FILE)

except FileNotFoundError as e:
    print("\nERROR: Could not find one of the required files.")
    print("Expected files:")
    print(f"  - {BUSINESS_FILE}")
    print(f"  - {CLUSTER_FILE}")
    print(f"\nDetails: {e}")
    exit()

print("Data loaded successfully.")

# ------------------------------------------------------------
# 3. Clean column names
# ------------------------------------------------------------

businesses.columns = businesses.columns.str.replace("*", "", regex=False).str.strip()
clusters.columns = clusters.columns.str.replace("*", "", regex=False).str.strip()

# ------------------------------------------------------------
# 4. Merge business data with cluster assignments
# ------------------------------------------------------------

if "business_id" not in businesses.columns:
    print("\nERROR: 'business_id' column not found in businesses_clean.csv.")
    exit()

if "business_id" not in clusters.columns:
    print("\nERROR: 'business_id' column not found in clusters.csv.")
    exit()

if "cluster" not in clusters.columns:
    print("\nERROR: 'cluster' column not found in clusters.csv.")
    exit()

df = businesses.merge(
    clusters[["business_id", "cluster"]],
    on="business_id",
    how="inner"
)

if df.empty:
    print("\nERROR: No businesses could be matched with cluster assignments.")
    exit()

# ------------------------------------------------------------
# 5. Basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GENERAL INFORMATION")
print("=" * 70)

print(f"\nTotal businesses analyzed : {len(df)}")
print(f"Number of clusters       : {df['cluster'].nunique()}")

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

# ------------------------------------------------------------
# 6. Generate cluster summaries
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DETAILED CLUSTER ANALYSIS")
print("=" * 70)

summary_rows = []

for cluster_id in sorted(df["cluster"].unique()):

    cluster_data = df[df["cluster"] == cluster_id]

    size = len(cluster_data)
    percentage = (size / len(df)) * 100

    # Average rating
    avg_rating = cluster_data["rating"].mean()

    # Average reviews
    avg_reviews = cluster_data["review_count"].mean()

    # Website percentage
    website_pct = cluster_data["has_website"].mean() * 100

    # Phone percentage
    phone_pct = cluster_data["has_phone"].mean() * 100

    # Social media percentage
    social_pct = cluster_data["has_social_media"].mean() * 100

    # Electrical percentage
    electrical_pct = cluster_data["electrical_related"].mean() * 100

    # Industrial percentage
    industrial_pct = cluster_data["industrial_related"].mean() * 100

    # Main category
    if "category" in cluster_data.columns:
        main_category = cluster_data["category"].value_counts().index[0]
    else:
        main_category = "N/A"

    # Main city
    if "city" in cluster_data.columns:
        main_city = cluster_data["city"].value_counts().index[0]
    else:
        main_city = "N/A"

    # Store summary
    summary_rows.append({
        "cluster": cluster_id,
        "businesses": size,
        "percentage": percentage,
        "avg_rating": avg_rating,
        "avg_reviews": avg_reviews,
        "website_pct": website_pct,
        "phone_pct": phone_pct,
        "social_pct": social_pct,
        "electrical_pct": electrical_pct,
        "industrial_pct": industrial_pct,
        "main_category": main_category,
        "main_city": main_city
    })

    # --------------------------------------------------------
    # Display cluster
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"CLUSTER {cluster_id}")
    print("-" * 70)

    print(f"Businesses           : {size}")
    print(f"Percentage of total  : {percentage:.1f}%")

    if pd.notna(avg_rating):
        print(f"Average rating       : {avg_rating:.2f}")
    else:
        print("Average rating       : N/A")

    if pd.notna(avg_reviews):
        print(f"Average reviews      : {avg_reviews:.2f}")
    else:
        print("Average reviews      : N/A")

    print(f"Website presence     : {website_pct:.1f}%")
    print(f"Phone presence       : {phone_pct:.1f}%")
    print(f"Social media         : {social_pct:.1f}%")
    print(f"Electrical related   : {electrical_pct:.1f}%")
    print(f"Industrial related   : {industrial_pct:.1f}%")
    print(f"Main category        : {main_category}")
    print(f"Main city            : {main_city}")

# ------------------------------------------------------------
# 7. Create summary DataFrame
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary_rows)

# ------------------------------------------------------------
# 8. Display comparison table
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLUSTER COMPARISON")
print("=" * 70)

display_columns = [
    "cluster",
    "businesses",
    "percentage",
    "avg_rating",
    "avg_reviews",
    "website_pct",
    "phone_pct",
    "social_pct",
    "electrical_pct",
    "industrial_pct"
]

comparison = summary_df[display_columns].copy()

comparison["percentage"] = comparison["percentage"].round(1)
comparison["avg_rating"] = comparison["avg_rating"].round(2)
comparison["avg_reviews"] = comparison["avg_reviews"].round(2)
comparison["website_pct"] = comparison["website_pct"].round(1)
comparison["phone_pct"] = comparison["phone_pct"].round(1)
comparison["social_pct"] = comparison["social_pct"].round(1)
comparison["electrical_pct"] = comparison["electrical_pct"].round(1)
comparison["industrial_pct"] = comparison["industrial_pct"].round(1)

print("\n")
print(comparison.to_string(index=False))

# ------------------------------------------------------------
# 9. Save summary
# ------------------------------------------------------------

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

output_file = RESULTS_DIR / "cluster_summary.csv"

summary_df.to_csv(output_file, index=False)

# ------------------------------------------------------------
# 10. Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLUSTER INTERPRETATION")
print("=" * 70)

for _, row in summary_df.iterrows():

    cluster_id = int(row["cluster"])

    characteristics = []

    if row["website_pct"] >= 70:
        characteristics.append("high website presence")
    elif row["website_pct"] == 0:
        characteristics.append("no website presence")

    if row["phone_pct"] >= 70:
        characteristics.append("strong phone presence")
    elif row["phone_pct"] == 0:
        characteristics.append("no phone presence")

    if row["social_pct"] >= 50:
        characteristics.append("strong social media presence")

    if row["electrical_pct"] >= 50:
        characteristics.append("mostly electrical-related")

    if row["industrial_pct"] >= 50:
        characteristics.append("mostly industrial-related")

    if row["avg_reviews"] >= 20:
        characteristics.append("high customer-review activity")
    elif row["avg_reviews"] <= 2:
        characteristics.append("low customer-review activity")

    description = ", ".join(characteristics)

    if not description:
        description = "mixed characteristics"

    print(f"\nCluster {cluster_id}:")
    print(f"  {description.capitalize()}.")

# ------------------------------------------------------------
# 11. Finish
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Analysis completed.")
print("=" * 70)

print(f"\nSummary saved to:")
print(f"  {output_file}")