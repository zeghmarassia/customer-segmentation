"""
UFMEEG - Prospect Scoring

Purpose:
    Rank businesses according to their potential value as customers
    for UFMEEG products:

        - Electricity meters
        - Gas meters
        - Gas leak detectors

Target:
    B2B customers, especially wholesalers, distributors, suppliers,
    electrical/electronic businesses and industrial businesses.

Input:
    data/processed/businesses_clean.csv
    results/clusters.csv

Output:
    results/prospect_scores.csv
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
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

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "prospect_scores.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_N = 20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(value, minimum, maximum):
    """
    Normalize a value between 0 and 1.
    """
    if maximum == minimum:
        return 0.5

    return (value - minimum) / (maximum - minimum)


def contains_any(text, keywords):
    """
    Check whether any keyword appears in text.
    """
    text = str(text).lower()

    return any(keyword in text for keyword in keywords)


# ============================================================
# PRODUCT RELEVANCE
# ============================================================

def electricity_meter_score(row):
    """
    Estimate how relevant the business is for electricity meters.
    """

    category = str(row["category"]).lower()
    name = str(row["business_name"]).lower()

    score = 0

    # Strong indicators
    if contains_any(category, [
        "electrical",
        "electric",
        "electronics",
        "appliance",
        "electrical equipment",
        "electrical supply"
    ]):
        score += 55

    if contains_any(category, [
        "wholesaler",
        "supplier",
        "distributor"
    ]):
        score += 25

    if row["electrical_related"]:
        score += 15

    if row["industrial_related"]:
        score += 5

    # Business name indicators
    if contains_any(name, [
        "elec",
        "electric",
        "electrique",
        "électricité",
        "electrical"
    ]):
        score += 5

    return min(score, 100)


def gas_meter_score(row):
    """
    Estimate relevance for gas meters.

    Since the original dataset contains limited explicit gas-related
    information, this score uses business categories and industrial/
    distribution characteristics rather than assigning zero to every
    business.
    """

    category = str(row["category"]).lower()
    name = str(row["business_name"]).lower()

    score = 0

    # Direct gas indicators
    if contains_any(category, [
        "gas",
        "gaz",
        "heating",
        "plumbing",
        "plumber",
        "sanitary"
    ]):
        score += 70

    if contains_any(name, [
        "gas",
        "gaz",
        "gpl"
    ]):
        score += 60

    # Businesses capable of distributing technical equipment
    if contains_any(category, [
        "wholesaler",
        "supplier",
        "distributor"
    ]):
        score += 20

    # Industrial businesses can be potential B2B buyers
    if row["industrial_related"]:
        score += 10

    return min(score, 100)


def gas_detector_score(row):
    """
    Estimate relevance for gas leak detectors.

    Gas detectors are particularly relevant to industrial,
    construction, electrical and technical businesses.
    """

    category = str(row["category"]).lower()
    name = str(row["business_name"]).lower()

    score = 0

    # Direct gas/safety indicators
    if contains_any(category, [
        "gas",
        "gaz",
        "safety",
        "security",
        "fire",
        "industrial equipment"
    ]):
        score += 70

    if contains_any(name, [
        "gas",
        "gaz",
        "sécurité",
        "security",
        "safety"
    ]):
        score += 50

    # Industrial businesses are strong candidates
    if row["industrial_related"]:
        score += 30

    # Electrical/technical businesses can also distribute detectors
    if row["electrical_related"]:
        score += 20

    # Suppliers and wholesalers have distribution potential
    if contains_any(category, [
        "supplier",
        "wholesaler",
        "distributor",
        "electrical equipment",
        "electrical supply"
    ]):
        score += 15

    return min(score, 100)


# ============================================================
# WHOLESALE / DISTRIBUTION POTENTIAL
# ============================================================

def wholesale_score(row):
    """
    Estimate the probability that the business could act as a
    reseller, wholesaler or distributor.
    """

    category = str(row["category"]).lower()
    name = str(row["business_name"]).lower()

    score = 0

    # Very strong indicators
    if contains_any(category, [
        "wholesaler",
        "wholesale"
    ]):
        score += 100

    elif contains_any(category, [
        "distributor",
        "supplier"
    ]):
        score += 90

    # Stores can also resell products
    elif contains_any(category, [
        "store",
        "shop",
        "retail"
    ]):
        score += 60

    # Companies/offices are possible B2B clients
    elif contains_any(category, [
        "company",
        "corporate",
        "manufacturer"
    ]):
        score += 45

    # Industrial businesses
    if row["industrial_related"]:
        score += 10

    # Business-name indicators
    if contains_any(name, [
        "distributeur",
        "distributor",
        "supplier",
        "grossiste",
        "wholesale"
    ]):
        score += 25

    return min(score, 100)


# ============================================================
# BUSINESS QUALITY
# ============================================================

def business_quality_score(row):
    """
    Estimate general commercial/business quality.

    Uses rating and review count.
    """

    rating = row["rating"]
    reviews = row["review_count"]

    # Rating component
    if pd.isna(rating):
        rating_score = 50
    else:
        rating_score = (rating / 5) * 100

    # Review activity component
    if pd.isna(reviews):
        review_score = 0
    else:
        # Log transformation prevents businesses with huge review
        # counts from dominating the score.
        review_score = min(
            np.log1p(reviews) / np.log1p(100),
            1
        ) * 100

    return (rating_score * 0.6) + (review_score * 0.4)


# ============================================================
# BUSINESS PRESENCE / MATURITY
# ============================================================

def business_presence_score(row):
    """
    Estimate business maturity based on digital/contact presence.
    """

    score = 0

    if row["has_website"]:
        score += 40

    if row["has_phone"]:
        score += 35

    if row["has_social_media"]:
        score += 25

    return score


# ============================================================
# PRODUCT FIT
# ============================================================

def calculate_product_fit(row):
    """
    Calculate all product-specific scores.
    """

    electricity = electricity_meter_score(row)
    gas = gas_meter_score(row)
    detector = gas_detector_score(row)

    return electricity, gas, detector


# ============================================================
# MAIN SCORING
# ============================================================

def calculate_prospect_score(row):
    """
    Calculate the overall prospect score.

    Weighting:

        Product relevance       40%
        Wholesale potential     25%
        Business quality        20%
        Business presence       15%
    """

    electricity = row["electricity_meter_score"]
    gas = row["gas_meter_score"]
    detector = row["gas_detector_score"]

    wholesale = row["wholesale_score"]
    quality = row["business_quality_score"]
    presence = row["business_presence_score"]

    # Best product opportunity
    best_product_score = max(
        electricity,
        gas,
        detector
    )

    # Product relevance
    product_component = best_product_score * 0.40

    # Wholesale/distribution potential
    wholesale_component = wholesale * 0.25

    # Business quality
    quality_component = quality * 0.20

    # Business presence
    presence_component = presence * 0.15

    total = (
        product_component
        + wholesale_component
        + quality_component
        + presence_component
    )

    return round(total, 2)


# ============================================================
# PROSPECT LEVEL
# ============================================================

def prospect_level(score):
    """
    Convert numerical score into a prospect level.
    """

    if score >= 75:
        return "HIGH"

    elif score >= 55:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# BEST PRODUCT
# ============================================================

def get_best_product(row):
    """
    Identify the product with the highest estimated relevance.
    """

    scores = {
        "Electricity meters": row["electricity_meter_score"],
        "Gas meters": row["gas_meter_score"],
        "Gas leak detectors": row["gas_detector_score"]
    }

    return max(scores, key=scores.get)


# ============================================================
# REASONS
# ============================================================

def generate_reasons(row):
    """
    Generate human-readable explanations for the prospect score.
    """

    reasons = []

    # Electricity
    if row["electricity_meter_score"] >= 70:
        reasons.append("strong electricity-meter fit")

    # Gas
    if row["gas_meter_score"] >= 70:
        reasons.append("strong gas-meter fit")

    # Detector
    if row["gas_detector_score"] >= 70:
        reasons.append("strong gas-detector fit")

    # Wholesale
    if row["wholesale_score"] >= 80:
        reasons.append("strong wholesale/distribution potential")

    elif row["wholesale_score"] >= 60:
        reasons.append("moderate resale potential")

    # Electrical
    if row["electrical_related"]:
        reasons.append("electrical-related activity")

    # Industrial
    if row["industrial_related"]:
        reasons.append("industrial-related activity")

    # Rating
    if not pd.isna(row["rating"]) and row["rating"] >= 4.5:
        reasons.append("high customer rating")

    # Reviews
    if not pd.isna(row["review_count"]) and row["review_count"] >= 20:
        reasons.append("significant customer review activity")

    # Website
    if row["has_website"]:
        reasons.append("has website")

    # Phone
    if row["has_phone"]:
        reasons.append("has phone contact")

    return "; ".join(reasons)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 75)
    print("                    UFMEEG PROSPECT SCORING")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading data...")

    try:
        businesses = pd.read_csv(DATA_FILE)
        clusters = pd.read_csv(CLUSTERS_FILE)

    except FileNotFoundError as e:

        print("\nERROR: Could not find one of the required files.")
        print("Expected files:")
        print(f"  - {DATA_FILE}")
        print(f"  - {CLUSTERS_FILE}")
        print(f"\nDetails: {e}")

        return

    print("Data loaded successfully.")

    # --------------------------------------------------------
    # CLEAN COLUMN NAMES
    # --------------------------------------------------------

    businesses.columns = (
        businesses.columns
        .str.replace("**", "", regex=False)
        .str.strip()
    )

    clusters.columns = (
        clusters.columns
        .str.replace("**", "", regex=False)
        .str.strip()
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    boolean_columns = [
        "has_website",
        "has_phone",
        "has_social_media",
        "electrical_related",
        "industrial_related"
    ]

    for column in boolean_columns:

        if column in businesses.columns:

            businesses[column] = (
                businesses[column]
                .astype(str)
                .str.lower()
                .map({
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False
                })
                .fillna(False)
            )

    businesses["rating"] = pd.to_numeric(
        businesses["rating"],
        errors="coerce"
    )

    businesses["review_count"] = pd.to_numeric(
        businesses["review_count"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # MERGE CLUSTER INFORMATION
    # --------------------------------------------------------

    if "business_id" in clusters.columns:

        cluster_columns = [
            column for column in [
                "business_id",
                "cluster"
            ]
            if column in clusters.columns
        ]

        if len(cluster_columns) == 2:

            businesses = businesses.merge(
                clusters[cluster_columns],
                on="business_id",
                how="left"
            )

    # If cluster information does not exist
    if "cluster" not in businesses.columns:
        businesses["cluster"] = -1

    # --------------------------------------------------------
    # CALCULATE PRODUCT FIT
    # --------------------------------------------------------

    print("\nCalculating product relevance...")

    product_scores = businesses.apply(
        calculate_product_fit,
        axis=1,
        result_type="expand"
    )

    product_scores.columns = [
        "electricity_meter_score",
        "gas_meter_score",
        "gas_detector_score"
    ]

    businesses = pd.concat(
        [businesses, product_scores],
        axis=1
    )

    # --------------------------------------------------------
    # WHOLESALE SCORE
    # --------------------------------------------------------

    businesses["wholesale_score"] = businesses.apply(
        wholesale_score,
        axis=1
    )

    # --------------------------------------------------------
    # BUSINESS QUALITY
    # --------------------------------------------------------

    businesses["business_quality_score"] = businesses.apply(
        business_quality_score,
        axis=1
    )

    # --------------------------------------------------------
    # BUSINESS PRESENCE
    # --------------------------------------------------------

    businesses["business_presence_score"] = businesses.apply(
        business_presence_score,
        axis=1
    )

    # --------------------------------------------------------
    # FINAL PROSPECT SCORE
    # --------------------------------------------------------

    businesses["prospect_score"] = businesses.apply(
        calculate_prospect_score,
        axis=1
    )

    # --------------------------------------------------------
    # PROSPECT LEVEL
    # --------------------------------------------------------

    businesses["prospect_level"] = businesses[
        "prospect_score"
    ].apply(prospect_level)

    # --------------------------------------------------------
    # BEST PRODUCT
    # --------------------------------------------------------

    businesses["best_product"] = businesses.apply(
        get_best_product,
        axis=1
    )

    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    businesses["reasons"] = businesses.apply(
        generate_reasons,
        axis=1
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    businesses = businesses.sort_values(
        by="prospect_score",
        ascending=False
    ).reset_index(drop=True)

    # Rank
    businesses["rank"] = businesses.index + 1

    # --------------------------------------------------------
    # DISPLAY TOP PROSPECTS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TOP 20 PROSPECTS")
    print("=" * 75)

    display_columns = [
        "rank",
        "business_name",
        "category",
        "prospect_score",
        "prospect_level",
        "best_product"
    ]

    print(
        businesses[
            display_columns
        ].head(TOP_N).to_string(index=False)
    )

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("PROSPECT DISTRIBUTION")
    print("=" * 75)

    print(
        businesses["prospect_level"]
        .value_counts()
        .reindex(
            ["HIGH", "MEDIUM", "LOW"],
            fill_value=0
        )
    )

    # --------------------------------------------------------
    # PRODUCT OVERVIEW
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("PRODUCT FIT OVERVIEW")
    print("=" * 75)

    print(
        f"\nStrong electricity-meter prospects : "
        f"{(businesses['electricity_meter_score'] >= 70).sum()}"
    )

    print(
        f"Strong gas-meter prospects         : "
        f"{(businesses['gas_meter_score'] >= 70).sum()}"
    )

    print(
        f"Strong gas-detector prospects      : "
        f"{(businesses['gas_detector_score'] >= 70).sum()}"
    )

    print(
        f"Strong wholesale prospects         : "
        f"{(businesses['wholesale_score'] >= 70).sum()}"
    )

    # --------------------------------------------------------
    # TOP 10 DETAILS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TOP 10 PROSPECT DETAILS")
    print("=" * 75)

    for _, row in businesses.head(10).iterrows():

        print("\n" + "-" * 75)

        print(f"Rank       : {int(row['rank'])}")
        print(f"Business   : {row['business_name']}")
        print(f"Category   : {row['category']}")
        print(f"Cluster    : {row['cluster']}")
        print(f"Score      : {row['prospect_score']:.2f}")
        print(f"Level      : {row['prospect_level']}")
        print(f"Best product: {row['best_product']}")

        print("\nProduct fit:")

        print(
            f"  Electricity meters : "
            f"{row['electricity_meter_score']:.1f}/100"
        )

        print(
            f"  Gas meters         : "
            f"{row['gas_meter_score']:.1f}/100"
        )

        print(
            f"  Gas leak detectors : "
            f"{row['gas_detector_score']:.1f}/100"
        )

        print(
            f"  Wholesale potential: "
            f"{row['wholesale_score']:.1f}/100"
        )

        print("\nReasons:")

        if row["reasons"]:
            for reason in row["reasons"].split("; "):
                print(f"  {reason}")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_columns = [
        "rank",
        "business_id",
        "business_name",
        "category",
        "city",
        "wilaya",
        "cluster",

        "rating",
        "review_count",

        "has_website",
        "has_phone",
        "has_social_media",

        "electrical_related",
        "industrial_related",

        "electricity_meter_score",
        "gas_meter_score",
        "gas_detector_score",
        "wholesale_score",

        "business_quality_score",
        "business_presence_score",

        "prospect_score",
        "prospect_level",
        "best_product",
        "reasons"
    ]

    # Keep only columns that exist
    output_columns = [
        column
        for column in output_columns
        if column in businesses.columns
    ]

    businesses[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 75)
    print("PROSPECT SCORING COMPLETED")
    print("=" * 75)

    print("\nResults saved to:")
    print(f"  {OUTPUT_FILE}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()