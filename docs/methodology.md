# UFMEEG Customer Segmentation & Prospect Scoring — Methodology

## 1. Methodology Overview

The objective of this project is to identify and prioritize potential B2B clients for UFMEEG/SAIEG based on business characteristics collected from publicly available business information.

UFMEEG's relevant products include:

* Electricity meters
* Gas meters
* Gas leak detectors

The project focuses particularly on identifying businesses that could act as commercial clients or distributors, such as wholesalers, electrical suppliers, industrial suppliers, and related businesses.

The methodology follows these stages:

```text
1. Data Collection
        ↓
2. Data Cleaning
        ↓
3. Exploratory Data Analysis
        ↓
4. Feature Engineering
        ↓
5. Business Clustering
        ↓
6. Cluster Evaluation
        ↓
7. Cluster Interpretation
        ↓
8. PCA Visualization
        ↓
9. Product Relevance Analysis
        ↓
10. Prospect Scoring
        ↓
11. Dashboard Generation
```

---

# 2. Data Collection

## 2.1 Geographic Scope

The initial data collection was limited to **Sétif Wilaya, Algeria**.

This limitation was intentional.

Instead of collecting a very large dataset across Algeria immediately, the project first uses a smaller geographic area to validate the complete analytical pipeline.

Once the methodology is validated, the same process can be extended to additional Wilayas.

---

## 2.2 Data Source

Business information was collected manually from publicly available online business listings.

The raw data was stored in:

```text
data/raw/businesses_raw.csv
```

The collection process and observations are documented in:

```text
data/raw/collection_notes.md
```

---

## 2.3 Collected Variables

The dataset contains variables describing each business:

| Variable                | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `business_id`           | Unique identifier                                               |
| `business_name`         | Name of the business                                            |
| `category`              | Business category                                               |
| `city`                  | City                                                            |
| `wilaya`                | Wilaya                                                          |
| `rating`                | Customer rating                                                 |
| `review_count`          | Number of reviews                                               |
| `has_website`           | Whether a website is present                                    |
| `has_phone`             | Whether a phone number is available                             |
| `has_social_media`      | Whether social media is present                                 |
| `social_platform_count` | Number of identified social platforms                           |
| `electrical_related`    | Whether the activity is related to electrical products/services |
| `industrial_related`    | Whether the activity is related to industrial activity          |
| `latitude`              | Geographic latitude                                             |
| `longitude`             | Geographic longitude                                            |

---

# 3. Data Cleaning

The collected data cannot be directly used for analysis because raw business information may contain missing values, inconsistent data types, formatting problems, and categorical variables.

The cleaned dataset is stored in:

```text
data/processed/businesses_clean.csv
```

---

## 3.1 Missing Values

Missing information was handled according to the meaning of each variable.

For example:

* Missing ratings are not interpreted as bad ratings.
* A business with zero reviews is distinguished from a business with missing review information.
* Missing Boolean information is treated carefully rather than automatically assuming the business has or does not have a particular feature.

This distinction is important because:

```text
No reviews ≠ Bad reviews
No website found ≠ Confirmed absence of a website
```

---

## 3.2 Boolean Variables

Boolean characteristics were converted into numerical representations where necessary.

For example:

```text
False → 0
True  → 1
```

This allows machine-learning algorithms to process these variables numerically.

---

## 3.3 Numerical Variables

Numerical variables such as rating and review count were converted into appropriate numerical data types.

Review counts can have a highly skewed distribution because some businesses have many more reviews than others.

For this reason, a logarithmic transformation can be applied during feature engineering.

---

# 4. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed before machine learning.

The purpose of EDA is to understand the dataset and identify important patterns before building the clustering model.

The analysis examines:

* Number of businesses
* Business categories
* Geographic distribution
* Rating distribution
* Review-count distribution
* Website presence
* Phone availability
* Social-media presence
* Electrical-related activity
* Industrial-related activity

EDA also helps identify potential problems in the data and determine which variables may be useful for segmentation.

The exploratory analysis is available in:

```text
notebooks/exploration.ipynb
```

---

# 5. Feature Engineering

Machine-learning algorithms operate on numerical features rather than raw textual business descriptions.

Feature engineering transforms the cleaned business data into a representation suitable for machine learning.

The resulting dataset is:

```text
data/features/ml_features.csv
```

---

## 5.1 Selected Features

The clustering representation includes relevant business characteristics such as:

* Rating
* Review activity
* Website presence
* Phone presence
* Social-media presence
* Social-platform count
* Electrical-related activity
* Industrial-related activity

The exact feature set is defined by the feature-engineering implementation.

---

## 5.2 Rating Standardization

Ratings are transformed into a standardized numerical representation so that the scale of the rating does not dominate the other variables.

Standardization generally follows:

```text
z = (x - mean) / standard_deviation
```

A standardized value indicates how far a business's value is from the dataset average.

---

## 5.3 Review Count Transformation

Review counts can be highly skewed.

For example, a dataset may contain businesses with:

```text
1 review
5 reviews
20 reviews
100 reviews
1000 reviews
```

Using the raw values could give businesses with very large review counts disproportionate influence.

A logarithmic transformation can therefore be used:

```text
log_review_count = log(1 + review_count)
```

The `+1` ensures that businesses with zero reviews can also be transformed.

---

## 5.4 Scaling

Because clustering is distance-based, features should be placed on comparable scales.

Standardization prevents a feature with a numerically larger range from dominating the distance calculation.

---

# 6. Business Clustering

## 6.1 Algorithm

The project uses **K-Means clustering**.

K-Means attempts to divide the businesses into `K` groups such that businesses within the same group are relatively similar according to the selected features.

Conceptually:

```text
Business features
       ↓
Distance calculation
       ↓
Cluster assignment
       ↓
Centroid update
       ↓
Repeat until convergence
```

---

## 6.2 Why K-Means?

K-Means was selected because:

* The dataset is relatively small.
* The clustering features are numerical.
* The algorithm is computationally efficient.
* The resulting clusters are relatively easy to interpret.
* It provides a useful baseline for business segmentation.

The algorithm does not know what a "good client" is.

It only identifies groups of businesses with similar characteristics.

This distinction is important.

---

# 7. Choosing the Number of Clusters

The number of clusters `K` must be selected before running K-Means.

Two evaluation approaches were used.

---

## 7.1 Elbow Method

The Elbow Method evaluates the within-cluster variation for different values of `K`.

As the number of clusters increases, within-cluster variation generally decreases.

The objective is to identify a point where increasing the number of clusters provides diminishing improvement.

The resulting visualization is:

```text
results/figures/elbow_method.png
```

---

## 7.2 Silhouette Analysis

The silhouette score evaluates how well each business fits within its assigned cluster compared with other clusters.

The score generally ranges from:

```text
-1 to +1
```

Higher values indicate better separation and cohesion.

The resulting visualization is:

```text
results/figures/silhouette_scores.png
```

The evaluation results are stored in:

```text
results/clustering_metrics.csv
```

---

# 8. Final Clustering

The selected configuration contains:

```text
K = 8 clusters
```

Each business receives a cluster identifier:

```text
0
1
2
3
4
5
6
7
```

The assignments are stored in:

```text
results/clusters.csv
```

The cluster identifier itself has no inherent business meaning.

For example:

```text
Cluster 3
```

does not automatically mean "bad prospects."

Its meaning must be determined by examining the characteristics of the businesses assigned to that cluster.

---

# 9. Cluster Interpretation

After clustering, each cluster is analyzed to understand what distinguishes it from the others.

The analysis examines:

* Cluster size
* Average rating
* Review activity
* Website presence
* Phone presence
* Social-media presence
* Business categories
* Electrical-related activity
* Industrial-related activity

Generated files include:

```text
results/cluster_summary.csv
results/cluster_profiles.csv
results/cluster_category_composition.csv
results/cluster_business_examples.csv
```

The purpose of this stage is to convert mathematical groups into **business-interpretable segments**.

---

# 10. PCA Dimensionality Reduction

## 10.1 Purpose

The clustering model operates in a multidimensional feature space.

If eight or more features are used, it is impossible to visualize all dimensions simultaneously in a normal two-dimensional graph.

Principal Component Analysis (PCA) is therefore used to create a two-dimensional representation.

---

## 10.2 PC1 and PC2

PCA generates new variables called principal components.

### PC1

PC1 is the direction that captures the largest amount of variation in the feature data.

### PC2

PC2 captures the next largest amount of variation while being orthogonal to PC1.

Therefore:

```text
Original features
       ↓
      PCA
       ↓
   PC1 + PC2
       ↓
2D visualization
```

PC1 and PC2 are not individual original variables such as "rating" or "reviews."

They are combinations of the original features.

---

## 10.3 Purpose of PCA in This Project

PCA is mainly used for:

* Visualization
* Understanding cluster separation
* Identifying overlapping clusters
* Communicating the clustering results

PCA does not replace the clustering model.

The clustering is performed using the engineered feature representation, while PCA provides a visual projection of that representation.

The generated files are:

```text
results/pca_coordinates.csv
results/figures/pca_clusters.png
```

---

# 11. Prospect Identification

Clustering and prospect scoring serve different purposes.

### Clustering asks:

> Which businesses have similar characteristics?

### Prospect scoring asks:

> Which businesses appear most commercially relevant to UFMEEG?

Therefore, clustering alone cannot determine the best clients.

A separate prospect-scoring stage is used.

---

# 12. Product Relevance

The scoring system considers the relationship between business characteristics and UFMEEG's products.

---

## 12.1 Electricity Meters

Potentially relevant businesses include:

* Electrical equipment suppliers
* Electrical supply stores
* Electrical wholesalers
* Electrical distributors
* Electrical installation businesses
* Industrial electrical businesses

These businesses may already operate in markets related to electrical equipment and may therefore represent potential commercial opportunities.

---

## 12.2 Gas Meters

Gas-meter relevance is more difficult to determine from the current dataset because explicit gas-related business information is limited.

Therefore, gas-meter relevance should be interpreted cautiously.

Future data collection should explicitly identify:

* Gas equipment suppliers
* Gas distributors
* Plumbing/gas businesses
* Gas installation companies
* Businesses selling gas-related equipment

---

## 12.3 Gas Leak Detectors

Potentially relevant businesses include companies involved in:

* Industrial equipment
* Maintenance
* Electrical equipment
* Technical services
* Construction
* Equipment supply

However, business category alone does not prove that a company is a buyer of gas leak detectors.

The score therefore represents **potential relevance**, not confirmed demand.

---

# 13. Wholesale Potential

Wholesale/distribution potential is an important component of the project because the commercial objective includes selling products to wholesalers.

Businesses such as:

* Wholesalers
* Electrical distributors
* Equipment suppliers
* Large supply stores

can potentially provide greater distribution opportunities than individual end users.

Therefore, wholesale potential is explicitly represented in the prospect-scoring methodology.

---

# 14. Prospect Scoring

The prospect-scoring stage combines multiple indicators into a single score.

The score considers factors such as:

* Electricity-meter relevance
* Gas-meter relevance
* Gas-leak-detector relevance
* Wholesale potential
* Electrical-related activity
* Industrial-related activity
* Customer rating
* Review activity
* Cluster characteristics

The result is a score representing the **relative priority of each business for commercial investigation**.

The output is:

```text
results/prospect_scores.csv
```

---

# 15. Prospect Levels

The numerical score is converted into priority levels.

The current system uses:

```text
HIGH
MEDIUM
LOW
```

These levels should be interpreted as:

### HIGH

Businesses with strong indicators of commercial relevance.

These prospects should generally be investigated first.

### MEDIUM

Businesses with meaningful potential but weaker or less complete indicators.

These can be considered after the high-priority group.

### LOW

Businesses for which the available data provides limited evidence of strong commercial relevance.

They are not necessarily bad prospects; they simply have lower priority according to the current information.

---

# 16. Important Distinction: Score vs Actual Customer

The prospect score does **not** predict whether a business will purchase UFMEEG products.

It is a prioritization mechanism.

For example:

```text
High score
     ↓
High potential according to available data
     ↓
Contact/investigate prospect
     ↓
Actual commercial response
     ↓
Confirmed opportunity or rejection
```

A high-scoring business may still reject the products.

Conversely, a low-scoring business could become a customer.

The model therefore supports human decision-making rather than replacing it.

---

# 17. Dashboard

The final stage combines the results into an HTML dashboard.

The dashboard provides a high-level view of:

* Dataset statistics
* Cluster distribution
* Cluster characteristics
* Prospect rankings
* Product relevance
* Business information
* Analytical visualizations

The generated dashboard is:

```text
results/UFMEEG_dashboard.html
```

The dashboard is intended to make the analytical results accessible to non-technical users.

---

# 18. End-to-End Data Flow

The complete data flow can be summarized as:

```text
                    RAW DATA
                       │
                       ▼
             businesses_raw.csv
                       │
                       ▼
                DATA CLEANING
                       │
                       ▼
             businesses_clean.csv
                       │
                       ▼
             FEATURE ENGINEERING
                       │
                       ▼
               ml_features.csv
                       │
                       ▼
                K-MEANS MODEL
                       │
                       ▼
                  clusters.csv
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
     CLUSTER ANALYSIS           PCA
             │                   │
             ▼                   ▼
    Cluster profiles      PCA coordinates
    Cluster summaries            │
             │                   ▼
             │             PCA visualization
             │
             └──────────┐
                        ▼
                PROSPECT SCORING
                        │
                        ▼
               prospect_scores.csv
                        │
                        ▼
                  FINAL DASHBOARD
                        │
                        ▼
              UFMEEG_dashboard.html
```

---

# 19. Limitations

Several limitations should be considered when interpreting the results.

## 19.1 Dataset Size

The current dataset contains 121 businesses.

This is sufficient for developing and testing the methodology but is relatively small for building a highly generalizable commercial prediction model.

---

## 19.2 Geographic Coverage

The current dataset focuses on Sétif Wilaya.

The results therefore should not automatically be generalized to all of Algeria.

---

## 19.3 Manual Data Collection

Manual collection introduces possible:

* Human errors
* Inconsistent categorization
* Missing information
* Duplicate businesses
* Subjective classification

Future automated or semi-automated collection could improve consistency.

---

## 19.4 Limited Commercial Data

The current dataset primarily describes publicly observable business characteristics.

It does not contain historical information such as:

* Previous UFMEEG purchases
* Purchase volume
* Revenue
* Number of meters purchased
* Previous quotations
* Sales conversion
* Customer lifetime value

Therefore, the current prospect score is primarily **rule-based and feature-based**, rather than trained from historical sales outcomes.

---

# 20. Future Methodological Improvements

The methodology can be improved substantially if additional data becomes available.

## 20.1 Expand the Geographic Scope

The collection can be extended from Sétif to additional Wilayas.

For example:

```text
Sétif
   ↓
Eastern Algeria
   ↓
Multiple Wilayas
   ↓
National dataset
```

---

## 20.2 Improve Business Classification

Generic online categories should be mapped into more meaningful commercial segments.

For example:

```text
Electrical Distributor
Electrical Wholesaler
Electrical Retailer
Industrial Supplier
Construction Company
Engineering Company
Maintenance Company
Real Estate
Other
```

This would improve interpretability.

---

## 20.3 Add Explicit Product-Related Features

Future data collection could identify whether a business explicitly sells:

```text
Electricity meters
Gas meters
Gas leak detectors
Electrical equipment
Gas equipment
Industrial equipment
```

This would make product relevance much more accurate.

---

## 20.4 Incorporate Real Sales Data

The strongest future improvement would be to connect the prospecting system with actual commercial outcomes.

For example:

```text
Prospect
   ↓
Contacted
   ↓
Responded
   ↓
Quotation requested
   ↓
Quotation accepted
   ↓
Purchase
```

Once enough historical data is available, the prospect-scoring model could be trained using actual business outcomes.

This could eventually transform the current prioritization system into a supervised machine-learning model for lead conversion prediction.

---

# 21. Reproducibility

The project is organized so that each stage can be executed independently.

The main scripts are located in:

```text
src/
```

The analytical pipeline can be reproduced by running the scripts in the appropriate order:

```bash
python src/final_analysis.py
python src/cluster_summary.py
python src/dimensionality_reduction.py
python src/display_clusters.py
python src/prospect_scoring.py
python src/dashboard_report.py
```

Generated datasets and visualizations are stored in:

```text
results/
```

---

# 22. Methodological Summary

The methodology can be summarized in five major analytical concepts:

### 1. Data Preparation

Transform raw business information into a reliable analytical dataset.

### 2. Feature Engineering

Convert business characteristics into numerical variables suitable for machine learning.

### 3. Clustering

Use K-Means to discover groups of businesses with similar characteristics.

### 4. Prospect Scoring

Use product relevance and commercial indicators to prioritize potential clients.

### 5. Visualization and Decision Support

Use PCA and the final dashboard to make the results easier to understand and use.

The fundamental principle of the project is:

```text
Raw business data
        ↓
Structured information
        ↓
Business segments
        ↓
Commercial relevance
        ↓
Prioritized prospects
```

The final objective is to transform a large and initially unstructured list of businesses into **actionable information that can support UFMEEG's B2B prospecting strategy**.
