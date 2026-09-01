"""
UFMEEG - Final Dashboard Report

Generates a standalone HTML dashboard from:
    data/processed/businesses_clean.csv
    results/clusters.csv
    results/prospect_scores.csv
    results/pca_coordinates.csv

Run from the project root:
    python src/dashboard_report.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import html
import webbrowser
import json

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BUSINESSES_FILE = PROJECT_ROOT / "data" / "processed" / "businesses_clean.csv"
CLUSTERS_FILE = PROJECT_ROOT / "results" / "clusters.csv"
PROSPECTS_FILE = PROJECT_ROOT / "results" / "prospect_scores.csv"
PCA_FILE = PROJECT_ROOT / "results" / "pca_coordinates.csv"

OUTPUT_FILE = PROJECT_ROOT / "results" / "UFMEEG_dashboard.html"


# ============================================================
# HELPERS
# ============================================================

def safe_read_csv(path):
    """Read a CSV file and return a dataframe."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def clean_column_names(df):
    """Remove accidental markdown markers from column names."""
    df = df.copy()

    df.columns = (
        df.columns
        .str.replace("**", "", regex=False)
        .str.strip()
    )

    return df


def esc(value):
    """Safely escape text for HTML."""
    if pd.isna(value):
        return ""

    return html.escape(str(value))


def number(value, decimals=1):
    """Format numeric values."""
    if pd.isna(value):
        return "N/A"

    return f"{float(value):,.{decimals}f}"


def percentage(value):
    """Format a value as a percentage."""
    if pd.isna(value):
        return "N/A"

    value = float(value)

    # Values such as 0.85 -> 85%
    if value <= 1:
        value *= 100

    return f"{value:.1f}%"


def get_column(df, possible_names, default=None):
    """Return the first matching column."""
    for name in possible_names:
        if name in df.columns:
            return name

    return default


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("                 UFMEEG FINAL DASHBOARD")
print("=" * 70)
print()
print("Loading data...")


try:
    businesses = clean_column_names(safe_read_csv(BUSINESSES_FILE))
    clusters = clean_column_names(safe_read_csv(CLUSTERS_FILE))
    prospects = clean_column_names(safe_read_csv(PROSPECTS_FILE))
    pca = clean_column_names(safe_read_csv(PCA_FILE))

except FileNotFoundError as error:
    print()
    print("ERROR: Required file not found.")
    print(error)
    print()
    print("Expected files:")
    print(f"  - {BUSINESSES_FILE}")
    print(f"  - {CLUSTERS_FILE}")
    print(f"  - {PROSPECTS_FILE}")
    print(f"  - {PCA_FILE}")
    raise SystemExit(1)


print("Data loaded successfully.")
print()


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

# Business ID
business_id_col = get_column(
    businesses,
    ["business_id", "id", "businessId"]
)

# Business name
business_name_col = get_column(
    businesses,
    ["business_name", "name", "businessName"]
)

# Category
category_col = get_column(
    businesses,
    ["category", "main_category"]
)

# Cluster column
cluster_col = get_column(
    clusters,
    ["cluster", "Cluster", "cluster_id"]
)

# Prospect score
score_col = get_column(
    prospects,
    ["prospect_score", "score", "Prospect Score"]
)

# Prospect level
level_col = get_column(
    prospects,
    ["prospect_level", "level", "Prospect Level"]
)


# ============================================================
# PREPARE DATA
# ============================================================

# Convert IDs to strings where available
if business_id_col:
    businesses[business_id_col] = businesses[business_id_col].astype(str)

if business_id_col in clusters.columns:
    clusters[business_id_col] = clusters[business_id_col].astype(str)

if business_id_col in prospects.columns:
    prospects[business_id_col] = prospects[business_id_col].astype(str)


# ------------------------------------------------------------
# MERGE CLUSTER DATA WITH BUSINESS DATA
# ------------------------------------------------------------

if business_id_col and business_id_col in clusters.columns:

    cluster_data = businesses.merge(
        clusters[[business_id_col, cluster_col]],
        on=business_id_col,
        how="left"
    )

else:
    cluster_data = businesses.copy()

    if cluster_col not in cluster_data.columns:
        cluster_data[cluster_col] = "Unknown"


# ------------------------------------------------------------
# MERGE PROSPECT DATA
# ------------------------------------------------------------

if (
    business_id_col
    and business_id_col in prospects.columns
    and business_id_col in cluster_data.columns
):

    prospect_columns = [
        business_id_col
    ]

    for col in prospects.columns:
        if col not in prospect_columns:
            prospect_columns.append(col)

    final_data = cluster_data.merge(
        prospects[prospect_columns],
        on=business_id_col,
        how="left",
        suffixes=("", "_prospect")
    )

else:
    final_data = cluster_data.copy()


# ============================================================
# BASIC STATISTICS
# ============================================================

total_businesses = len(businesses)

if cluster_col in final_data.columns:
    number_of_clusters = final_data[cluster_col].nunique()
else:
    number_of_clusters = 0


# Prospect distribution
if level_col in final_data.columns:

    prospect_distribution = (
        final_data[level_col]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .value_counts()
    )

else:
    prospect_distribution = pd.Series(dtype=int)


high_count = int(prospect_distribution.get("HIGH", 0))
medium_count = int(prospect_distribution.get("MEDIUM", 0))
low_count = int(prospect_distribution.get("LOW", 0))


# ============================================================
# PRODUCT FIT
# ============================================================

product_fit_columns = {
    "Electricity meters": [
        "electricity_meter_fit",
        "electricity_meter_score",
        "electricity_fit"
    ],
    "Gas meters": [
        "gas_meter_fit",
        "gas_meter_score",
        "gas_fit"
    ],
    "Gas leak detectors": [
        "gas_detector_fit",
        "gas_detector_score",
        "gas_leak_detector_fit",
        "gas_leak_detector_score"
    ],
    "Wholesale potential": [
        "wholesale_potential",
        "wholesale_score",
        "wholesale_fit"
    ]
}


product_fit_values = {}

for product_name, possible_columns in product_fit_columns.items():

    column = get_column(
        final_data,
        possible_columns
    )

    if column:
        numeric_values = pd.to_numeric(
            final_data[column],
            errors="coerce"
        )

        product_fit_values[product_name] = float(
            numeric_values.mean()
        )

    else:
        product_fit_values[product_name] = None


# ============================================================
# TOP PROSPECTS
# ============================================================

if score_col in final_data.columns:

    final_data[score_col] = pd.to_numeric(
        final_data[score_col],
        errors="coerce"
    )

    top_prospects = (
        final_data
        .dropna(subset=[score_col])
        .sort_values(score_col, ascending=False)
        .head(20)
        .copy()
    )

else:
    top_prospects = pd.DataFrame()


# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

if cluster_col in final_data.columns:

    cluster_distribution = (
        final_data[cluster_col]
        .value_counts()
        .sort_index()
    )

else:
    cluster_distribution = pd.Series(dtype=int)


# ============================================================
# PCA DATA
# ============================================================

pca_x_col = get_column(
    pca,
    ["PC1", "pca1", "PCA1", "component_1"]
)

pca_y_col = get_column(
    pca,
    ["PC2", "pca2", "PCA2", "component_2"]
)

pca_cluster_col = get_column(
    pca,
    ["cluster", "Cluster", "cluster_id"]
)

pca_id_col = get_column(
    pca,
    ["business_id", "id", "businessId"]
)


# ============================================================
# BUILD PCA POINTS
# ============================================================

pca_points = []

if pca_x_col and pca_y_col:

    for _, row in pca.iterrows():

        try:
            x = float(row[pca_x_col])
            y = float(row[pca_y_col])
        except (ValueError, TypeError):
            continue

        cluster_value = (
            row[pca_cluster_col]
            if pca_cluster_col
            else "Unknown"
        )

        name = ""

        if (
            pca_id_col
            and business_id_col
            and business_name_col
        ):

            matching = businesses[
                businesses[business_id_col].astype(str)
                == str(row[pca_id_col])
            ]

            if not matching.empty:
                name = str(
                    matching.iloc[0][business_name_col]
                )

        pca_points.append({
            "x": x,
            "y": y,
            "cluster": str(cluster_value),
            "name": name
        })


# ============================================================
# GENERATE CLUSTER TABLE
# ============================================================

cluster_rows = ""

for cluster_id, count in cluster_distribution.items():

    cluster_subset = final_data[
        final_data[cluster_col] == cluster_id
    ]

    avg_rating = np.nan

    if "rating" in cluster_subset.columns:
        avg_rating = pd.to_numeric(
            cluster_subset["rating"],
            errors="coerce"
        ).mean()

    website_pct = np.nan

    if "has_website" in cluster_subset.columns:
        website_pct = (
            cluster_subset["has_website"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .mean()
            * 100
        )

    phone_pct = np.nan

    if "has_phone" in cluster_subset.columns:
        phone_pct = (
            cluster_subset["has_phone"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .mean()
            * 100
        )

    cluster_rows += f"""
    <tr>
        <td><strong>Cluster {esc(cluster_id)}</strong></td>
        <td>{int(count)}</td>
        <td>{number(avg_rating, 2)}</td>
        <td>{percentage(website_pct)}</td>
        <td>{percentage(phone_pct)}</td>
    </tr>
    """


# ============================================================
# GENERATE TOP PROSPECT TABLE
# ============================================================

prospect_rows = ""

for rank, (_, row) in enumerate(
    top_prospects.iterrows(),
    start=1
):

    name = (
        row[business_name_col]
        if business_name_col in row
        else "Unknown"
    )

    category = (
        row[category_col]
        if category_col in row
        else "Unknown"
    )

    score = row[score_col]

    level = (
        row[level_col]
        if level_col and level_col in row
        else ""
    )

    cluster = (
        row[cluster_col]
        if cluster_col in row
        else "N/A"
    )

    level_upper = str(level).upper()

    badge_class = (
        "high"
        if level_upper == "HIGH"
        else "medium"
        if level_upper == "MEDIUM"
        else "low"
    )

    prospect_rows += f"""
    <tr>
        <td>{rank}</td>
        <td><strong>{esc(name)}</strong></td>
        <td>{esc(category)}</td>
        <td>Cluster {esc(cluster)}</td>
        <td><strong>{number(score, 2)}</strong></td>
        <td>
            <span class="badge {badge_class}">
                {esc(level_upper)}
            </span>
        </td>
    </tr>
    """


# ============================================================
# GENERATE PRODUCT FIT TABLE
# ============================================================

product_rows = ""

for product_name, value in product_fit_values.items():

    if value is None:
        display_value = "N/A"
        width = 0
    else:
        display_value = f"{value:.1f}/100"
        width = max(0, min(100, value))

    product_rows += f"""
    <div class="product-row">
        <div class="product-label">
            <span>{esc(product_name)}</span>
            <strong>{display_value}</strong>
        </div>

        <div class="progress">
            <div class="progress-bar"
                 style="width:{width}%">
            </div>
        </div>
    </div>
    """


# ============================================================
# PCA JAVASCRIPT DATA
# ============================================================

pca_js = []

for point in pca_points:

    pca_js.append(
        "{"
        f"x:{point['x']},"
        f"y:{point['y']},"
        f"cluster:{json.dumps(point['cluster'])},"
        f"name:{json.dumps(point['name'])}"
        "}"
    )

pca_js_string = ",\n".join(pca_js)


# ============================================================
# CLUSTER CHART DATA
# ============================================================

cluster_labels_js = ", ".join(
    json.dumps(f"Cluster {cluster_id}")
    for cluster_id in cluster_distribution.index
)

cluster_values_js = ", ".join(
    str(int(value))
    for value in cluster_distribution.values
)


# ============================================================
# PROSPECT DISTRIBUTION DATA
# ============================================================

prospect_labels = ["HIGH", "MEDIUM", "LOW"]

prospect_values = [
    high_count,
    medium_count,
    low_count
]


# ============================================================
# HTML DASHBOARD
# ============================================================

html_content = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>UFMEEG Customer Segmentation Dashboard</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;

    background: #f4f6f9;
    color: #1f2937;
}}

.header {{
    background: #111827;
    color: white;
    padding: 32px 45px;
}}

.header h1 {{
    margin: 0 0 8px 0;
    font-size: 30px;
}}

.header p {{
    margin: 0;
    color: #cbd5e1;
}}

.container {{
    max-width: 1400px;
    margin: auto;
    padding: 30px;
}}

.section-title {{
    margin-top: 35px;
    margin-bottom: 18px;
    font-size: 22px;
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(210px, 1fr));

    gap: 18px;
}}

.card {{
    background: white;
    border-radius: 12px;
    padding: 22px;
    box-shadow:
        0 2px 8px rgba(0,0,0,0.06);
}}

.card-title {{
    font-size: 13px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: .05em;
}}

.card-value {{
    font-size: 30px;
    font-weight: 700;
    margin-top: 8px;
}}

.card-description {{
    margin-top: 5px;
    font-size: 13px;
    color: #6b7280;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(420px, 1fr));

    gap: 22px;
}}

.panel {{
    background: white;
    border-radius: 12px;
    padding: 22px;
    margin-top: 22px;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.06);
}}

.panel h3 {{
    margin-top: 0;
}}

.chart-container {{
    position: relative;
    height: 330px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    text-align: left;
    background: #f8fafc;
    padding: 12px;
    font-size: 13px;
    color: #475569;
}}

td {{
    padding: 12px;
    border-top: 1px solid #e5e7eb;
    font-size: 14px;
}}

tr:hover {{
    background: #f8fafc;
}}

.badge {{
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
}}

.badge.high {{
    background: #dcfce7;
    color: #166534;
}}

.badge.medium {{
    background: #fef3c7;
    color: #92400e;
}}

.badge.low {{
    background: #fee2e2;
    color: #991b1b;
}}

.product-row {{
    margin-bottom: 20px;
}}

.product-label {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 14px;
}}

.progress {{
    height: 10px;
    background: #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
}}

.progress-bar {{
    height: 100%;
    background: #2563eb;
    border-radius: 10px;
}}

.insight {{
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 15px 18px;
    margin: 10px 0;
    border-radius: 5px;
}}

.footer {{
    margin-top: 40px;
    padding: 25px;
    text-align: center;
    color: #64748b;
    font-size: 13px;
}}

@media (max-width: 700px) {{

    .container {{
        padding: 15px;
    }}

    .header {{
        padding: 25px;
    }}

    .grid {{
        grid-template-columns: 1fr;
    }}

}}

</style>

</head>


<body>


<div class="header">

    <h1>UFMEEG Customer Segmentation & Prospect Analysis</h1>

    <p>
        Data-driven identification of potential B2B customers
        for electricity meters, gas meters and gas leak detectors.
    </p>

</div>


<div class="container">


<!-- ====================================================== -->
<!-- EXECUTIVE SUMMARY -->
<!-- ====================================================== -->

<h2 class="section-title">
    Executive Summary
</h2>


<div class="cards">

    <div class="card">
        <div class="card-title">
            Businesses analyzed
        </div>

        <div class="card-value">
            {total_businesses}
        </div>

        <div class="card-description">
            Businesses in the collected dataset
        </div>
    </div>


    <div class="card">
        <div class="card-title">
            Clusters
        </div>

        <div class="card-value">
            {number_of_clusters}
        </div>

        <div class="card-description">
            Customer segments identified
        </div>
    </div>


    <div class="card">
        <div class="card-title">
            High prospects
        </div>

        <div class="card-value">
            {high_count}
        </div>

        <div class="card-description">
            Priority targets for sales
        </div>
    </div>


    <div class="card">
        <div class="card-title">
            Medium prospects
        </div>

        <div class="card-value">
            {medium_count}
        </div>

        <div class="card-description">
            Potential secondary targets
        </div>
    </div>


    <div class="card">
        <div class="card-title">
            Low prospects
        </div>

        <div class="card-value">
            {low_count}
        </div>

        <div class="card-description">
            Lower current priority
        </div>
    </div>

</div>


<!-- ====================================================== -->
<!-- CHARTS -->
<!-- ====================================================== -->

<div class="grid">


    <div class="panel">

        <h3>
            Prospect Distribution
        </h3>

        <div class="chart-container">

            <canvas id="prospectChart"></canvas>

        </div>

    </div>


    <div class="panel">

        <h3>
            Businesses per Cluster
        </h3>

        <div class="chart-container">

            <canvas id="clusterChart"></canvas>

        </div>

    </div>


</div>


<!-- ====================================================== -->
<!-- PRODUCT FIT -->
<!-- ====================================================== -->

<div class="panel">

    <h3>
        Product Fit Overview
    </h3>

    <p style="color:#64748b;">
        Average product relevance across the analyzed businesses.
        Higher values indicate stronger potential fit.
    </p>

    {product_rows}

</div>


<!-- ====================================================== -->
<!-- PCA -->
<!-- ====================================================== -->

<div class="panel">

    <h3>
        Customer Segmentation — PCA Visualization
    </h3>

    <p style="color:#64748b;">
        Each point represents a business. Businesses located near
        each other have similar characteristics according to the
        features used for clustering.
    </p>

    <div class="chart-container"
         style="height:500px;">

        <canvas id="pcaChart"></canvas>

    </div>

</div>


<!-- ====================================================== -->
<!-- TOP PROSPECTS -->
<!-- ====================================================== -->

<div class="panel">

    <h3>
        Top 20 Prospects
    </h3>

    <p style="color:#64748b;">
        Businesses ranked according to the prospect scoring model.
    </p>

    <div style="overflow-x:auto;">

        <table>

            <thead>

                <tr>
                    <th>Rank</th>
                    <th>Business</th>
                    <th>Category</th>
                    <th>Cluster</th>
                    <th>Score</th>
                    <th>Level</th>
                </tr>

            </thead>

            <tbody>

                {prospect_rows}

            </tbody>

        </table>

    </div>

</div>


<!-- ====================================================== -->
<!-- CLUSTER SUMMARY -->
<!-- ====================================================== -->

<div class="panel">

    <h3>
        Cluster Summary
    </h3>

    <p style="color:#64748b;">
        Overview of the main characteristics of each customer segment.
    </p>

    <div style="overflow-x:auto;">

        <table>

            <thead>

                <tr>
                    <th>Cluster</th>
                    <th>Businesses</th>
                    <th>Average Rating</th>
                    <th>Website Presence</th>
                    <th>Phone Presence</th>
                </tr>

            </thead>

            <tbody>

                {cluster_rows}

            </tbody>

        </table>

    </div>

</div>


<!-- ====================================================== -->
<!-- BUSINESS INTERPRETATION -->
<!-- ====================================================== -->

<div class="panel">

    <h3>
        Business Interpretation
    </h3>


    <div class="insight">

        <strong>1. Customer segmentation:</strong>

        The clustering stage groups businesses with similar
        characteristics into {number_of_clusters} segments.
        This allows the sales team to avoid treating every
        business as an identical prospect.

    </div>


    <div class="insight">

        <strong>2. Prospect prioritization:</strong>

        The prospect-scoring stage ranks businesses according
        to their potential commercial value for the factory's
        products.

    </div>


    <div class="insight">

        <strong>3. Sales focus:</strong>

        The highest-ranked businesses should be investigated
        first, especially those whose activities suggest that
        they can distribute or resell electricity meters,
        gas meters and gas leak detectors.

    </div>


    <div class="insight">

        <strong>4. B2B orientation:</strong>

        Wholesale potential is considered because the objective
        is to identify potential clients for the factory's
        products, particularly wholesalers and businesses
        capable of distributing them.

    </div>


    <div class="insight">

        <strong>5. Important limitation:</strong>

        The prospect score is a prioritization model based on
        the available dataset. A high score means that a
        business is a strong candidate for commercial
        investigation; it does not prove that the business
        will purchase the products.

    </div>

</div>


<div class="footer">

    UFMEEG — Customer Segmentation & Prospect Scoring

    <br>

    Generated automatically from the project analysis results.

</div>


</div>


<script>


// ==========================================================
// PROSPECT DISTRIBUTION
// ==========================================================

new Chart(
    document.getElementById("prospectChart"),
    {{
        type: "doughnut",

        data: {{
            labels: {json.dumps(prospect_labels)},

            datasets: [{{
                data: {json.dumps(prospect_values)}
            }}]
        }},

        options: {{
            responsive: true,
            maintainAspectRatio: false,

            plugins: {{
                legend: {{
                    position: "bottom"
                }}
            }}
        }}
    }}
);


// ==========================================================
// CLUSTER DISTRIBUTION
// ==========================================================

new Chart(
    document.getElementById("clusterChart"),
    {{
        type: "bar",

        data: {{
            labels: [{cluster_labels_js}],

            datasets: [{{
                label: "Businesses",

                data: [{cluster_values_js}],

                borderWidth: 1
            }}]
        }},

        options: {{
            responsive: true,
            maintainAspectRatio: false,

            scales: {{
                y: {{
                    beginAtZero: true,

                    ticks: {{
                        precision: 0
                    }}
                }}
            }},

            plugins: {{
                legend: {{
                    display: false
                }}
            }}
        }}
    }}
);


// ==========================================================
// PCA SCATTER PLOT
// ==========================================================

const pcaPoints = [
    {pca_js_string}
];


const grouped = {{}};


pcaPoints.forEach(point => {{

    if (!grouped[point.cluster]) {{
        grouped[point.cluster] = [];
    }}

    grouped[point.cluster].push({{
        x: point.x,
        y: point.y,
        name: point.name
    }});

}});


const pcaDatasets = Object.keys(grouped).map(
    cluster => {{

        return {{
            label: "Cluster " + cluster,

            data: grouped[cluster],

            pointRadius: 6,

            pointHoverRadius: 9
        }};

    }}
);


new Chart(
    document.getElementById("pcaChart"),
    {{

        type: "scatter",

        data: {{
            datasets: pcaDatasets
        }},

        options: {{

            responsive: true,

            maintainAspectRatio: false,

            scales: {{

                x: {{
                    title: {{
                        display: true,
                        text: "Principal Component 1"
                    }}
                }},

                y: {{
                    title: {{
                        display: true,
                        text: "Principal Component 2"
                    }}
                }}

            }},

            plugins: {{

                tooltip: {{

                    callbacks: {{

                        label: function(context) {{

                            const point =
                                context.raw;

                            if (point.name) {{

                                return (
                                    point.name +
                                    " — Cluster " +
                                    context.dataset.label
                                        .replace(
                                            "Cluster ",
                                            ""
                                        )
                                );

                            }}

                            return (
                                "Cluster " +
                                context.dataset.label
                            );

                        }}

                    }}

                }}

            }}

        }}

    }}
);

</script>


</body>

</html>
"""


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(html_content)


# ============================================================
# FINISHED
# ============================================================

print("=" * 70)
print("Dashboard generated successfully.")
print("=" * 70)
print()
print(f"Report saved to:")
print(f"  {OUTPUT_FILE}")
print()
print("Open the HTML file in your browser to view the dashboard.")
print()


# Automatically open the dashboard
try:
    webbrowser.open(
        OUTPUT_FILE.resolve().as_uri()
    )
except Exception:
    pass