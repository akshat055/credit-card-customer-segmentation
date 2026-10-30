# Credit Card Customer Segmentation

An end-to-end Machine Learning project that uses **K-Means Clustering** to segment credit card customers based on their spending behaviour, purchase patterns and financial activity.

The project includes Exploratory Data Analysis (EDA), data preprocessing, clustering, customer profiling, model validation and an interactive Streamlit dashboard.

## Live Demo

**Streamlit Application:** [View Live App](https://credit-card-customer-segmentation-geaofnmbppdgqxhyfcmvhn.streamlit.app/)

## Project Overview

Credit card companies serve customers with different spending habits and financial behaviours. Understanding these differences helps businesses develop targeted marketing strategies, improve customer retention and provide better financial services.

This project applies unsupervised machine learning to group customers into meaningful segments based on their financial activity.

### Objectives

- Explore and understand customer financial behaviour.
- Identify meaningful customer groups using clustering.
- Determine the optimal number of clusters.
- Visualize and interpret customer segments.
- Build an interactive dashboard for customer analysis.
- Predict customer segments for newly uploaded customer data.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Clustering | K-Means |
| Dimensionality Reduction | PCA |
| Deployment | Streamlit |
| Model Persistence | Joblib |
| Development | Jupyter Notebook, VS Code |

## Dataset

The project uses a credit card customer dataset containing **8,950 customer records and 17 numerical features**.

The features describe:

- Customer balance and balance frequency
- Purchase amounts and purchase frequency
- One-off and installment purchases
- Cash advance activity
- Number of purchase and cash advance transactions
- Credit limit and payments
- Minimum payments and full payment percentage
- Customer tenure

`CUST_ID` is used for customer identification and excluded from model training.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Examined dataset structure, dimensions and data types.
- Identified missing values and duplicate records.
- Analyzed feature distributions and outliers.
- Studied correlations between numerical variables.
- Visualized customer spending and payment patterns.

### 2. Data Preprocessing

- Removed the customer ID from model features.
- Handled missing numerical values.
- Applied transformations to highly skewed features where appropriate.
- Standardized numerical features using StandardScaler.
- Prepared the final feature matrix for clustering.

### 3. K-Means Clustering

K-Means was selected to group customers based on similarities in their financial behaviour.

The optimal number of clusters was investigated using:

- **Elbow Method:** Examined the relationship between the number of clusters and within-cluster sum of squares.
- **Silhouette Score:** Evaluated cluster separation and cohesion.

The final model uses **K = 4 clusters**, balancing clustering quality and business interpretability.

Silhouette scores obtained during model selection:

| Number of Clusters | Silhouette Score |
|---|---:|
| 2 | 0.2515 |
| 3 | 0.2244 |
| 4 | 0.2166 |
| 5 | 0.2183 |
| 6 | 0.2172 |
| 7 | 0.2204 |
| 8 | 0.2100 |
| 9 | 0.1805 |
| 10 | 0.1817 |

Although K=2 achieved the highest silhouette score, K=4 was selected for more granular and useful customer segmentation.

### 4. PCA Visualization

Principal Component Analysis (PCA) was used to reduce the feature dimensions to two principal components for visualizing the customer clusters.

### 5. Customer Segmentation and Business Insights

The four clusters were interpreted based on their average financial characteristics.

| Cluster | Customer Segment | Characteristics |
|---|---|---|
| 0 | Low Engagement Customers | Relatively low purchase activity and balance |
| 1 | Cash Advance Users | Higher cash advance usage and relatively low purchases |
| 2 | High Value Customers | Higher purchase amounts, balances and payments |
| 3 | Frequent Installment Shoppers | High purchase frequency and installment-oriented behaviour |

These labels represent interpretations of the observed cluster profiles rather than predefined customer categories.

### 6. Model Validation

The deployed prediction pipeline was compared against the original K-Means cluster assignments using the same 8,950 customer records.

| Metric | Result |
|---|---:|
| Customers Tested | 8,950 |
| Exact Cluster Agreement | 100% |
| Adjusted Rand Index (ARI) | 1.0000 |

The results confirm that the deployed prediction pipeline reproduces the original clustering assignments exactly.

**Note:** This is a consistency check using the original dataset, not an independent evaluation on unseen customers.

## Streamlit Application

The interactive application provides three main sections:

### 1. Dashboard

- Total number of customers
- Number of customer segments
- Average customer purchases
- Average customer balance
- Customer segment distribution
- Average purchases by segment

### 2. Customer Segment Analysis

- Cluster-wise customer profiles
- Segment distribution
- Purchase behaviour visualizations
- Sample customer records
- Business recommendations for each segment

### 3. Predict Customer Segment

- Upload customer financial data through a CSV file.
- Process customer records using the saved preprocessing pipeline.
- Predict customer cluster assignments using the trained K-Means model.
- Display predicted customer segments.
- Visualize predicted customer distribution.
- Download the prediction results.

The uploaded CSV must contain the original 17 numerical features used during model training. `CUST_ID` is optional.
prediction_test.csv is a prepared sample used to demonstrate the Streamlit prediction feature.

## Project Structure

```text
credit-card-customer-segmentation/
│
├── data/
│   ├── credit_card.csv
│   ├── clustered_customers.csv
│   └── cluster_profiles.csv
|   └── prediction_test.csv
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── pca_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Validation.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Habeebulk/credit-card-customer-segmentation.git
cd credit-card-customer-segmentation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Future Improvements

- Compare K-Means with Hierarchical Clustering and DBSCAN.
- Evaluate clustering stability on different customer samples.
- Explore additional customer behavioural features.
- Improve cluster interpretation using additional business metrics.
- Test predictions on genuinely unseen customer records.

## Conclusion

This project demonstrates an end-to-end application of unsupervised machine learning for customer segmentation.

By combining data analysis, preprocessing, K-Means clustering, PCA, customer profiling and Streamlit deployment, it provides an interactive way to understand customer behaviour and derive actionable business insights.

---

**Built using Python, Scikit-learn and Streamlit.**
