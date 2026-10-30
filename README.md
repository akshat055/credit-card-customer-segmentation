# 💳 Credit Card Customer Segmentation

An unsupervised machine learning project that segments credit card customers based on their spending habits, payment behaviour, balances and credit utilization patterns.

## 🎯 Project Objective

The objective is to identify distinct customer groups using clustering algorithms and provide actionable business insights that help financial institutions develop targeted marketing strategies.

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- K-Means Clustering
- PCA (Principal Component Analysis)
- Streamlit
- Plotly
- Joblib

## 📊 Dataset

Credit Card Dataset for Clustering from Kaggle.

- Approximately 8,950 customer records.
- 18 original columns.
- Financial and behavioural attributes.

Dataset: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

## ⚙️ Project Workflow

1. Exploratory Data Analysis (EDA)
2. Missing Value Treatment
3. Data Preprocessing and Feature Engineering
4. Log Transformation and Standardization
5. K-Means Clustering
6. Optimal Cluster Selection using Elbow Method and Silhouette Score
7. Customer Profiling and Segment Analysis
8. PCA Visualization
9. Interactive Streamlit Dashboard

## 👥 Customer Segments

The K-Means model identifies four customer segments:

| Cluster | Segment | Description |
|---|---|---|
| 0 | Low Engagement Customers | Relatively low purchase activity |
| 1 | Cash Advance Users | High cash advance activity |
| 2 | High Value Customers | High purchase and payment activity |
| 3 | Frequent Installment Shoppers | High purchase frequency and installment activity |

## 📈 Dashboard Features

- Overall customer statistics
- Customer segment distribution
- Individual segment analysis
- Customer behaviour visualizations
- Business recommendations
- CSV upload for customer segment prediction
- Downloadable prediction results

## 📁 Project Structure

```text
credit-card-customer-segmentation/
│
├── data/
│