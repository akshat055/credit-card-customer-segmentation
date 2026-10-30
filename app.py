
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide"
)

# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ------------------------------------------------
# LOAD MODELS AND DATA
# ------------------------------------------------

@st.cache_resource
def load_models():
    kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return kmeans, scaler


@st.cache_data
def load_data():
    customers = pd.read_csv(DATA_DIR / "clustered_customers.csv")
    profiles = pd.read_csv(
        DATA_DIR / "cluster_profiles.csv",
        index_col=0
    )
    return customers, profiles


try:
    model, scaler = load_models()
    customers, profiles = load_data()

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

cluster_names = {
    0: "Low Engagement Customers",
    1: "Cash Advance Users",
    2: "High Value Customers",
    3: "Frequent Installment Shoppers"
}

recommendations = {
    0: "Offer cashback incentives and re-engagement campaigns.",
    1: "Promote responsible credit usage and alternative payment options.",
    2: "Provide premium rewards and exclusive loyalty benefits.",
    3: "Offer installment-based promotions and flexible EMI options."
}

skewed_features = [
    'BALANCE',
    'PURCHASES',
    'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE',
    'CASH_ADVANCE_TRX',
    'PURCHASES_TRX',
    'CREDIT_LIMIT',
    'PAYMENTS',
    'MINIMUM_PAYMENTS'
]

expected_features = list(scaler.feature_names_in_)


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("💳 Customer Segmentation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Customer Segments",
        "Predict Customer Segment"
    ]
)


# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------

if page == "Dashboard":

    st.title("Credit Card Customer Segmentation")

    st.markdown(
        "Analyze customer financial behaviour using "
        "unsupervised machine learning and K-Means clustering."
    )

    st.divider()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{len(customers):,}"
    )

    col2.metric(
        "Customer Segments",
        customers["Cluster"].nunique()
    )

    col3.metric(
        "Average Purchases",
        f"{customers['PURCHASES'].mean():,.2f}"
    )

    col4.metric(
        "Average Balance",
        f"{customers['BALANCE'].mean():,.2f}"
    )

    st.divider()

    # CUSTOMER DISTRIBUTION

    st.subheader("Customer Distribution")

    distribution = (
        customers["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    distribution.columns = ["Cluster", "Customers"]

    distribution["Segment"] = distribution["Cluster"].map(
        cluster_names
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            distribution,
            names="Segment",
            values="Customers",
            hole=0.45,
            title="Customer Segment Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            distribution,
            x="Segment",
            y="Customers",
            color="Segment",
            title="Customers per Segment"
        )

        st.plotly_chart(fig, use_container_width=True)

    # PURCHASE COMPARISON

    st.subheader("Average Purchases by Segment")

    purchase_data = (
        customers.groupby("Cluster")["PURCHASES"]
        .mean()
        .reset_index()
    )

    purchase_data["Segment"] = purchase_data["Cluster"].map(
        cluster_names
    )

    fig = px.bar(
        purchase_data,
        x="Segment",
        y="PURCHASES",
        color="Segment",
        title="Average Customer Purchases"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# CUSTOMER SEGMENTS
# ------------------------------------------------

elif page == "Customer Segments":

    st.title("Customer Segment Analysis")

    selected = st.selectbox(
        "Select a Customer Segment",
        list(cluster_names.keys()),
        format_func=lambda x: cluster_names[x]
    )

    segment = customers[
        customers["Cluster"] == selected
    ]

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers", f"{len(segment):,}")

    col2.metric(
        "Average Purchases",
        f"{segment['PURCHASES'].mean():,.2f}"
    )

    col3.metric(
        "Average Balance",
        f"{segment['BALANCE'].mean():,.2f}"
    )

    st.subheader("Customer Characteristics")

    st.dataframe(
        profiles.loc[selected].to_frame("Average Value"),
        use_container_width=True
    )

    st.subheader("Purchase Behaviour")

    fig = px.histogram(
        segment,
        x="PURCHASES",
        nbins=40,
        title="Purchase Distribution in Selected Segment"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business Recommendation")

    st.info(recommendations[selected])

    st.subheader("Customer Records")

    st.dataframe(
        segment.head(100),
        use_container_width=True
    )


# ------------------------------------------------
# PREDICT CUSTOMER SEGMENT
# ------------------------------------------------

elif page == "Predict Customer Segment":

    st.title("Predict Customer Segment")

    st.write(
        "Upload a CSV containing customer financial information "
        "to predict their customer segments."
    )

    st.info(
        "The uploaded CSV must contain the original 17 numerical "
        "features used during model training. CUST_ID is optional."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        new_data = pd.read_csv(uploaded_file)

        if "CUST_ID" in new_data.columns:
            new_data = new_data.drop(columns=["CUST_ID"])

        missing = set(expected_features) - set(new_data.columns)

        if missing:

            st.error(f"Missing columns: {missing}")

        else:

            new_data = new_data[expected_features].copy()

            # Convert features to numeric
            new_data = new_data.apply(
                pd.to_numeric,
                errors="coerce"
            )

            if new_data.isnull().any().any():

                st.error(
                    "Your dataset contains missing or non-numeric values."
                )

            elif (new_data < 0).any().any():

                st.error(
                    "Financial features cannot contain negative values."
                )

            else:

                # Apply same transformations as training
                transformed = new_data.copy()

                for col in skewed_features:
                    transformed[col] = np.log1p(
                        transformed[col]
                    )

                # Apply saved scaler
                scaled = scaler.transform(transformed)

                # Predict clusters
                predictions = model.predict(scaled)

                new_data["Predicted Cluster"] = predictions

                new_data["Customer Segment"] = [
                    cluster_names[int(x)]
                    for x in predictions
                ]

                st.success("Customer segmentation completed!")

                st.subheader("Prediction Results")

                st.dataframe(
                    new_data,
                    use_container_width=True
                )

                # Prediction distribution

                counts = (
                    new_data["Customer Segment"]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = ["Segment", "Customers"]

                fig = px.pie(
                    counts,
                    names="Segment",
                    values="Customers",
                    hole=0.4,
                    title="Predicted Customer Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Download results

                st.download_button(
                    "Download Predictions",
                    data=new_data.to_csv(index=False),
                    file_name="customer_predictions.csv",
                    mime="text/csv"
                )
