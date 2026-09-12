import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ---------------------------------------------------------
# Load trained model + scaler (exported from the notebook)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return kmeans, scaler

try:
    kmeans, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure kmeans_model.pkl and scaler.pkl "
        "are in the same folder as app.py (see README for how to export them)."
    )
    st.stop()

# ---------------------------------------------------------
# Derive cluster profiles directly from the trained model
# (centroids are inverse-scaled so they're in real RFM units)
# ---------------------------------------------------------
centroids_scaled = kmeans.cluster_centers_
centroids_real = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(
    centroids_real, columns=["MonetaryValue", "Frequency", "Recency"]
)
centroid_df.index.name = "Cluster"

# Rank clusters by a simple value score so labels/actions map correctly
# regardless of which raw cluster index KMeans assigned.
centroid_df["Score"] = (
    centroid_df["MonetaryValue"].rank()
    + centroid_df["Frequency"].rank()
    - centroid_df["Recency"].rank()
)
ordered = centroid_df.sort_values("Score", ascending=False).index.tolist()

PROFILES = {
    ordered[0]: {
        "name": "VIP / Best Customers",
        "insight": "High spenders, frequent buyers, recently active.",
        "action": "Prioritize for loyalty perks and early access to protect retention.",
        "color": "#2ca02c",
    },
    ordered[1]: {
        "name": "Regular / Core Customers",
        "insight": "Moderate spend and frequency, purchased somewhat recently.",
        "action": "Target with upsell/cross-sell campaigns to push toward VIP status.",
        "color": "#1f77b4",
    },
    ordered[2]: {
        "name": "At-Risk / Lapsed Customers",
        "insight": "Low spend and frequency, long time since last purchase.",
        "action": "Launch win-back offers before they're lost for good.",
        "color": "#ff7f0e",
    },
}

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------
st.title("Customer Segmentation — RFM Analysis")

tab_report, tab_predict = st.tabs(["📊 Segment Report", "🔮 Predict Your Segment"])

# ---------------------------------------------------------
# TAB 1: Report
# ---------------------------------------------------------
with tab_report:
    st.subheader("Cluster Profiles")

    cols = st.columns(3)
    for i, cluster_id in enumerate(ordered):
        p = PROFILES[cluster_id]
        row = centroid_df.loc[cluster_id]
        with cols[i]:
            st.markdown(f"### {p['name']}")
            st.metric("Avg. Monetary Value", f"${row['MonetaryValue']:.0f}")
            st.metric("Avg. Frequency", f"{row['Frequency']:.1f}")
            st.metric("Avg. Recency (days)", f"{row['Recency']:.0f}")
            st.write(f"**Insight:** {p['insight']}")
            st.write(f"**Action:** {p['action']}")

    st.divider()
    st.subheader("Cluster Comparison")

    plot_df = centroid_df.copy()
    plot_df["Segment"] = [PROFILES[i]["name"] for i in plot_df.index]
    plot_df["Color"] = [PROFILES[i]["color"] for i in plot_df.index]

    fig = px.bar(
        plot_df,
        x="Segment",
        y=["MonetaryValue", "Frequency", "Recency"],
        barmode="group",
        title="Average RFM Values by Segment",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig3d = go.Figure()
    for cluster_id in ordered:
        p = PROFILES[cluster_id]
        row = centroid_df.loc[cluster_id]
        fig3d.add_trace(
            go.Scatter3d(
                x=[row["MonetaryValue"]],
                y=[row["Frequency"]],
                z=[row["Recency"]],
                mode="markers",
                marker=dict(size=14, color=p["color"]),
                name=p["name"],
            )
        )
    fig3d.update_layout(
        title="Segment Centroids in RFM Space",
        scene=dict(
            xaxis_title="Monetary Value",
            yaxis_title="Frequency",
            zaxis_title="Recency",
        ),
        height=600,
    )
    st.plotly_chart(fig3d, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Predictor
# ---------------------------------------------------------
with tab_predict:
    st.subheader("Which segment does a customer fall into?")
    st.write("Enter a customer's RFM values to see their predicted segment.")

    c1, c2, c3 = st.columns(3)
    with c1:
        monetary = st.number_input("Monetary Value ($)", min_value=0.0, value=500.0, step=10.0)
    with c2:
        frequency = st.number_input("Frequency (# purchases)", min_value=0, value=2, step=1)
    with c3:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)

    if st.button("Predict Segment", type="primary"):
        input_df = pd.DataFrame(
            [[monetary, frequency, recency]],
            columns=["MonetaryValue", "Frequency", "Recency"],
        )
        scaled_input = scaler.transform(input_df)
        predicted_cluster = kmeans.predict(scaled_input)[0]
        p = PROFILES[predicted_cluster]

        st.success(f"Predicted Segment: **{p['name']}**")
        st.write(f"**Insight:** {p['insight']}")
        st.write(f"**Recommended Action:** {p['action']}")
