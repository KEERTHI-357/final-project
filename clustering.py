import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Warehouse Space Optimization", layout="wide")

st.title("📦 Warehouse Space Optimization using Smart Clustering")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("warehouse_cleaned.csv")

df = load_data()
st.sidebar.header("Filters")
cluster = st.sidebar.multiselect("Select Cluster(s):", sorted(df['Cluster'].unique()), default=df['Cluster'].unique())
filtered_df = df[df['Cluster'].isin(cluster)]

st.subheader("Cleaned Dataset Preview")
st.dataframe(filtered_df.head())

st.subheader("Cluster Count")
st.bar_chart(filtered_df['Cluster'].value_counts())

st.subheader("Cluster Visualization (PCA)")
fig, ax = plt.subplots()
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="tab10", data=filtered_df, ax=ax)
st.pyplot(fig)

# Count of product categories per cluster
if 'Product Category' in filtered_df.columns:
    st.subheader("Product Category Counts per Cluster")
    st.dataframe(filtered_df.groupby(['Cluster', 'Product Category']).size().unstack(fill_value=0))

st.subheader("Cluster Profiles (Mean Values)")
numeric_cols = filtered_df.select_dtypes(include="number").columns
cluster_profiles = filtered_df.groupby("Cluster")[numeric_cols].mean()
st.dataframe(cluster_profiles)
