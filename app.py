import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence",
                    layout="wide", page_icon="📈")

# Data loading
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="utf-8-sig")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    return df

df = load_data()

@st.cache_data
def monthly_series(data, category=None, region=None):
    d = data.copy()
    if category:
        d = d[d["Category"] == category]
    if region:
        d = d[d["Region"] == region]
    s = d.set_index("Order Date").resample("MS")["Sales"].sum()
    s.index.freq = "MS"
    return s

@st.cache_data
def weekly_series(data):
    s = data.set_index("Order Date").resample("W")["Sales"].sum()
    return s

@st.cache_data
def run_sarima_forecast(series, steps=3):
    train = series.iloc[:-steps] if len(series) > steps + 6 else series
    test = series.iloc[-steps:] if len(series) > steps + 6 else None
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=steps)
    pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    mae = rmse = None
    if test is not None and len(test) == steps:
        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
    return pred, ci, mae, rmse

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sales Overview", "Forecast Explorer",
                                   "Anomaly Report", "Product Demand Segments"])

st.sidebar.markdown("---")
st.sidebar.caption("Superstore Sales Dataset · 2015–2018 · Intelligent Sales Forecasting System")

# PAGE 1 — Sales Overview Dashboard
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col3.metric("Avg Order Value", f"${df['Sales'].mean():,.2f}")
    col4.metric("Date Range", f"{df['Order Date'].min().year}–{df['Order Date'].max().year}")

    st.subheader("Total Sales by Year")
    yearly = df.groupby("Order Year")["Sales"].sum().reset_index()
    fig = px.bar(yearly, x="Order Year", y="Sales", text_auto=".2s",
                 color="Sales", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    monthly = monthly_series(df)
    fig2 = px.line(x=monthly.index, y=monthly.values, markers=True,
                    labels={"x": "Month", "y": "Sales ($)"})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Region & Category (filterable)")
    c1, c2 = st.columns(2)
    with c1:
        region_filter = st.multiselect("Filter by Region", options=df["Region"].unique(),
                                        default=list(df["Region"].unique()))
    with c2:
        cat_filter = st.multiselect("Filter by Category", options=df["Category"].unique(),
                                     default=list(df["Category"].unique()))
    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(cat_filter)]
    grp = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig3 = px.bar(grp, x="Region", y="Sales", color="Category", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

# PAGE 2 — Forecast Explorer
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Best-performing model: SARIMA (lowest MAE/RMSE/MAPE in Task 3 comparison)")

    c1, c2 = st.columns(2)
    with c1:
        dim = st.selectbox("Select dimension", ["Category", "Region", "Overall"])
    with c2:
        if dim == "Category":
            value = st.selectbox("Select value", df["Category"].unique())
        elif dim == "Region":
            value = st.selectbox("Select value", df["Region"].unique())
        else:
            value = None

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    if dim == "Category":
        series = monthly_series(df, category=value)
    elif dim == "Region":
        series = monthly_series(df, region=value)
    else:
        series = monthly_series(df)

    with st.spinner("Fitting SARIMA model..."):
        pred, ci, mae, rmse = run_sarima_forecast(series, steps=3)
        pred = pred.iloc[:horizon]
        ci = ci.iloc[:horizon]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index[-12:], y=series.values[-12:],
                              mode="lines+markers", name="Actual", line=dict(color="#2c6e9c")))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines+markers",
                              name="Forecast", line=dict(color="#c0392b", dash="dash")))
    fig.add_trace(go.Scatter(x=list(pred.index) + list(pred.index[::-1]),
                              y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                              fill="toself", fillcolor="rgba(192,57,43,0.15)",
                              line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
    fig.update_layout(title=f"{horizon}-Month Forecast — {value or 'Overall'}",
                       xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(pd.DataFrame({"Date": pred.index.date, "Forecasted Sales": pred.values.round(2)}))

    if mae is not None:
        m1, m2 = st.columns(2)
        m1.metric("MAE (holdout)", f"${mae:,.2f}")
        m2.metric("RMSE (holdout)", f"${rmse:,.2f}")
    else:
        st.info("Not enough history to compute a holdout MAE/RMSE for this series.")

# PAGE 3 — Anomaly Report
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly = weekly_series(df)
    wdf = pd.DataFrame({"Sales": weekly})
    iso = IsolationForest(contamination=0.07, random_state=42)
    wdf["iso_anomaly"] = iso.fit_predict(wdf[["Sales"]])

    window = 8
    wdf["rolling_mean"] = wdf["Sales"].rolling(window, min_periods=4, center=True).mean()
    wdf["rolling_std"] = wdf["Sales"].rolling(window, min_periods=4, center=True).std()
    wdf["z_score"] = (wdf["Sales"] - wdf["rolling_mean"]) / wdf["rolling_std"]
    wdf["z_anomaly"] = wdf["z_score"].abs() > 2

    iso_anom = wdf[wdf["iso_anomaly"] == -1]
    z_anom = wdf[wdf["z_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wdf.index, y=wdf["Sales"], mode="lines", name="Weekly Sales",
                              line=dict(color="#2c6e9c")))
    fig.add_trace(go.Scatter(x=iso_anom.index, y=iso_anom["Sales"], mode="markers",
                              name="Isolation Forest Anomaly",
                              marker=dict(color="#c0392b", size=10, symbol="x")))
    fig.add_trace(go.Scatter(x=z_anom.index, y=z_anom["Sales"], mode="markers",
                              name="Z-Score Anomaly",
                              marker=dict(color="#e67e22", size=12, symbol="circle-open", line=dict(width=2))))
    fig.update_layout(title="Weekly Sales — Detected Anomalies", xaxis_title="Week", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Isolation Forest anomalies", len(iso_anom))
    c2.metric("Z-Score anomalies", len(z_anom))

    st.subheader("Detected Anomaly Weeks (Isolation Forest)")
    st.dataframe(iso_anom[["Sales"]].sort_values("Sales", ascending=False)
                 .rename_axis("Week").reset_index())

    st.subheader("Detected Anomaly Weeks (Z-Score > 2σ)")
    st.dataframe(z_anom[["Sales", "z_score"]].rename_axis("Week").reset_index())

# PAGE 4 — Product Demand Segment
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")

    feats = []
    for sc, g in df.groupby("Sub-Category"):
        total_sales = g["Sales"].sum()
        avg_order_value = g["Sales"].mean()
        yearly = g.groupby("Order Year")["Sales"].sum().sort_index()
        growth = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100 if len(yearly) >= 2 else 0
        vol = g.set_index("Order Date").resample("MS")["Sales"].sum().std()
        feats.append({"Sub-Category": sc, "Total Sales": total_sales, "Growth Rate %": growth,
                       "Volatility": vol, "Avg Order Value": avg_order_value})
    feat_df = pd.DataFrame(feats).set_index("Sub-Category")

    X_scaled = StandardScaler().fit_transform(feat_df.values)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat_df["Cluster"] = km.fit_predict(X_scaled)

    stats = feat_df.groupby("Cluster")[["Total Sales", "Growth Rate %", "Volatility"]].mean()

    def label_cluster(row):
        if row["Total Sales"] > stats["Total Sales"].median() and row["Volatility"] < stats["Volatility"].median():
            return "High Volume, Stable Demand"
        elif row["Growth Rate %"] > 20:
            return "Growing Demand"
        elif row["Growth Rate %"] < -10:
            return "Declining Demand"
        return "Low Volume, High Volatility"

    labels = {c: label_cluster(stats.loc[c]) for c in stats.index}
    feat_df["Cluster Label"] = feat_df["Cluster"].map(labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    feat_df["PCA1"], feat_df["PCA2"] = coords[:, 0], coords[:, 1]

    fig = px.scatter(feat_df.reset_index(), x="PCA1", y="PCA2", color="Cluster Label",
                      text="Sub-Category", size="Total Sales", hover_data=["Growth Rate %", "Volatility"])
    fig.update_traces(textposition="top center")
    fig.update_layout(title="Product Demand Clusters (PCA projection)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Categories by Demand Cluster")
    st.dataframe(feat_df[["Total Sales", "Growth Rate %", "Volatility", "Cluster Label"]]
                 .sort_values("Cluster Label").round(2))

    st.subheader("Recommended Stocking Strategy")
    strategy = {
        "High Volume, Stable Demand": "Maintain steady safety stock; use simple reorder-point replenishment.",
        "Growing Demand": "Increase stock ahead of forecasted growth; monitor lead times closely.",
        "Declining Demand": "Reduce future orders; consider clearance or bundling to avoid overstock.",
        "Low Volume, High Volatility": "Keep lean stock; use faster, smaller, more frequent reorders.",
    }
    for label, advice in strategy.items():
        if label in feat_df["Cluster Label"].values:
            st.markdown(f"**{label}:** {advice}")