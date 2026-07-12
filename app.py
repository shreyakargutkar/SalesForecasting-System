import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
from sklearn.ensemble import IsolationForest

# Page configuration
st.set_page_config(
    page_title="Superstore Sales Operations Portal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (premium theme)
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background: #2c3e50;
        color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #3498db;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
        text-transform: uppercase;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Week Number'] = df['Order Date'].dt.isocalendar().week
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df = df.dropna(subset=['Sales', 'Order Date'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Reconstruct monthly sales index
try:
    monthly_series = df.set_index('Order Date').resample('ME')['Sales'].sum()
except ValueError:
    monthly_series = df.set_index('Order Date').resample('M')['Sales'].sum()

# Title and header
st.title("📊 Superstore Sales Analytics & Operations Portal")
st.markdown("---")

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Sales Overview Dashboard",
    "Forecast Explorer",
    "Anomaly Report",
    "Product Demand Segments"
])

# Helper function for metrics
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# Season mapper for XGBoost
def get_season_num(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

# ==============================================================================
# Page 1: Sales Overview Dashboard
# ==============================================================================
if page == "Sales Overview Dashboard":
    st.header("📈 Sales Overview Dashboard")
    
    # KPIs Top Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">${:,.2f}</div><div class="metric-label">Total Historical Sales</div></div>'.format(df['Sales'].sum()), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">{:,}</div><div class="metric-label">Total Order Transactions</div></div>'.format(df['Order ID'].nunique()), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">${:,.2f}</div><div class="metric-label">Average Order Value</div></div>'.format(df.groupby('Order ID')['Sales'].sum().mean()), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card" style="border-top-color:#2ecc71"><div class="metric-value">{:,}</div><div class="metric-label">Unique Products Sold</div></div>'.format(df['Product ID'].nunique()), unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filters
    st.subheader("Interactive Filters & Visuals")
    f_cols = st.columns(2)
    with f_cols[0]:
        selected_regions = st.multiselect("Filter by Region", options=df['Region'].unique().tolist(), default=df['Region'].unique().tolist())
    with f_cols[1]:
        selected_categories = st.multiselect("Filter by Category", options=df['Category'].unique().tolist(), default=df['Category'].unique().tolist())
        
    filtered_df = df[(df['Region'].isin(selected_regions)) & (df['Category'].isin(selected_categories))]
    
    # Layout with columns for charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### **Annual Sales Performance**")
        annual_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=annual_sales, x='Year', y='Sales', hue='Year', palette='Blues_r', ax=ax, legend=False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax.set_ylabel("Sales ($)", fontweight='bold')
        ax.set_xlabel("Year", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
    with c2:
        st.markdown("### **Monthly Sales Trend**")
        try:
            monthly_trend = filtered_df.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
        except ValueError:
            monthly_trend = filtered_df.set_index('Order Date').resample('M')['Sales'].sum().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(monthly_trend['Order Date'], monthly_trend['Sales'], color='#3498db', marker='o', linewidth=2.5)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax2.set_ylabel("Sales ($)", fontweight='bold')
        ax2.set_xlabel("Date", fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig2)

# ==============================================================================
# Page 2: Forecast Explorer
# ==============================================================================
elif page == "Forecast Explorer":
    st.header("🔮 Forecast Explorer (XGBoost)")
    
    st.markdown("Select a specific business segment (Category or Region) and forecast future monthly sales up to 3 months ahead using our high-performing XGBoost machine learning model.")
    
    # Selectors
    c1, c2, c3 = st.columns(3)
    with c1:
        segment_type = st.selectbox("Select Segment Type", ["Category", "Region"])
    with c2:
        if segment_type == "Category":
            segment_val = st.selectbox("Select Category Value", df['Category'].unique().tolist())
        else:
            segment_val = st.selectbox("Select Region Value", df['Region'].unique().tolist())
    with c3:
        horizon = st.slider("Select Forecast Horizon (Months)", min_value=1, max_value=3, value=3)
        
    # Segment data aggregation
    seg_df = df[df[segment_type] == segment_val].copy()
    try:
        seg_series = seg_df.set_index('Order Date').resample('ME')['Sales'].sum()
    except ValueError:
        seg_series = seg_df.set_index('Order Date').resample('M')['Sales'].sum()
    
    # Align dates with general dataset monthly range to prevent gaps
    seg_series = seg_series.reindex(monthly_series.index, fill_value=0.0)
    
    # 1. Feature Engineering
    xgb_df = seg_series.to_frame('Sales')
    xgb_df['Month'] = xgb_df.index.month
    xgb_df['Quarter'] = xgb_df.index.quarter
    xgb_df['Season'] = xgb_df['Month'].apply(get_season_num)
    xgb_df['lag1'] = xgb_df['Sales'].shift(1)
    xgb_df['lag2'] = xgb_df['Sales'].shift(2)
    xgb_df['lag3'] = xgb_df['Sales'].shift(3)
    xgb_df['rolling_mean_3'] = xgb_df[['lag1', 'lag2', 'lag3']].mean(axis=1)
    xgb_df_clean = xgb_df.dropna()
    
    features = ['lag1', 'lag2', 'lag3', 'rolling_mean_3', 'Month', 'Quarter', 'Season']
    
    # Train/Validation Metrics (on last 3 months of history)
    train_series = seg_series[:-3]
    test_series = seg_series[-3:]
    
    train_xgb = xgb_df_clean[xgb_df_clean.index < test_series.index[0]]
    X_train = train_xgb[features]
    y_train = train_xgb['Sales']
    
    # Fit validation model
    model_val = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model_val.fit(X_train, y_train)
    
    # Autoregressive Validation prediction
    xgb_val_pred = []
    last_lags_val = list(train_series[-3:])
    for idx, date in enumerate(test_series.index):
        l1 = last_lags_val[-1]
        l2 = last_lags_val[-2]
        l3 = last_lags_val[-3]
        rm3 = np.mean([l1, l2, l3])
        month = date.month
        quarter = date.quarter
        season = get_season_num(month)
        
        feat_row = pd.DataFrame([[l1, l2, l3, rm3, month, quarter, season]], columns=features)
        pred_val = model_val.predict(feat_row)[0]
        xgb_val_pred.append(pred_val)
        last_lags_val.append(pred_val)
        
    mae, rmse, _ = compute_metrics(test_series.values, xgb_val_pred)
    
    # 2. Fit Full Model and Forecast
    X_full = xgb_df_clean[features]
    y_full = xgb_df_clean['Sales']
    model_full = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model_full.fit(X_full, y_full)
    
    xgb_future = []
    last_lags_full = list(seg_series[-3:])
    future_dates = [
        seg_series.index[-1] + pd.offsets.MonthEnd(1),
        seg_series.index[-1] + pd.offsets.MonthEnd(2),
        seg_series.index[-1] + pd.offsets.MonthEnd(3)
    ]
    
    for date in future_dates[:horizon]:
        l1 = last_lags_full[-1]
        l2 = last_lags_full[-2]
        l3 = last_lags_full[-3]
        rm3 = np.mean([l1, l2, l3])
        month = date.month
        quarter = date.quarter
        season = get_season_num(month)
        
        feat_row = pd.DataFrame([[l1, l2, l3, rm3, month, quarter, season]], columns=features)
        pred_future = model_full.predict(feat_row)[0]
        xgb_future.append(pred_future)
        last_lags_full.append(pred_future)
        
    # Plot forecast
    fig_f, ax_f = plt.subplots(figsize=(12, 6))
    ax_f.plot(seg_series.index, seg_series.values, label='Historical Sales', color='#2c3e50', marker='o', linewidth=2)
    ax_f.plot(test_series.index, xgb_val_pred, label='Validation Fit (Oct-Dec 2018)', color='#e67e22', linestyle='--', marker='s')
    ax_f.plot(future_dates[:horizon], xgb_future, label='Future Forecast (XGBoost)', color='#9b59b6', marker='^', linewidth=2.5)
    
    ax_f.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax_f.set_title(f"XGBoost Sales Forecast: {segment_val} {segment_type}", fontsize=14, fontweight='bold', pad=15)
    ax_f.set_xlabel("Date", fontsize=11)
    ax_f.set_ylabel("Sales ($)", fontsize=11)
    ax_f.grid(True, linestyle=':', alpha=0.6)
    ax_f.legend()
    plt.tight_layout()
    st.pyplot(fig_f)
    
    # Display performance metrics and forecast table
    st.markdown("---")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader("Model Performance Indicators (Validation Set)")
        st.markdown(f"""
        - **Model Selected**: XGBoost Regressor (Autoregressive Lag Features)
        - **Mean Absolute Error (MAE)**: **${mae:,.2f}**
        - **Root Mean Squared Error (RMSE)**: **${rmse:,.2f}**
        """)
    with m_col2:
        st.subheader("Forecast Values Table")
        forecast_table = pd.DataFrame({
            'Forecast Date': [d.strftime('%Y-%m-%d') for d in future_dates[:horizon]],
            'Forecasted Sales': ["${:,.2f}".format(v) for v in xgb_future]
        })
        st.table(forecast_table)

# ==============================================================================
# Page 3: Anomaly Report
# ==============================================================================
elif page == "Anomaly Report":
    st.header("🚨 Weekly Sales Anomaly Report")
    
    st.markdown("We monitor sales spikes and drops at the weekly level using two methods:")
    st.markdown("- **Isolation Forest (Global)**: Catches macro-level capacity outliers.")
    st.markdown("- **Rolling Z-Score (Local)**: Flags deviations $> 2$ standard deviations from the 8-week local window mean.")
    
    # Display saved charts if they exist, otherwise regenerate
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Isolation Forest (Global Anomalies)")
        if os.path.exists("charts/anomalies_isolationforest.png"):
            st.image("charts/anomalies_isolationforest.png", use_container_width=True)
        else:
            st.warning("Isolation Forest anomaly plot not found. Run analysis.ipynb first.")
            
    with c2:
        st.subheader("2. Rolling Z-Score (Local Anomalies)")
        if os.path.exists("charts/anomalies_zscore.png"):
            st.image("charts/anomalies_zscore.png", use_container_width=True)
        else:
            st.warning("Z-Score anomaly plot not found. Run analysis.ipynb first.")
            
    # Calculate anomaly tables to display dynamically
    st.markdown("---")
    st.subheader("Detected Anomalous Weeks Table")
    
    weekly_dates = df.groupby(['Year', 'Week Number'])['Order Date'].min().reset_index()
    weekly_sales = df.groupby(['Year', 'Week Number'])['Sales'].sum().reset_index()
    weekly_sales = pd.merge(weekly_sales, weekly_dates, on=['Year', 'Week Number'])
    weekly_sales = weekly_sales.sort_values('Order Date').reset_index(drop=True)
    
    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly_sales['anomaly_iso'] = iso.fit_predict(weekly_sales[['Sales']])
    
    # Z-Score
    weekly_sales['rolling_mean'] = weekly_sales['Sales'].rolling(window=8, min_periods=1).mean()
    weekly_sales['rolling_std'] = weekly_sales['Sales'].rolling(window=8, min_periods=1).std().fillna(0)
    weekly_sales['z_score'] = np.where(
        weekly_sales['rolling_std'] > 0,
        (weekly_sales['Sales'] - weekly_sales['rolling_mean']) / weekly_sales['rolling_std'],
        0.0
    )
    weekly_sales['anomaly_zscore'] = np.where(np.abs(weekly_sales['z_score']) > 2, -1, 1)
    
    # Combine anomaly results for table
    weekly_sales['Detected By'] = np.select(
        [
            (weekly_sales['anomaly_iso'] == -1) & (weekly_sales['anomaly_zscore'] == -1),
            (weekly_sales['anomaly_iso'] == -1),
            (weekly_sales['anomaly_zscore'] == -1)
        ],
        [
            "Both Methods",
            "Isolation Forest (Global)",
            "Rolling Z-Score (Local)"
        ],
        default="Normal"
    )
    
    anomaly_weeks = weekly_sales[weekly_sales['Detected By'] != "Normal"].copy()
    anomaly_weeks['Order Date'] = anomaly_weeks['Order Date'].dt.strftime('%Y-%m-%d')
    anomaly_weeks['Sales'] = anomaly_weeks['Sales'].map(lambda x: "${:,.2f}".format(x))
    anomaly_weeks['z_score'] = anomaly_weeks['z_score'].map(lambda x: "{:.2f}".format(x))
    
    st.dataframe(
        anomaly_weeks[['Order Date', 'Year', 'Week Number', 'Sales', 'z_score', 'Detected By']]
        .rename(columns={'Order Date': 'Week Commencing', 'z_score': 'Z-Score Deviation'}),
        use_container_width=True
    )

# ==============================================================================
# Page 4: Product Demand Segments
# ==============================================================================
elif page == "Product Demand Segments":
    st.header("🎯 Product Demand Segmentation (K-Means)")
    
    st.markdown("We run K-Means Clustering on the Superstore product sub-categories to group them into demand segments based on volume, YoY growth, volatility, and Average Order Value.")
    
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Product Clusters PCA Visualization")
        if os.path.exists("charts/clusters.png"):
            st.image("charts/clusters.png", use_container_width=True)
        else:
            st.warning("Segmentation cluster plot not found. Run analysis.ipynb first.")
            
    with c2:
        st.subheader("Tailored Inventory Recommendations")
        st.markdown("""
        **Cluster 0: Low-Volume Stable Commodities**
        - *Replenishment*: Just-in-Time (JIT)
        - *Action*: Minimize buffer stocks to reduce capital lockup. Set automated reorders.
        
        **Cluster 1: Premium High-Volume Revenue Drivers**
        - *Replenishment*: Dynamic Safety Stock Buffers (2-3 weeks of sales)
        - *Action*: Prevent stockouts at all costs, track forecasts weekly.
        
        **Cluster 2: Binders (Hyper-Growth Outlier)**
        - *Replenishment*: Agile/Responsive Scaling
        - *Action*: Keep margins flexible. Align supply chain directly with promotions.
        """)
        
    st.markdown("---")
    st.subheader("Sub-Category Assignments Table")
    
    # Calculate features and assign clusters for display
    sales_2017 = df[df['Year'] == 2017].groupby('Sub-Category')['Sales'].sum()
    sales_2018 = df[df['Year'] == 2018].groupby('Sub-Category')['Sales'].sum()
    yoy_growth = ((sales_2018 - sales_2017) / sales_2017).fillna(0.0).reset_index()
    yoy_growth.columns = ['Sub-Category', 'YoY_Growth']
    
    monthly_sub_sales = df.groupby(['Sub-Category', 'Year', 'Month'])['Sales'].sum().reset_index()
    volatility = monthly_sub_sales.groupby('Sub-Category')['Sales'].std().fillna(0.0).reset_index()
    volatility.columns = ['Sub-Category', 'Volatility']
    
    order_sub_sales = df.groupby(['Sub-Category', 'Order ID'])['Sales'].sum().reset_index()
    aov = order_sub_sales.groupby('Sub-Category')['Sales'].mean().reset_index()
    aov.columns = ['Sub-Category', 'AOV']
    
    sub_sales_total = df.groupby('Sub-Category')['Sales'].sum().reset_index()
    features_df = sub_sales_total.merge(yoy_growth, on='Sub-Category')
    features_df = features_df.merge(volatility, on='Sub-Category')
    features_df = features_df.merge(aov, on='Sub-Category')
    features_df.columns = ['Sub-Category', 'Total_Sales_Volume', 'YoY_Growth', 'Volatility', 'AOV']
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    scaler = StandardScaler()
    feature_cols = ['Total_Sales_Volume', 'YoY_Growth', 'Volatility', 'AOV']
    scaled = scaler.fit_transform(features_df[feature_cols])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    features_df['Cluster'] = kmeans.fit_predict(scaled)
    
    cluster_names = {
        0: "Low-Volume Stable Commodities (Cluster 0)",
        1: "Premium High-Volume Revenue Drivers (Cluster 1)",
        2: "Hyper-Growth Segment (Cluster 2)"
    }
    features_df['Segment Name'] = features_df['Cluster'].map(cluster_names)
    
    # Display assignments
    features_df['Total_Sales_Volume'] = features_df['Total_Sales_Volume'].map(lambda x: "${:,.2f}".format(x))
    features_df['YoY_Growth'] = features_df['YoY_Growth'].map(lambda x: "{:+.1f}%".format(x * 100))
    features_df['Volatility'] = features_df['Volatility'].map(lambda x: "${:,.2f}".format(x))
    features_df['AOV'] = features_df['AOV'].map(lambda x: "${:,.2f}".format(x))
    
    st.dataframe(
        features_df[['Sub-Category', 'Segment Name', 'Total_Sales_Volume', 'YoY_Growth', 'Volatility', 'AOV']]
        .sort_values('Segment Name'),
        use_container_width=True
    )
