import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0f1117; color: #ffffff; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #162032 100%);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #4da6ff;
        margin: 0;
    }
    .kpi-label {
        font-size: 13px;
        color: #8892a4;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-delta {
        font-size: 12px;
        color: #48bb78;
        margin: 4px 0 0 0;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1e2a3a, transparent);
        border-left: 4px solid #4da6ff;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 15px 0;
        font-size: 16px;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #1a2744 0%, #162032 100%);
        border: 1px solid #2d4a6e;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        font-size: 14px;
        color: #cbd5e0;
        line-height: 1.6;
    }
    .insight-box b { color: #4da6ff; }

    /* Page title */
    .page-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #4da6ff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .page-subtitle {
        color: #8892a4;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8892a4;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #2d4a6e !important;
        color: #4da6ff !important;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']       = df['Order Date'].dt.year
    df['Month']      = df['Order Date'].dt.month
    df['Week']       = df['Order Date'].dt.isocalendar().week.astype(int)
    df['ShipLag']    = (df['Ship Date'] - df['Order Date']).dt.days
    return df

df = load_data()

# ── Plotly dark theme ─────────────────────────────────────────
PLOT_THEME = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e2e8f0'),
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 5px 0;'>
        <div style='font-size:40px'>📊</div>
        <div style='font-size:18px; font-weight:700; color:#4da6ff;'>Sales Intelligence</div>
        <div style='font-size:12px; color:#8892a4;'>Superstore USA · 2015–2018</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    total  = df['Sales'].sum()
    orders = df['Order ID'].nunique()
    avg    = df['Sales'].mean()

    st.markdown(f"""
    <div class='kpi-card'>
        <p class='kpi-value'>${total/1e6:.2f}M</p>
        <p class='kpi-label'>Total Revenue</p>
    </div>
    <div class='kpi-card' style='margin-top:8px'>
        <p class='kpi-value'>{orders:,}</p>
        <p class='kpi-label'>Unique Orders</p>
    </div>
    <div class='kpi-card' style='margin-top:8px'>
        <p class='kpi-value'>${avg:.0f}</p>
        <p class='kpi-label'>Avg Order Value</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.selectbox("Navigate", [
        "📊 Sales Overview",
        "🔮 Forecast Explorer",
        "⚠️ Anomaly Report",
        "🧩 Demand Segments",
        "🤖 Ask the Data (AI)"
    ])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#8892a4; line-height:2;'>
        📅 &nbsp;Data: 2015–2018<br>
        🤖 &nbsp;Model: Prophet<br>
        🗂️ &nbsp;Records: 9,800<br>
        📍 &nbsp;Market: USA Retail
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built by Ritika Sen · Internship Project")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Sales Overview
# ══════════════════════════════════════════════════════════════
if page == "📊 Sales Overview":
    st.markdown("<p class='page-title'>Sales Overview</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>A high-level view of revenue performance across years, categories, and regions.</p>", unsafe_allow_html=True)

    # Filters
    col_f1, col_f2 = st.columns(2)
    region_filter   = col_f1.multiselect("Filter by Region", df['Region'].unique(), default=list(df['Region'].unique()))
    category_filter = col_f2.multiselect("Filter by Category", df['Category'].unique(), default=list(df['Category'].unique()))
    filtered = df[df['Region'].isin(region_filter) & df['Category'].isin(category_filter)]

    st.markdown("---")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    yoy = ((filtered[filtered['Year']==2018]['Sales'].sum() -
            filtered[filtered['Year']==2017]['Sales'].sum()) /
            filtered[filtered['Year']==2017]['Sales'].sum() * 100)

    col1.markdown(f"<div class='kpi-card'><p class='kpi-value'>${filtered['Sales'].sum():,.0f}</p><p class='kpi-label'>Total Revenue</p><p class='kpi-delta'>↑ All Years</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi-card'><p class='kpi-value'>{filtered['Order ID'].nunique():,}</p><p class='kpi-label'>Orders</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi-card'><p class='kpi-value'>${filtered['Sales'].mean():,.0f}</p><p class='kpi-label'>Avg Order Value</p></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='kpi-card'><p class='kpi-value'>{yoy:.1f}%</p><p class='kpi-label'>YoY Growth 2018</p><p class='kpi-delta'>↑ vs 2017</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📈 Revenue Trends</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        monthly = filtered.groupby(filtered['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly['Order Date'] = monthly['Order Date'].dt.to_timestamp()
        fig = px.area(monthly, x='Order Date', y='Sales',
                      labels={'Sales':'Sales ($)', 'Order Date':'Month'},
                      color_discrete_sequence=['#4da6ff'])
        fig.update_layout(**PLOT_THEME, title='Monthly Sales Trend',
                          xaxis_title='', yaxis_title='Sales ($)')
        fig.update_traces(fillcolor='rgba(77,166,255,0.15)', line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        yearly = filtered.groupby('Year')['Sales'].sum().reset_index()
        fig2 = px.bar(yearly, x='Year', y='Sales',
                      color='Sales', color_continuous_scale='Blues',
                      labels={'Sales':'Sales ($)'})
        fig2.update_layout(**PLOT_THEME, title='Sales by Year',
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>🗂️ Category & Region Breakdown</div>", unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        cat = filtered.groupby('Category')['Sales'].sum().reset_index()
        fig3 = px.pie(cat, names='Category', values='Sales',
                      color_discrete_sequence=['#4da6ff','#a78bfa','#48bb78'],
                      hole=0.4)
        fig3.update_layout(**PLOT_THEME, title='Sales by Category')
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        reg = filtered.groupby(['Region','Year'])['Sales'].sum().reset_index()
        fig4 = px.bar(reg, x='Year', y='Sales', color='Region',
                      barmode='group',
                      color_discrete_sequence=['#4da6ff','#a78bfa','#48bb78','#f6ad55'])
        fig4.update_layout(**PLOT_THEME, title='Sales by Region per Year')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-header'>🗺️ Geographic Sales Distribution</div>", unsafe_allow_html=True)

    state_abbrev = {
        'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR',
        'California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE',
        'Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID',
        'Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS',
        'Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
        'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS',
        'Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV',
        'New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY',
        'North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
        'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
        'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT',
        'Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV',
        'Wisconsin':'WI','Wyoming':'WY','District of Columbia':'DC'
    }
    state_sales = filtered.groupby('State')['Sales'].sum().reset_index()
    state_sales['Code'] = state_sales['State'].map(state_abbrev)
    fig5 = px.choropleth(state_sales, locations='Code', locationmode='USA-states',
                         color='Sales', scope='usa', hover_name='State',
                         color_continuous_scale='Blues',
                         labels={'Sales':'Total Sales ($)'})
    fig5.update_layout(**PLOT_THEME, title='Total Sales by US State')
    st.plotly_chart(fig5, use_container_width=True)

    # Insights
    st.markdown("<div class='section-header'>💡 Key Insights</div>", unsafe_allow_html=True)
    top_cat    = filtered.groupby('Category')['Sales'].sum().idxmax()
    top_region = filtered.groupby('Region')['Sales'].sum().idxmax()
    top_state  = filtered.groupby('State')['Sales'].sum().idxmax()
    st.markdown(f"""
    <div class='insight-box'>🏆 <b>Top Category:</b> {top_cat} leads revenue across all years.</div>
    <div class='insight-box'>📍 <b>Strongest Region:</b> {top_region} shows the highest and most consistent growth.</div>
    <div class='insight-box'>🗺️ <b>Top State:</b> {top_state} generates the most sales geographically.</div>
    <div class='insight-box'>📈 <b>Growth:</b> Overall revenue grew <b>{yoy:.1f}%</b> from 2017 to 2018, indicating strong business momentum.</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Forecast Explorer
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Forecast Explorer":
    st.markdown("<p class='page-title'>Forecast Explorer</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Run Prophet forecasts on any category or region for up to 3 months ahead.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    segment_type = col1.selectbox("Segment Type", ["Category", "Region"])
    if segment_type == "Category":
        segment_val = col2.selectbox("Select", df['Category'].unique())
        seg_df = df[df['Category'] == segment_val]
    else:
        segment_val = col2.selectbox("Select", df['Region'].unique())
        seg_df = df[df['Region'] == segment_val]
    horizon = col3.slider("Forecast Horizon (months)", 1, 3, 3)

    if st.button("🚀 Run Forecast", use_container_width=True):
        with st.spinner("Running Prophet model..."):
            monthly = seg_df.groupby(
                seg_df['Order Date'].dt.to_period('M')
            )['Sales'].sum().reset_index()
            monthly['Order Date'] = monthly['Order Date'].dt.to_timestamp()
            monthly = monthly.rename(columns={'Order Date':'ds','Sales':'y'})

            train = monthly.iloc[:-3]
            test  = monthly.iloc[-3:]

            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        seasonality_mode='additive', interval_width=0.95)
            m.fit(train)
            future   = m.make_future_dataframe(periods=3+horizon, freq='MS')
            forecast = m.predict(future)

            test_pred = forecast[forecast['ds'].isin(test['ds'])]['yhat'].values
            mae  = np.mean(np.abs(test['y'].values - test_pred))
            rmse = np.sqrt(np.mean((test['y'].values - test_pred)**2))
            mape = np.mean(np.abs((test['y'].values - test_pred) / test['y'].values)) * 100

            # Metric cards
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='kpi-card'><p class='kpi-value'>${mae:,.0f}</p><p class='kpi-label'>MAE</p></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='kpi-card'><p class='kpi-value'>${rmse:,.0f}</p><p class='kpi-label'>RMSE</p></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='kpi-card'><p class='kpi-value'>{mape:.1f}%</p><p class='kpi-label'>MAPE</p></div>", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>📈 Forecast Chart</div>", unsafe_allow_html=True)

            future_fc = forecast.tail(horizon)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly['ds'], y=monthly['y'],
                                     mode='lines', name='Actual',
                                     line=dict(color='#4da6ff', width=2)))
            fig.add_trace(go.Scatter(x=future_fc['ds'], y=future_fc['yhat'],
                                     mode='lines+markers', name='Forecast',
                                     line=dict(color='#f6ad55', width=2, dash='dash'),
                                     marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=pd.concat([future_fc['ds'], future_fc['ds'][::-1]]),
                                     y=pd.concat([future_fc['yhat_upper'], future_fc['yhat_lower'][::-1]]),
                                     fill='toself', fillcolor='rgba(246,173,85,0.15)',
                                     line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
            fig.update_layout(**PLOT_THEME, title=f'{segment_val} — {horizon} Month Forecast',
                              xaxis_title='Date', yaxis_title='Sales ($)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='section-header'>📋 Forecast Values</div>", unsafe_allow_html=True)
            fc_table = future_fc[['ds','yhat','yhat_lower','yhat_upper']].copy()
            fc_table.columns = ['Date','Forecast ($)','Lower Bound ($)','Upper Bound ($)']
            fc_table = fc_table.round(2)
            st.dataframe(fc_table, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — Anomaly Report
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Anomaly Report":
    st.markdown("<p class='page-title'>Anomaly Report</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Detecting unusual sales weeks using two independent methods.</p>", unsafe_allow_html=True)

    weekly_sales = df.groupby(df['Order Date'].dt.to_period('W'))['Sales'].sum().reset_index()
    weekly_sales['Order Date'] = weekly_sales['Order Date'].dt.to_timestamp()

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly_sales['anomaly_if'] = iso.fit_predict(weekly_sales[['Sales']])

    roll_mean = weekly_sales['Sales'].rolling(8, min_periods=1).mean()
    roll_std  = weekly_sales['Sales'].rolling(8, min_periods=1).std().fillna(1)
    weekly_sales['z_score']   = (weekly_sales['Sales'] - roll_mean) / roll_std
    weekly_sales['anomaly_z'] = weekly_sales['z_score'].abs() > 2

    retail_events = {
        1:'Post New Year clearance', 2:"Valentine's Day", 3:'Spring sale',
        4:'Easter promotions', 5:"Mother's Day", 6:'Mid-year clearance',
        7:'Summer sales', 8:'Back-to-school', 9:'Fall collection launch',
        10:'Pre-holiday stocking', 11:'Black Friday', 12:'Holiday season'
    }

    # Summary cards
    if_count = (weekly_sales['anomaly_if'] == -1).sum()
    z_count  = weekly_sales['anomaly_z'].sum()
    both     = ((weekly_sales['anomaly_if'] == -1) & weekly_sales['anomaly_z']).sum()

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='kpi-card'><p class='kpi-value'>{if_count}</p><p class='kpi-label'>Isolation Forest Anomalies</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><p class='kpi-value'>{z_count}</p><p class='kpi-label'>Z-Score Anomalies</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><p class='kpi-value'>{both}</p><p class='kpi-label'>Agreed Anomalies</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🔍 Detection Results</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌲 Isolation Forest", "📊 Z-Score Method"])

    def anomaly_plotly(df_w, mask, color, title):
        normal = df_w[~mask]
        anom   = df_w[mask]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_w['Order Date'], y=df_w['Sales'],
                                 mode='lines', name='Weekly Sales',
                                 line=dict(color='#4da6ff', width=1.5)))
        fig.add_trace(go.Scatter(x=anom['Order Date'], y=anom['Sales'],
                                 mode='markers', name='Anomaly',
                                 marker=dict(color=color, size=10, symbol='circle',
                                             line=dict(color='white', width=1))))
        fig.update_layout(**PLOT_THEME, title=title,
                          xaxis_title='Week', yaxis_title='Sales ($)')
        return fig

    with tab1:
        mask_if = weekly_sales['anomaly_if'] == -1
        st.plotly_chart(anomaly_plotly(weekly_sales, mask_if, '#ff6b6b', 'Isolation Forest Anomalies'), use_container_width=True)
        anom_if = weekly_sales[mask_if][['Order Date','Sales']].copy()
        anom_if['Likely Cause'] = anom_if['Order Date'].dt.month.map(retail_events)
        anom_if['Sales'] = anom_if['Sales'].round(2)
        st.dataframe(anom_if.reset_index(drop=True), use_container_width=True)

    with tab2:
        mask_z = weekly_sales['anomaly_z']
        st.plotly_chart(anomaly_plotly(weekly_sales, mask_z, '#f6ad55', 'Z-Score Anomalies'), use_container_width=True)
        anom_z = weekly_sales[mask_z][['Order Date','Sales','z_score']].copy()
        anom_z['Likely Cause'] = anom_z['Order Date'].dt.month.map(retail_events)
        anom_z['Sales']   = anom_z['Sales'].round(2)
        anom_z['z_score'] = anom_z['z_score'].round(3)
        st.dataframe(anom_z.reset_index(drop=True), use_container_width=True)

    st.markdown("<div class='section-header'>💡 Interpretation</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>🌲 <b>Isolation Forest</b> is sensitive to both unusually HIGH and LOW sales weeks, catching outliers in both directions.</div>
    <div class='insight-box'>📊 <b>Z-Score</b> flags weeks that deviate more than 2 standard deviations from the rolling mean — primarily catching upward spikes.</div>
    <div class='insight-box'>🤝 <b>Agreement:</b> Both methods agree on the March 2015 spike ($37,703) — the strongest signal in the data, likely a large bulk corporate order during spring sale season.</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — Demand Segments
# ══════════════════════════════════════════════════════════════
elif page == "🧩 Demand Segments":
    st.markdown("<p class='page-title'>Demand Segments</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>K-Means clustering of product sub-categories by demand behavior.</p>", unsafe_allow_html=True)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    sub_monthly = df.groupby(['Sub-Category', df['Order Date'].dt.to_period('M')])['Sales'].sum().reset_index()
    sub_monthly.columns = ['Sub-Category','Period','Sales']
    sub_yearly  = df.groupby(['Sub-Category','Year'])['Sales'].sum().unstack(fill_value=0)
    yoy_growth  = ((sub_yearly.iloc[:,-1] - sub_yearly.iloc[:,0]) / sub_yearly.iloc[:,0] * 100).round(2)
    volatility  = sub_monthly.groupby('Sub-Category')['Sales'].std().round(2)

    features = pd.DataFrame({
        'total_sales':   df.groupby('Sub-Category')['Sales'].sum(),
        'yoy_growth':    yoy_growth,
        'volatility':    volatility,
        'avg_order_val': df.groupby('Sub-Category')['Sales'].mean(),
    }).dropna()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    km       = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['Cluster'] = km.fit_predict(X_scaled)

    cluster_stats  = features.groupby('Cluster')[['total_sales','yoy_growth','volatility']].mean()
    cluster_labels = {
        cluster_stats['total_sales'].idxmax(): 'High Volume, Stable Demand',
        cluster_stats['yoy_growth'].idxmax():  'Growing Demand',
        cluster_stats['volatility'].idxmax():  'High Volatility',
    }
    remaining = [c for c in range(4) if c not in cluster_labels]
    for c in remaining:
        cluster_labels[c] = 'Low Volume, Declining Demand'
    features['Cluster Label'] = features['Cluster'].map(cluster_labels)

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    features['PC1'] = coords[:,0]
    features['PC2'] = coords[:,1]

    # Segment summary cards
    for label in cluster_labels.values():
        count = (features['Cluster Label'] == label).sum()

    cols = st.columns(4)
    colors_map = {
        'High Volume, Stable Demand':   ('#48bb78', '✅'),
        'Growing Demand':               ('#4da6ff', '📈'),
        'High Volatility':              ('#f6ad55', '⚡'),
        'Low Volume, Declining Demand': ('#fc8181', '⚠️'),
    }
    for col, (label, (color, icon)) in zip(cols, colors_map.items()):
        count    = (features['Cluster Label'] == label).sum()
        products = features[features['Cluster Label'] == label].index.tolist()
        col.markdown(f"""
        <div class='kpi-card'>
            <p style='font-size:24px; margin:0'>{icon}</p>
            <p class='kpi-value' style='color:{color}; font-size:20px'>{count} Products</p>
            <p class='kpi-label'>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🔬 PCA Cluster Visualization</div>", unsafe_allow_html=True)

    color_seq = ['#48bb78','#4da6ff','#f6ad55','#fc8181']
    fig = px.scatter(features.reset_index(), x='PC1', y='PC2',
                     color='Cluster Label', text='Sub-Category',
                     color_discrete_sequence=color_seq,
                     size_max=15)
    fig.update_traces(textposition='top center', marker=dict(size=14))
    fig.update_layout(**PLOT_THEME, title='Product Demand Segments (PCA)',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>📦 Stocking Recommendations</div>", unsafe_allow_html=True)

    strategies = {
        'High Volume, Stable Demand':   ('✅', '#48bb78', 'Maintain consistent safety stock. Automate reordering at fixed thresholds.'),
        'Growing Demand':               ('📈', '#4da6ff', 'Increase stock 20–30% ahead of peak months. Monitor closely.'),
        'High Volatility':              ('⚡', '#f6ad55', 'Keep flexible buffer stock. Avoid long-term bulk orders. Use just-in-time.'),
        'Low Volume, Declining Demand': ('⚠️', '#fc8181', 'Reduce stock. Run clearance promotions. Consider discontinuing slow SKUs.'),
    }
    for label, (icon, color, strategy) in strategies.items():
        products = ', '.join(features[features['Cluster Label'] == label].index.tolist())
        st.markdown(f"""
        <div class='insight-box'>
            {icon} <b style='color:{color}'>{label}</b><br>
            <span style='color:#8892a4'>Strategy:</span> {strategy}<br>
            <span style='color:#8892a4'>Products:</span> {products}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — AI: Ask the Data
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Ask the Data (AI)":
    st.markdown("<p class='page-title'>Ask the Data</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Ask any business question in plain English. Powered by Claude AI.</p>", unsafe_allow_html=True)

    total_sales  = df['Sales'].sum()
    top_category = df.groupby('Category')['Sales'].sum().idxmax()
    top_region   = df.groupby('Region')['Sales'].sum().idxmax()
    top_state    = df.groupby('State')['Sales'].sum().idxmax()
    avg_ship_lag = df['ShipLag'].mean()
    yearly_sales = df.groupby('Year')['Sales'].sum().to_dict()

    data_context = f"""
You are a retail business analyst assistant. Answer questions based only on this dataset summary:

DATASET: Superstore Sales Data (2015–2018), USA retail company
- Total Revenue: ${total_sales:,.0f}
- Top Category by Revenue: {top_category}
- Top Region by Revenue: {top_region}
- Top State by Revenue: {top_state}
- Average Ship Lag: {avg_ship_lag:.1f} days
- Yearly Sales: {yearly_sales}
- Categories: Furniture, Technology, Office Supplies
- Regions: West, East, Central, South
- Prophet Model Forecast (next 3 months Jan-Mar 2019):
  * Overall: $61,446 / $100,917 / $90,328
  * Technology: $13,816 / $11,357 / $42,216 (strongest growth)
  * West Region: $14,626 / $11,908 / $26,433 (top region)
- Anomalies: March 2015 spike ($37,703), Nov 2018 Black Friday surge
- Demand Segments:
  * Growing: Copiers
  * High Volatility: Machines
  * High Volume Stable: Accessories, Binders, Chairs, Phones, Storage, Tables
  * Declining: Appliances, Art, Bookcases, Envelopes, Fasteners, Furnishings, Labels, Paper, Supplies

Answer in 3-5 sentences, be specific with numbers, give a concrete business recommendation.
    """

    st.markdown("""
    <div class='insight-box'>
        💡 <b>Try asking:</b> "Which product should I stock more next month?" · 
        "What's causing the sales spike in November?" · 
        "Which region should we focus marketing on?" · 
        "Should we discontinue any products?"
    </div>
    """, unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    user_input = st.chat_input("Ask a business question about your sales data...")

    if user_input:
        st.session_state.chat_history.append({'role':'user','content':user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analysing data..."):
                import requests

                messages = [{'role':'user',
                             'content': data_context + f"\n\nUser question: {user_input}"}]

                if len(st.session_state.chat_history) > 1:
                    messages = [{'role':'user',
                                 'content': data_context + "\n\nUser question: " +
                                 st.session_state.chat_history[0]['content']}]
                    for msg in st.session_state.chat_history[1:]:
                        messages.append({'role': msg['role'], 'content': msg['content']})

                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json",
                        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                        "anthropic-version": "2023-06-01"
                        },
                    json={
                        "model": "claude-sonnet-4-6",
                        "max_tokens": 1000,
                        "system": "You are a helpful retail business analyst.",
                        "messages": messages
                    }
                )
                resp_json = response.json()
                if 'content' in resp_json:
                    answer = resp_json['content'][0]['text']
                else:
                     answer = f"API Error: {resp_json}"
                st.write(answer)
                st.session_state.chat_history.append({'role':'assistant','content':answer})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
