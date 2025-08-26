"""
Interactive Exchange Rate Analysis Dashboard
============================================

A comprehensive Streamlit dashboard for exploring exchange rate data,
visualizing trends, and analyzing currency performance.

Author: Rohan Bhoge
Created: 2025
Project: Exchange Rate Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Exchange Rate Analysis Dashboard",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the cleaned exchange rate data"""
    try:
        df = pd.read_csv('data/cleaned_exchange_rates.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please run the data cleaning notebook first.")
        return None

@st.cache_data
def get_currency_stats(df):
    """Calculate currency statistics for the dashboard"""
    stats = df.groupby('country').agg({
        'exchange_rate': ['count', 'mean', 'std', 'min', 'max'],
        'quarterly_return': ['mean', 'std'],
        'volatility_4q': 'mean'
    }).round(4)
    
    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    stats = stats.reset_index()
    
    # Calculate additional metrics
    stats['coefficient_variation'] = stats['exchange_rate_std'] / stats['exchange_rate_mean']
    stats['data_completeness'] = stats['exchange_rate_count'] / df['quarter'].nunique()
    
    return stats

def create_time_series_plot(df, selected_countries, metric='exchange_rate'):
    """Create interactive time series plot"""
    filtered_df = df[df['country'].isin(selected_countries)]
    
    fig = px.line(
        filtered_df, 
        x='date', 
        y=metric, 
        color='country',
        title=f'{metric.replace("_", " ").title()} Over Time',
        labels={'date': 'Date', metric: metric.replace('_', ' ').title()}
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_volatility_comparison(df, selected_countries):
    """Create volatility comparison chart"""
    country_vol = df[df['country'].isin(selected_countries)].groupby('country')['quarterly_return'].std().sort_values()
    
    fig = px.bar(
        x=country_vol.values,
        y=country_vol.index,
        orientation='h',
        title='Return Volatility Comparison',
        labels={'x': 'Volatility (%)', 'y': 'Country'}
    )
    
    fig.update_layout(height=400)
    return fig

def create_correlation_heatmap(df, selected_countries):
    """Create correlation heatmap"""
    pivot_data = df[df['country'].isin(selected_countries)].pivot_table(
        index='date', columns='country', values='quarterly_return'
    )
    
    correlation_matrix = pivot_data.corr()
    
    fig = px.imshow(
        correlation_matrix,
        title='Currency Return Correlations',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    fig.update_layout(height=500)
    return fig

def create_risk_return_scatter(df, selected_countries):
    """Create risk-return scatter plot"""
    stats = df[df['country'].isin(selected_countries)].groupby('country').agg({
        'quarterly_return': ['mean', 'std']
    }).round(4)
    
    stats.columns = ['mean_return', 'volatility']
    stats = stats.reset_index()
    
    fig = px.scatter(
        stats,
        x='volatility',
        y='mean_return',
        text='country',
        title='Risk-Return Profile',
        labels={'volatility': 'Volatility (%)', 'mean_return': 'Mean Return (%)'}
    )
    
    fig.update_traces(textposition="top center")
    fig.update_layout(height=500)
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">💱 Exchange Rate Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">📊 Dashboard Controls</div>', unsafe_allow_html=True)
    
    # Country selection
    available_countries = sorted(df['country'].unique())
    default_countries = ['United States', 'United Kingdom', 'Japan', 'Germany', 'Canada']
    default_countries = [c for c in default_countries if c in available_countries]
    
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        available_countries,
        default=default_countries[:5],
        help="Choose currencies to analyze"
    )
    
    if not selected_countries:
        st.warning("⚠️ Please select at least one country to display data.")
        return
    
    # Date range selection
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter data by date range"
    )
    
    # Filter data by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & 
                        (df['date'] <= pd.to_datetime(end_date))]
    else:
        df_filtered = df
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Overview", "Time Series", "Volatility Analysis", "Correlation Analysis", "Risk-Return Analysis"],
        help="Choose the type of analysis to display"
    )
    
    # Main content area
    if analysis_type == "Overview":
        st.header("📊 Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Countries", df['country'].nunique())
        
        with col2:
            st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
        
        with col3:
            st.metric("Total Observations", f"{len(df):,}")
        
        with col4:
            data_completeness = (df['exchange_rate'].notna().sum() / len(df)) * 100
            st.metric("Data Completeness", f"{data_completeness:.1f}%")
        
        st.markdown("---")
        
        # Currency statistics table
        st.subheader("📈 Currency Statistics")
        
        currency_stats = get_currency_stats(df_filtered)
        currency_stats_display = currency_stats[currency_stats['country'].isin(selected_countries)]
        
        display_cols = ['country', 'exchange_rate_mean', 'quarterly_return_mean', 
                       'quarterly_return_std', 'data_completeness']
        
        st.dataframe(
            currency_stats_display[display_cols].round(4),
            use_container_width=True,
            hide_index=True
        )
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Most Stable Currencies")
            most_stable = currency_stats.nsmallest(5, 'quarterly_return_std')[['country', 'quarterly_return_std']]
            st.dataframe(most_stable, hide_index=True)
        
        with col2:
            st.subheader("🌪️ Most Volatile Currencies")
            most_volatile = currency_stats.nlargest(5, 'quarterly_return_std')[['country', 'quarterly_return_std']]
            st.dataframe(most_volatile, hide_index=True)
    
    elif analysis_type == "Time Series":
        st.header("📈 Time Series Analysis")
        
        # Metric selection
        metric_options = {
            'exchange_rate': 'Exchange Rate',
            'quarterly_return': 'Quarterly Return (%)',
            'volatility_4q': '4-Quarter Rolling Volatility'
        }
        
        selected_metric = st.selectbox(
            "Select Metric",
            list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )
        
        # Create and display time series plot
        fig = create_time_series_plot(df_filtered, selected_countries, selected_metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        summary_stats = df_filtered[df_filtered['country'].isin(selected_countries)].groupby('country')[selected_metric].describe().round(4)
        st.dataframe(summary_stats, use_container_width=True)
    
    elif analysis_type == "Volatility Analysis":
        st.header("🌪️ Volatility Analysis")
        
        # Volatility comparison chart
        fig_vol = create_volatility_comparison(df_filtered, selected_countries)
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Volatility over time
        st.subheader("📈 Volatility Over Time")
        fig_vol_time = create_time_series_plot(df_filtered, selected_countries, 'volatility_4q')
        st.plotly_chart(fig_vol_time, use_container_width=True)
        
        # Volatility statistics
        st.subheader("📊 Volatility Statistics")
        vol_stats = df_filtered[df_filtered['country'].isin(selected_countries)].groupby('country').agg({
            'quarterly_return': ['std', 'min', 'max'],
            'volatility_4q': ['mean', 'std']
        }).round(4)
        vol_stats.columns = ['_'.join(col).strip() for col in vol_stats.columns]
        st.dataframe(vol_stats, use_container_width=True)
    
    elif analysis_type == "Correlation Analysis":
        st.header("🔗 Correlation Analysis")
        
        if len(selected_countries) < 2:
            st.warning("⚠️ Please select at least 2 countries for correlation analysis.")
        else:
            # Correlation heatmap
            fig_corr = create_correlation_heatmap(df_filtered, selected_countries)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation matrix table
            st.subheader("📊 Correlation Matrix")
            pivot_data = df_filtered[df_filtered['country'].isin(selected_countries)].pivot_table(
                index='date', columns='country', values='quarterly_return'
            )
            correlation_matrix = pivot_data.corr().round(4)
            st.dataframe(correlation_matrix, use_container_width=True)
    
    elif analysis_type == "Risk-Return Analysis":
        st.header("⚖️ Risk-Return Analysis")
        
        # Risk-return scatter plot
        fig_risk = create_risk_return_scatter(df_filtered, selected_countries)
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Risk metrics table
        st.subheader("📊 Risk Metrics")
        risk_stats = df_filtered[df_filtered['country'].isin(selected_countries)].groupby('country').agg({
            'quarterly_return': ['mean', 'std', lambda x: np.percentile(x.dropna(), 5)]
        }).round(4)
        risk_stats.columns = ['Mean Return', 'Volatility', 'VaR (5%)']
        risk_stats['Sharpe Ratio'] = (risk_stats['Mean Return'] / risk_stats['Volatility']).round(4)
        st.dataframe(risk_stats, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        📊 Exchange Rate Analysis Dashboard | Built with Streamlit | Data: 2001-2016
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
