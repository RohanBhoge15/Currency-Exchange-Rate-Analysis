"""
Helper Functions for Exchange Rate Analysis
==========================================

Reusable utility functions for data processing, analysis, and visualization.

Author: Rohan Bhoge
Created: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def save_plot(fig, filename, folder='visualizations/static_plots'):
    """Save matplotlib figure to specified folder"""
    import os
    os.makedirs(folder, exist_ok=True)
    filepath = f"{folder}/{filename}"
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved: {filepath}")

def calculate_summary_stats(df, group_col='country', value_col='quarterly_return'):
    """Calculate comprehensive summary statistics"""
    stats = df.groupby(group_col)[value_col].agg([
        'count', 'mean', 'std', 'min', 'max', 'skew'
    ]).round(4)
    return stats

def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

def format_percentage(value, decimals=2):
    """Format number as percentage"""
    return f"{value:.{decimals}f}%"

def create_date_range(start_date, end_date, freq='Q'):
    """Create date range for time series analysis"""
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def export_results_to_excel(dataframes_dict, filename='analysis_results.xlsx'):
    """Export multiple DataFrames to Excel with separate sheets"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"📊 Results exported to: {filename}")

# Analysis-specific functions
def calculate_var(returns, confidence_level=0.05):
    """Calculate Value at Risk"""
    return np.percentile(returns.dropna(), confidence_level * 100)

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

def get_crisis_periods():
    """Return predefined crisis periods for analysis"""
    return {
        'financial_crisis': ('2007-06-01', '2009-06-30'),
        'european_debt_crisis': ('2010-01-01', '2012-12-31'),
        'china_slowdown': ('2015-06-01', '2016-03-31')
    }

print("🛠️ Helper functions loaded successfully!")
