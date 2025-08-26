# Exchange Rate Analysis Project
## Comprehensive Analysis of Global Currency Movements (2001-2016)

---

## 📊 Project Overview

This project provides a comprehensive analysis of quarterly exchange rate data for 221 countries/territories from 2001 Q1 to 2016 Q3. We'll explore currency trends, volatility patterns, economic event impacts, and provide actionable insights for understanding global currency movements.

## 🎯 Business Context & Objectives

### Why Exchange Rate Analysis Matters:
- **International Trade**: Understanding currency movements for import/export decisions
- **Investment Strategy**: Currency risk assessment for international investments
- **Economic Policy**: Central bank and government policy impact analysis
- **Risk Management**: Hedging strategies for multinational corporations

### Key Stakeholders:
- Financial analysts and traders
- International business managers
- Economic researchers
- Policy makers

## 🔍 Research Questions

1. **Volatility Analysis**: Which currencies are most/least volatile over the study period?
2. **Trend Identification**: What are the long-term appreciation/depreciation trends?
3. **Crisis Impact**: How did major economic events (2008 financial crisis) affect different currencies?
4. **Regional Patterns**: Are there regional currency movement patterns?
5. **Currency Transitions**: How do countries manage currency changes/reforms?
6. **Stability Rankings**: Which currencies provide the most stability for international trade?
7. **Seasonal Effects**: Do currencies show seasonal patterns?
8. **Correlation Analysis**: Which currencies move together?

## 📈 Expected Outcomes

### Analytical Insights:
- Currency volatility rankings and risk profiles
- Economic event impact quantification
- Regional currency behavior patterns
- Currency stability recommendations

### Deliverables:
- Clean, analysis-ready dataset
- Comprehensive statistical analysis
- Interactive visualization dashboard
- Executive summary with actionable recommendations

## 🛠️ Technical Approach

### Data Processing:
- Pandas for data manipulation and cleaning
- Statistical analysis with scipy and statsmodels
- Time series analysis techniques

### Visualization:
- Static plots: matplotlib, seaborn
- Interactive visualizations: plotly
- Dashboard: Streamlit
- Geographic maps: folium

### Analysis Methods:
- Descriptive statistics and distribution analysis
- Correlation and regression analysis
- Time series decomposition
- Event study methodology
- Volatility modeling

## 📋 Project Structure

```
Exchange_Rate_Analysis/
├── 01_project_overview.md              # This document
├── 02_data_cleaning.ipynb             # Data preprocessing
├── 03_exploratory_analysis.ipynb      # EDA and initial insights
├── 04_statistical_analysis.ipynb      # Advanced statistical analysis
├── 05_economic_events.ipynb           # Event impact analysis
├── 06_time_series_analysis.ipynb      # Time series modeling
├── 07_advanced_analytics.ipynb        # Risk metrics and advanced analysis
├── 08_dashboard.py                    # Streamlit dashboard
├── README.md                          # Final report and documentation
├── data/
│   ├── quarterly-edited.csv           # Raw data
│   └── cleaned_exchange_rates.csv     # Processed data
├── visualizations/                    # Saved charts and plots
└── utils/
    └── helper_functions.py            # Reusable functions
```

## 📊 Dataset Overview

**Source Data**: `quarterly-edited.csv`
- **Rows**: 222 (including header)
- **Countries/Territories**: 221
- **Time Period**: 2001 Q1 to 2016 Q3 (63 quarters)
- **Format**: Wide format (quarters as columns)

### Key Features:
- Country names and currency units
- Quarterly exchange rates vs USD
- Mix of major and emerging market currencies
- Some currency transitions captured

## ⚠️ Data Considerations

### Potential Challenges:
- Missing values for some country-quarter combinations
- Currency regime changes (fixed vs floating)
- Economic sanctions affecting data availability
- Different currency denominations

### Quality Assumptions:
- Exchange rates are end-of-quarter values
- Data represents official exchange rates
- Missing values are genuinely unavailable (not errors)

## 🎯 Success Metrics

### Technical Quality:
- Clean, validated dataset with <5% missing values
- Reproducible analysis with documented methodology
- Professional-quality visualizations

### Business Value:
- Actionable insights for currency risk management
- Clear identification of stable vs volatile currencies
- Quantified impact of major economic events
- Regional currency behavior patterns

## 📅 Timeline

**Week 1**: Data cleaning and preprocessing
**Week 2**: Exploratory data analysis
**Week 3**: Statistical analysis and hypothesis testing
**Week 4**: Economic event analysis
**Week 5**: Time series analysis and advanced analytics
**Week 6**: Dashboard creation and final reporting

---

## 🚀 Next Steps

1. **Data Import and Initial Inspection** → `02_data_cleaning.ipynb`
2. **Data Quality Assessment**
3. **Data Cleaning and Preprocessing**
4. **Feature Engineering**

Let's begin our analysis journey! 📈

---

**Author**: Rohan Bhoge 
**Created**: 2025   
**Project**: Exchange Rate Analysis  
**Status**: Completed
