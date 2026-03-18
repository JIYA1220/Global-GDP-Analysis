<div align="center">

# Global GDP Analysis: Socio-Economic Drivers of National Wealth

A complete data science workflow from data engineering and statistical testing to machine learning and a Power BI dashboard.

<br/>

<!-- Tech icons row -->
<p>
  <img src="https://skillicons.dev/icons?i=python,pycharm,git,github" height="48" />
</p>

<!-- Badge row (includes Power BI + data/ML stack) -->
<p>
  <img src="https://img.shields.io/badge/Power%20BI-Dashboarding-F2C811?style=for-the-badge&logo=powerbi&logoColor=000000" />
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Seaborn-Statistical%20Viz-4C72B0?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SciPy-Statistics-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

</div>

---

## Overview

This project investigates why some nations are wealthier than others by analyzing a dataset of 227 countries with 20 socio-economic features spanning demographics, infrastructure, and sector composition. The workflow combines rigorous cleaning, statistical validation, feature engineering, predictive modeling, and interactive reporting.

---

## Pipeline Summary

### 1) Data Engineering and Integrity
- Dataset: 227 countries with 20 socio-economic indicators
- Regional median imputation for missing values (geography-aware cleaning)
- Data scrubbing: numeric parsing, decimal-comma fixes, whitespace cleanup

### 2) Exploratory Analysis and Statistical Awareness
- Identified strong right-skew in GDP per Capita
- Applied log1p transformation to stabilize the target for modeling
- Outlier profiling using Z-score and IQR

### 3) Feature Engineering
Engineered efficiency-style features to capture deeper economic structure:
- Tech-to-Wealth Ratio
- Industry Efficiency
- Service Dominance Index

### 4) Statistical Testing
- Shapiro-Wilk test to validate distribution assumptions and transformation impact
- One-way ANOVA to test whether regional differences are statistically significant (p < 0.05)

### 5) Predictive Modeling
Compared:
- Linear Regression (baseline)
- Random Forest Regressor (non-linear ensemble)

Metrics:
- RMSE
- MAE

Also includes feature importance analysis for interpretability.

### 6) Power BI Dashboard
A 2-page Power BI dashboard presenting:
- global maps and ranking views
- distribution and skew analysis
- correlation heatmaps
- sector composition analysis
- log-scale relationship plots
- Python visuals (Matplotlib/Seaborn) embedded where needed

---

## Results and Visual Outputs

### Section 01: Global Wealth and Data Integrity
![Section 01 - Global Wealth and Data Integrity](assets/section-01.png)

### Section 02: Advanced Correlation and Sector Analysis
![Section 02 - Advanced Correlation and Sector Analysis](assets/section-02.png)

### Model and Statistical Results
![Model and Statistical Results](assets/model-results.png)

---

## Repository Structure

- `countries of the world.csv` - source dataset
- `mtech_advanced.py` - main Python pipeline (cleaning, EDA, stats, feature engineering, ML)
- `export_for_powerbi.py` - export script for BI-ready dataset
- `powerbi_data.csv` - curated dataset used by Power BI
- `data analysis.pbix` - Power BI dashboard file

---

## How to Run

Run the core analysis:
```bash
python mtech_advanced.py
```

Export dataset for Power BI:
```bash
python export_for_powerbi.py
```

Open the dashboard:
- Open `data analysis.pbix` in Power BI Desktop

---

