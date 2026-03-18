import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ── 1. DATA PREPARATION ──────────────────────────────────────────────────────
def prepare_data():
    df = pd.read_csv('countries of the world.csv', decimal=',')
    df.columns = df.columns.str.strip()
    df['Country'] = df['Country'].str.strip()
    df['Region']  = df['Region'].str.strip()
    
    # Imputation (Regional Median)
    for col in df.columns:
        if df[col].isnull().sum() == 0: continue
        if col == 'Climate':
            guess = df.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
        else:
            guess = df.groupby('Region')[col].median()
        for region in df['Region'].unique():
            mask = (df[col].isnull()) & (df['Region'] == region)
            df.loc[mask, col] = guess[region]
    return df

df = prepare_data()

# ── 2. FEATURE ENGINEERING (MTech Level) ─────────────────────────────────────
print("\n[STEP 1] PERFORMING FEATURE ENGINEERING...")
# A. Tech-to-Wealth Ratio: Does more tech infrastructure lead to better wealth efficiency?
df['Tech_Wealth_Ratio'] = df['Phones (per 1000)'] / (df['GDP ($ per capita)'] + 1)

# B. Industry Efficiency: How much industrial output per person density?
df['Ind_Efficiency'] = df['Industry'] / (df['Pop. Density (per sq. mi.)'] + 1)

# C. Service Dominance Index: Measure of modernization (Service / (Agri + 0.001))
df['Service_Dominance'] = df['Service'] / (df['Agriculture'] + 0.001)

print("✅ New Features Created: Tech_Wealth_Ratio, Ind_Efficiency, Service_Dominance")

# ── 3. STATISTICAL INFERENCE (Scientific Rigor) ──────────────────────────────
print("\n[STEP 2] RUNNING STATISTICAL TESTS...")

# A. Shapiro-Wilk Test for Normality
print("\n--- Shapiro-Wilk Test (Normality of GDP) ---")
stat_raw, p_raw = stats.shapiro(df['GDP ($ per capita)'])
stat_log, p_log = stats.shapiro(np.log1p(df['GDP ($ per capita)']))

print(f"Raw GDP: p-value = {p_raw:.5f} ({'Normal' if p_raw > 0.05 else 'Not Normal'})")
print(f"Log GDP: p-value = {p_log:.5f} ({'Normal' if p_log > 0.05 else 'Significantly improved'})")

# B. One-Way ANOVA (Regional Differences)
print("\n--- One-Way ANOVA (GDP vs Region) ---")
regions = [df[df['Region'] == r]['GDP ($ per capita)'] for r in df['Region'].unique()]
f_stat, p_val = stats.f_oneway(*regions)
print(f"F-Statistic: {f_stat:.2f}")
print(f"p-value: {p_val:.5e}")
if p_val < 0.05:
    print("✅ Result: Statistically significant differences exist between regional GDPs.")

# C. Outlier Detection (Z-score & IQR)
print("\n--- Outlier Detection (GDP) ---")
# Z-score method (threshold = 3)
z_scores = np.abs(stats.zscore(df['GDP ($ per capita)']))
outliers_z = df[z_scores > 3]

# IQR method
Q1 = df['GDP ($ per capita)'].quantile(0.25)
Q3 = df['GDP ($ per capita)'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['GDP ($ per capita)'] < (Q1 - 1.5 * IQR)) | (df['GDP ($ per capita)'] > (Q3 + 1.5 * IQR))]

print(f"Z-score outliers: {len(outliers_z)} detected")
print(f"IQR outliers:     {len(outliers_iqr)} detected")
print("✅ Outlier validation completed for publication-quality reporting.")

# ── 4. MACHINE LEARNING (Predictive Modeling) ────────────────────────────────
print("\n[STEP 3] BUILDING PREDICTIVE MODELS...")

# Features and Target
X = df.drop(['Country', 'Region', 'GDP ($ per capita)'], axis=1)
y = df['GDP ($ per capita)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# A. Linear Regression (Baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# B. Random Forest (Advanced)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation Metrics
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\nModel: {name}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R2 Score: {r2:.3f}")
    return mae, rmse, r2

evaluate(y_test, y_pred_lr, "Linear Regression (Baseline)")
evaluate(y_test, y_pred_rf, "Random Forest (MTech Standard)")

# ── 5. VISUALIZATION OF RESULTS ──────────────────────────────────────────────
plt.figure(figsize=(16, 10))
plt.style.use('seaborn-v0_8-whitegrid')

# Chart 1: Feature Importance (The "Insight" chart)
plt.subplot(2, 2, 1)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
sns.barplot(x=importances.values, y=importances.index, palette='magma')
plt.title('Top 10 Features for GDP Prediction', fontweight='bold')

# Chart 2: Predicted vs Actual (RF)
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Actual vs Predicted (Random Forest)', fontweight='bold')

# Chart 3: ANOVA Results (Boxplot)
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='GDP ($ per capita)', y='Region', palette='viridis')
plt.title('Regional GDP Variance (ANOVA basis)', fontweight='bold')

# Chart 4: Service Dominance Index vs GDP
plt.subplot(2, 2, 4)
plt.scatter(df['Service_Dominance'], df['GDP ($ per capita)'], alpha=0.5, c=df['GDP ($ per capita)'], cmap='cool')
plt.xlabel('Service Dominance Index')
plt.ylabel('GDP per Capita')
plt.title('Impact of Service Sector Dominance on Wealth', fontweight='bold')

plt.tight_layout()
plt.savefig('mtech_results.png')
print("\n✅ Advanced Analytics Complete! Plots saved to 'mtech_results.png'")
plt.show()
