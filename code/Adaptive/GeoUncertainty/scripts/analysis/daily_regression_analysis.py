import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Load the combined daily data
df = pd.read_csv('/Users/tlxy/Research/Ambiguity/data/com_daily_data.csv')
df['date'] = pd.to_datetime(df['date'])

# --- Regression 1: Non-Linear Risk-Return Relationship ---
df['risk_1_sq'] = df['risk_1']**2
X1 = df[['ambiguity_metric_1', 'risk_1', 'risk_1_sq']]
X1 = sm.add_constant(X1)
y1 = df['daily_return']
model1 = sm.OLS(y1, X1).fit()

# --- Regression 2: GPR and Ambiguity (Weekly Analysis with Lags) ---
df_weekly = df.set_index('date').resample('W').mean()
df_weekly['GPRD_log'] = np.log(df_weekly['GPRD'])
df_weekly['GPRD_log_lag1'] = df_weekly['GPRD_log'].shift(1)
df_weekly['GPRD_log_lag2'] = df_weekly['GPRD_log'].shift(2)
df_weekly['GPRD_log_lag3'] = df_weekly['GPRD_log'].shift(3)
df_weekly['GPRD_log_lag4'] = df_weekly['GPRD_log'].shift(4)
df_weekly['GPRD_log_lag5'] = df_weekly['GPRD_log'].shift(5)
df_weekly = df_weekly.dropna()

X2 = df_weekly[['GPRD_log_lag1', 'GPRD_log_lag2', 'GPRD_log_lag3', 'GPRD_log_lag4', 'GPRD_log_lag5']]
X2 = sm.add_constant(X2)
y2 = df_weekly['ambiguity_metric_1']
model2 = sm.OLS(y2, X2).fit()

# --- Save results to daily_analysis_report.md ---
with open('/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/daily_analysis_report.md', 'w') as f:
    f.write("# Daily Data Analysis Report\n\n")
    f.write("## Regression 1: Non-Linear Risk-Return Relationship\n\n")
    f.write("### Model Specification\n\n")
    f.write("```\nReturns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(Risk²) + ε\n```\n\n")
    f.write("### Results\n\n")
    f.write(summary_col([model1], stars=True, float_format='%0.4f', model_names=['Model 1'], regressor_order=X1.columns.tolist()).as_text())
    f.write("\n\n")
    f.write("## Regression 2: GPR and Ambiguity (Weekly Analysis with Lags)\n\n")
    f.write("### Model Specification\n\n")
    f.write("```\nAmbiguity = β₀ + β₁(log(GPR)_lag_1) + β₂(log(GPR)_lag_2) + β₃(log(GPR)_lag_3) + β₄(log(GPR)_lag_4) + β₅(log(GPR)_lag_5) + ε\n```\n\n")
    f.write("### Results\n\n")
    f.write(summary_col([model2], stars=True, float_format='%0.4f', model_names=['Model 2'], regressor_order=X2.columns.tolist()).as_text())

print("Regressions complete. Results saved to /Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/daily_analysis_report.md")

# --- Stationarity Check ---
print("--- Stationarity Check (ADF Test) ---")
adf_ambiguity = adfuller(df_weekly['ambiguity_metric_1'])
print(f'ADF Statistic for ambiguity_metric_1: {adf_ambiguity[0]}')
print(f'p-value: {adf_ambiguity[1]}')

adf_gprd = adfuller(df_weekly['GPRD'])
print(f'ADF Statistic for GPRD: {adf_gprd[0]}')
print(f'p-value: {adf_gprd[1]}')
print("-----------------------------------------")

# --- Regression 2: Combined Model ---
df = df.dropna()
X2 = df[['ambiguity_metric_1', 'risk_1', 'GPRD']]
X2 = sm.add_constant(X2)
y2 = df['daily_return']
model2 = sm.OLS(y2, X2).fit()

# --- Save results to daily_analysis_report.md ---
with open('/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/daily_analysis_report.md', 'w') as f:
    f.write("# Daily Data Analysis Report\n\n")
    f.write("## Regression 1: Non-Linear Risk-Return Relationship\n\n")
    f.write("### Model Specification\n\n")
    f.write("```\nReturns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(Risk²) + ε\n```\n\n")
    f.write("### Results\n\n")
    f.write(summary_col([model1], stars=True, float_format='%0.4f', model_names=['Model 1'], regressor_order=X1.columns.tolist()).as_text())
    f.write("\n\n")
    f.write("## Regression 2: Combined Model for Daily Returns\n\n")
    f.write("### Model Specification\n\n")
    f.write("```\nReturns = β₀ + β₁(Ambiguity) + β₂(Risk) + β₃(GPRD) + ε\n```\n\n")
    f.write("### Results\n\n")
    f.write(summary_col([model2], stars=True, float_format='%0.4f', model_names=['Model 2'], regressor_order=X2.columns.tolist()).as_text())

print("Regressions complete. Results saved to /Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/outputs/results/daily_analysis_report.md")