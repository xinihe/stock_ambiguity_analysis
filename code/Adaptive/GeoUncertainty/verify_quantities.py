import pandas as pd
import numpy as np
from scipy import stats

def ols(y, X):
    # Add constant
    X = np.column_stack([np.ones(len(X)), X])
    # Solve (X'X)^-1 X'y
    try:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        # Residuals
        y_pred = X @ beta
        resid = y - y_pred
        sigma2 = np.sum(resid**2) / (len(y) - X.shape[1])
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov_beta))
        return beta, se
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1]), np.zeros(X.shape[1])

# Load data
df = pd.read_csv('/Users/tlxy/Research/Ambiguity/data/com_daily_data.csv')

# Create Delta variables
df['d_GPR'] = df['GPRD'].diff()
df['d_Amb1'] = df['ambiguity_metric_1'].diff()
df['d_Amb2'] = df['ambiguity_metric_2'].diff()
df['d_Risk1'] = df['risk_1'].diff()
df['d_Risk2'] = df['risk_2'].diff()

# Drop NA
df_clean = df.dropna()

# 1. Check SDs and Standardized Effects of GPR on Amb/Risk
# Target: 0.42 (Amb) vs 0.18 (Risk)
# We test both Amb1 and Amb2
# Standardized Beta = b * SD(X) / SD(Y)

sd_GPR = df_clean['d_GPR'].std()
sd_Amb1 = df_clean['d_Amb1'].std()
sd_Amb2 = df_clean['d_Amb2'].std()
sd_Risk1 = df_clean['d_Risk1'].std()
sd_Risk2 = df_clean['d_Risk2'].std()
sd_Ret = df_clean['daily_return'].std()

print(f"SD(d_GPR): {sd_GPR}")
print(f"SD(d_Amb1): {sd_Amb1}")
print(f"SD(d_Amb2): {sd_Amb2}")
print(f"SD(d_Risk1): {sd_Risk1}")
print(f"SD(d_Risk2): {sd_Risk2}")

# Regression: d_Amb ~ d_GPR
beta_Amb1_GPR, _ = ols(df_clean['d_Amb1'], df_clean[['d_GPR']])
std_beta_Amb1 = beta_Amb1_GPR[1] * sd_GPR / sd_Amb1
print(f"Std Beta GPR -> Amb1: {std_beta_Amb1}")

beta_Amb2_GPR, _ = ols(df_clean['d_Amb2'], df_clean[['d_GPR']])
std_beta_Amb2 = beta_Amb2_GPR[1] * sd_GPR / sd_Amb2
print(f"Std Beta GPR -> Amb2: {std_beta_Amb2}")

# Regression: d_Risk ~ d_GPR
beta_Risk1_GPR, _ = ols(df_clean['d_Risk1'], df_clean[['d_GPR']])
std_beta_Risk1 = beta_Risk1_GPR[1] * sd_GPR / sd_Risk1
print(f"Std Beta GPR -> Risk1: {std_beta_Risk1}")

# Standardized regression (z-scored) to interpret SD changes
z = lambda s: (s - s.mean()) / s.std()
beta_z_Amb1_GPR, _ = ols(z(df_clean['d_Amb1']).values, z(df_clean['d_GPR']).values.reshape(-1, 1))
beta_z_Risk1_GPR, _ = ols(z(df_clean['d_Risk1']).values, z(df_clean['d_GPR']).values.reshape(-1, 1))
print(f"Z-OLS slope GPR -> Amb1 (SD units): {beta_z_Amb1_GPR[1]}")
print(f"Z-OLS slope GPR -> Risk1 (SD units): {beta_z_Risk1_GPR[1]}")

# 2. Check 47% Inflation of Volatility Role
# Risk-Only: Ret ~ d_GPR + d_Risk
# Full: Ret ~ d_GPR + d_Risk + d_Amb
# We use Amb1/Risk1 first (check coeff magnitude)

# Risk-Only (Amb1)
beta_RiskOnly, _ = ols(df_clean['daily_return'], df_clean[['d_GPR', 'd_Risk1']])
coeff_Risk_RiskOnly = beta_RiskOnly[2]

# Full (Amb1)
beta_Full, _ = ols(df_clean['daily_return'], df_clean[['d_GPR', 'd_Risk1', 'd_Amb1']])
coeff_Risk_Full = beta_Full[2]
coeff_Amb_Full = beta_Full[3]

print(f"Risk-Only Coeff (Risk1): {coeff_Risk_RiskOnly}")
print(f"Full Model Coeff (Risk1): {coeff_Risk_Full}")
inflation = (coeff_Risk_RiskOnly - coeff_Risk_Full) / coeff_Risk_Full
print(f"Inflation (Risk1): {inflation * 100:.2f}%")

# Check with Amb2/Risk2
beta_RiskOnly2, _ = ols(df_clean['daily_return'], df_clean[['d_GPR', 'd_Risk2']])
coeff_Risk_RiskOnly2 = beta_RiskOnly2[2]
beta_Full2, _ = ols(df_clean['daily_return'], df_clean[['d_GPR', 'd_Risk2', 'd_Amb2']])
coeff_Risk_Full2 = beta_Full2[2]
inflation2 = (coeff_Risk_RiskOnly2 - coeff_Risk_Full2) / coeff_Risk_Full2
print(f"Inflation (Risk2): {inflation2 * 100:.2f}%")

# 3. Check 11.3 bps claim
# 1 SD Amb -> Returns. Coeff is coeff_Amb_Full.
impact_bps = coeff_Amb_Full * sd_Amb1 * 10000
print(f"Impact of 1 SD Amb1 (bps): {impact_bps:.2f}")
# Try Amb2
coeff_Amb_Full2 = beta_Full2[3]
impact_bps2 = coeff_Amb_Full2 * sd_Amb2 * 10000
print(f"Impact of 1 SD Amb2 (bps): {impact_bps2:.2f}")

# 4. Check 1.4x larger impact
# Impact Amb vs Impact Vol
impact_Vol_bps = coeff_Risk_Full * sd_Risk1 * 10000
print(f"Impact of 1 SD Risk1 (bps): {impact_Vol_bps:.2f}")
print(f"Ratio Amb1/Risk1: {impact_bps / impact_Vol_bps:.2f}")

impact_Vol_bps2 = coeff_Risk_Full2 * sd_Risk2 * 10000
print(f"Impact of 1 SD Risk2 (bps): {impact_Vol_bps2:.2f}")
print(f"Ratio Amb2/Risk2: {impact_bps2 / impact_Vol_bps2:.2f}")

# 5. Check 67 bps monthly underperformance
# Convert to monthly
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month').apply(lambda x: pd.Series({
    'ret': (1 + x['daily_return']).prod() - 1,
    'amb': x['ambiguity_metric_1'].mean(), # or last? usually mean or lag
    'risk': x['risk_1'].mean()
})).reset_index()

# Sort into portfolios (Median split) based on lagged Ambiguity?
# Simplified: Contemporaneous correlation or sorting
# Paper says "Minimum-variance portfolios underperform ... relative to ambiguity-aware portfolios"
# This might be complex optimization.
# Let's just check High Amb vs Low Amb return difference
high_amb = monthly[monthly['amb'] > monthly['amb'].median()]['ret'].mean()
low_amb = monthly[monthly['amb'] <= monthly['amb'].median()]['ret'].mean()
print(f"High Amb Monthly Ret: {high_amb*100:.2f}%")
print(f"Low Amb Monthly Ret: {low_amb*100:.2f}%")
print(f"Diff (Low - High) (bps): {(low_amb - high_amb)*10000:.2f}")
