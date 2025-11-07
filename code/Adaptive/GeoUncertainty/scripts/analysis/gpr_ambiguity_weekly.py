import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

def load_and_prepare_data():
    # Load monthly GPR and daily ambiguity data
    gpr_monthly_path = '/Users/tlxy/Research/Ambiguity/data/data_gpr_export.xls'
    ambiguity_daily_path = '/Users/tlxy/Research/Ambiguity/data/daily_ambiguity_risk_metrics.csv'
    
    gpr_monthly = pd.read_excel(gpr_monthly_path)
    ambiguity_daily = pd.read_csv(ambiguity_daily_path)
    
    # --- Prepare GPR data ---
    gpr_monthly['Date'] = pd.to_datetime(gpr_monthly['month'], format='%Ym%m')
    gpr_monthly.set_index('Date', inplace=True)
    
    gpr_columns_orig = ['GPRHC_USA', 'GPRHC_CHN', 'GPRHC_JPN']
    gpr_columns_new = ['GPR_US', 'GPR_China', 'GPR_Japan']
    
    gpr_monthly = gpr_monthly[gpr_columns_orig]
    gpr_monthly.columns = gpr_columns_new
    
    gpr_weekly = gpr_monthly.resample('D').interpolate(method='linear').resample('W').mean()

    # --- Prepare Ambiguity data ---
    ambiguity_daily['Date'] = pd.to_datetime(ambiguity_daily['date'])
    ambiguity_daily = ambiguity_daily[['Date', 'ambiguity_metric']]
    ambiguity_daily.set_index('Date', inplace=True)
    ambiguity_weekly = ambiguity_daily.resample('W').mean()
    
    # --- Merge data ---
    data = pd.merge(ambiguity_weekly, gpr_weekly, on='Date', how='inner')
    
    ambiguity_metric = 'ambiguity_metric'
    
    data = data[[ambiguity_metric] + gpr_columns_new].dropna()
    
    print("Data loading and preparation complete.")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data, ambiguity_metric, gpr_columns_new

def run_benchmark_regressions(data, ambiguity_metric, gpr_columns):
    y = data[ambiguity_metric]
    X = data[gpr_columns]
    X = sm.add_constant(X)
    
    # Linear Model
    model_linear = sm.OLS(y, X).fit()
    
    return model_linear

def run_lag_analysis(data, ambiguity_metric, gpr_columns):
    data_lag = data.copy()
    for col in gpr_columns:
        for i in range(1, 4): # Lags from 1 to 3 weeks
            data_lag[f'{col}_lag{i}'] = data_lag[col].shift(i)
    
    data_lag.dropna(inplace=True)
    
    y = data_lag[ambiguity_metric]
    X = data_lag[[col for col in data_lag.columns if 'GPR' in col]]
    X = sm.add_constant(X)
    
    model_lag = sm.OLS(y, X).fit()
    
    return model_lag

def summarize_results(model_linear, model_lag):
    summary = "### Key Findings from Weekly Analysis\n"
    summary += "1.  **Model Performance**:\n"
    summary += f"    -   **Linear Model R²**: {model_linear.rsquared:.4f}\n"
    summary += f"    -   **Lag Model R²**: {model_lag.rsquared:.4f}\n"

    summary += "\n2.  **Regression Summaries**:\n"
    summary += "    **Linear Model**:\n"
    summary += str(model_linear.summary()) + "\n"
    summary += "    **Lag Model**:\n"
    summary += str(model_lag.summary()) + "\n"
    
    print(summary)
    return summary

def main():
    data, ambiguity_metric, gpr_columns = load_and_prepare_data()
    
    model_linear = run_benchmark_regressions(data, ambiguity_metric, gpr_columns)
    model_lag = run_lag_analysis(data, ambiguity_metric, gpr_columns)
    
    summarize_results(model_linear, model_lag)

if __name__ == "__main__":
    main()