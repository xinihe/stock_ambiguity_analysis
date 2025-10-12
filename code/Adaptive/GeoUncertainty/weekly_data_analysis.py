import pandas as pd
import statsmodels.api as sm

def prepare_weekly_data():
    """Prepare the weekly data for analysis."""
    # Load data
    ambiguity_data = pd.read_csv("/Users/tlxy/Research/Ambiguity/data/daily_ambiguity_risk_metrics.csv")
    return_data = pd.read_csv("/Users/tlxy/Research/Ambiguity/data/SSE.000300.csv")
    gpr_data = pd.read_excel("/Users/tlxy/Research/Ambiguity/data/data_gpr_export.xls")

    # Prepare ambiguity data
    ambiguity_data['date'] = pd.to_datetime(ambiguity_data['date'])
    ambiguity_data = ambiguity_data.set_index('date')
    ambiguity_weekly = ambiguity_data.resample('W-FRI').mean()
    ambiguity_weekly['risk_sq'] = ambiguity_weekly['risk'] ** 2

    # Prepare return data
    return_data.rename(columns={'datetime': 'date'}, inplace=True)
    return_data['date'] = pd.to_datetime(return_data['date'])
    return_data = return_data.set_index('date')
    return_weekly = return_data['SSE.000300.close'].pct_change().resample('W-FRI').sum().to_frame()
    return_weekly.rename(columns={'SSE.000300.close': 'return'}, inplace=True)

    # Prepare GPR data
    gpr_data['date'] = pd.to_datetime(gpr_data['month'])
    gpr_data = gpr_data.set_index('date')
    gpr_china = gpr_data[['GPRC_CHN']]
    gpr_weekly_linear = gpr_china.resample('W-FRI').interpolate(method='linear')
    gpr_weekly_linear.rename(columns={'GPRC_CHN': 'gpr_china_linear'}, inplace=True)

    # Merge data
    merged_data = pd.merge(return_weekly, ambiguity_weekly, on='date', how='inner')
    merged_data = pd.merge(merged_data, gpr_weekly_linear, on='date', how='inner')
    merged_data.dropna(inplace=True)

    # Save to CSV
    merged_data.to_csv("/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/data/processed/weekly_data.csv")

    return merged_data

def main():
    """Main function to run the analysis."""
    weekly_data = prepare_weekly_data()

    # Correlation analysis
    correlation_matrix = weekly_data[['risk', 'risk_sq', 'ambiguity_metric', 'gpr_china_linear']].corr()
    print("--- Correlation Matrix ---")
    print(correlation_matrix)

    # Regression analysis
    X = weekly_data[['ambiguity_metric', 'risk', 'risk_sq']]
    y = weekly_data['return']
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print("\n--- Regression Results ---")
    print(model.summary())

if __name__ == '__main__':
    main()