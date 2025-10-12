import pandas as pd
import statsmodels.api as sm

def load_and_prepare_data(ambiguity_path, return_path, freq='M'):
    """Load, prepare, and merge ambiguity and return data."""
    ambiguity_data = pd.read_csv(ambiguity_path)
    return_data = pd.read_csv(return_path)

    # Prepare ambiguity data
    ambiguity_data['date'] = pd.to_datetime(ambiguity_data['date'])
    ambiguity_data = ambiguity_data.set_index('date')
    # Select only numeric columns for resampling
    ambiguity_numeric = ambiguity_data.select_dtypes(include=['number'])
    ambiguity_resampled = ambiguity_numeric.resample(freq).mean()

    # Prepare return data
    return_data.rename(columns={'datetime': 'date'}, inplace=True)
    return_data['date'] = pd.to_datetime(return_data['date'])
    return_data['return'] = return_data['SSE.000300.close'].pct_change()
    return_data = return_data.set_index('date')
    return_resampled = return_data['return'].resample(freq).sum().to_frame()

    # Merge data
    merged_data = pd.merge(ambiguity_resampled, return_resampled, on='date', how='inner')
    merged_data.dropna(inplace=True)
    merged_data.reset_index(inplace=True)
    
    return merged_data

def run_regression(data):
    """Run regression of return on ambiguity."""
    # Ensure data is not empty
    if data.empty:
        print("Data for regression is empty. Skipping analysis.")
        return None

    X = data['ambiguity_metric']
    y = data['return']
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model

def main():
    """Main function to run the analysis."""
    ambiguity_file = '/Users/tlxy/Research/Ambiguity/data/daily_ambiguity_risk_metrics.csv'
    return_file = '/Users/tlxy/Research/Ambiguity/data/SSE.000300.csv'

    # Monthly Analysis
    monthly_data = load_and_prepare_data(ambiguity_file, return_file, freq='M')
    monthly_model = run_regression(monthly_data)
    print("--- Monthly Analysis ---")
    if monthly_model:
        print(monthly_model.summary())

    # Weekly Analysis
    weekly_data = load_and_prepare_data(ambiguity_file, return_file, freq='W-FRI')
    weekly_model = run_regression(weekly_data)
    print("\n--- Weekly Analysis ---")
    if weekly_model:
        print(weekly_model.summary())

if __name__ == '__main__':
    main()