# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 16:07:35 2025

@author: Richard
"""

# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
from typing import Tuple, List, Union  # For type hints
from datetime import date  # For date handling

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the input DataFrame by calculating returns and extracting dates.
    This function handles the initial data preprocessing steps.
    
    Args:
        df: DataFrame with 'datetime' and 'close' columns
        
    Returns:
        DataFrame with added 'return' and 'date' columns, sorted by datetime
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Convert datetime column to pandas datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort data chronologically to ensure correct return calculations
    df.sort_values('datetime', inplace=True)
    
    # Calculate percentage returns (price changes)
    # pct_change() computes the percentage change between consecutive elements
    df['return'] = df['close'].pct_change()
    
    # Extract date component from datetime for daily grouping
    df['date'] = df['datetime'].dt.date
    
    # Remove rows with NaN returns (first row will have NaN as there's no previous price)
    return df.dropna(subset=['return'])

def get_window_dates(df: pd.DataFrame, specific_date: Union[str, date], 
                     window_size: int) -> List[date]:
    """
    Get the dates for the rolling window analysis.
    This function ensures we have enough data for the analysis and returns
    the appropriate date range.
    
    Args:
        df: Prepared DataFrame with 'date' column
        specific_date: Target date for analysis
        window_size: Number of days in rolling window
        
    Returns:
        List of dates in the rolling window
        
    Raises:
        ValueError: If specific_date not found or insufficient data
    """
    # Convert specific_date to datetime.date object if it's a string
    specific_date = pd.to_datetime(specific_date).date()
    
    # Get unique dates and sort them chronologically
    unique_dates = sorted(df['date'].unique())
    
    # Validate that the specific date exists in our data
    if specific_date not in unique_dates:
        raise ValueError("Specific date not found in the data.")
    
    # Find the index of our target date
    date_index = unique_dates.index(specific_date)
    
    # Check if we have enough historical data for the window
    if date_index < window_size - 1:
        raise ValueError("Not enough data for the specified window size.")
    
    # Return the window of dates (including the specific date)
    return unique_dates[date_index - window_size + 1 : date_index + 1]

def calculate_daily_probabilities(window_data: pd.DataFrame, 
                                num_bins: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Calculate probability distributions for each day in the window.
    This function creates histograms of returns and converts them to probability distributions.
    
    Args:
        window_data: DataFrame containing window data with 'return' and 'date' columns
        num_bins: Number of bins for histogram (determines granularity of distribution)
        
    Returns:
        Tuple containing:
        - DataFrame of daily probability distributions
        - Array of bin edges used for the histograms
    """
    # Find the range of returns in our window
    min_return = window_data['return'].min()
    max_return = window_data['return'].max()
    
    # Create evenly spaced bins across the return range
    bins = np.linspace(min_return, max_return, num_bins + 1)
    
    # Calculate probability distribution for each day
    daily_probs = []
    for date in window_data['date'].unique():
        # Get returns for this specific day
        daily_returns = window_data[window_data['date'] == date]['return']
        
        # Create histogram of returns
        counts, _ = np.histogram(daily_returns, bins=bins)
        
        # Convert counts to probabilities
        # If there are no returns (counts.sum() == 0), use zeros
        probabilities = counts / counts.sum() if counts.sum() > 0 else np.zeros(num_bins)
        daily_probs.append(probabilities)
    
    # Create DataFrame with dates as index and probabilities as columns
    return pd.DataFrame(daily_probs, index=window_data['date'].unique()), bins

def calculate_risk(df: pd.DataFrame, window_dates: List[date]) -> float:
    """
    Calculate risk as standard deviation of daily returns.
    This is a traditional measure of volatility/risk in financial markets.
    
    Args:
        df: Original DataFrame with price data
        window_dates: List of dates in the analysis window
        
    Returns:
        Risk metric (standard deviation of daily returns)
    """
    # Get the last price of each day
    daily_close = df.groupby('date')['close'].last()
    
    # Calculate daily returns
    daily_returns = daily_close.pct_change().dropna()
    
    # Filter returns for our window
    window_daily_returns = daily_returns.loc[window_dates]
    
    # Calculate standard deviation of daily returns
    return window_daily_returns.std()

def calculate_ambiguity_and_risk(df: pd.DataFrame, 
                               specific_date: Union[str, date],
                               window_size: int = 5, 
                               num_bins: int = 20) -> Tuple[float, float]:
    """
    Main function to calculate ambiguity and risk metrics for a specific date.
    This function orchestrates the entire analysis process.
    
    Args:
        df: DataFrame with 'datetime' and 'close' columns
        specific_date: Target date for analysis
        window_size: Number of days in rolling window (default: 5)
        num_bins: Number of bins for histogram (default: 20)
        
    Returns:
        Tuple containing:
        - ambiguity_metric: Average standard deviation across bins
        - interval_std: Standard deviations for each bin
        - risk: Standard deviation of daily returns
    """
    # Step 1: Prepare the data
    df = prepare_data(df)
    
    # Step 2: Get the dates for our analysis window
    window_dates = get_window_dates(df, specific_date, window_size)
    
    # Step 3: Filter data for our window
    window_data = df[df['date'].isin(window_dates)].copy()
    
    # Step 4: Calculate probability distributions
    prob_df, _ = calculate_daily_probabilities(window_data, num_bins)
    
    # Step 5: Calculate ambiguity metrics using cross entropy
    # Compute average probability distribution q(i) over the window
    q = prob_df.mean(axis=0)  # Average probability for each bin across days
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    q = q + epsilon
    q = q / q.sum()  # Re-normalize to ensure sum to 1

    # Find the target date's index in prob_df
    target_date = pd.to_datetime(specific_date).date()
    if target_date not in prob_df.index:
        daily_ambiguity_metric = float('nan')
    else:
        # Get p(i,t) for the target date
        p = prob_df.loc[target_date]
        # Calculate cross entropy: H(p, q) = -sum(p(i) * log(q(i)))
        daily_ambiguity_metric = -np.sum(p * np.log(q))
    
    # Step 6: Calculate risk
    risk = calculate_risk(df, window_dates)
    
    return daily_ambiguity_metric, risk

def load_and_prepare_data(file_path):
    # Read the CSV with the expected columns
    df = pd.read_csv(file_path, usecols=['datetime_nano', 'SSE.000300.close'])
    df.rename(columns={'SSE.000300.close': 'close'}, inplace=True)
    
    # Convert 'datetime_nano' to a datetime object.
    # First interpret as UTC date, then convert to local time (Asia/Shanghai), then floor to minute.
    df['datetime'] = (pd.to_datetime(df['datetime_nano'], utc=True)
                        .dt.tz_convert('Asia/Shanghai')
                        .dt.floor('5min')
                        .dt.tz_localize(None))
# 按五分钟时间戳聚合数据，取每个五分钟窗口的最后一个收盘价
    df = df.groupby('datetime').agg({'close': 'last'}).reset_index()
    
    # 按时间排序
    df.sort_values('datetime', inplace=True)
    

    df.drop('datetime_nano', axis=1, inplace=True, errors='ignore')
    return df

    
# Example usage
if __name__ == "__main__":
    # Example of how to use the code
    # Note: You need to have a DataFrame 'df' with 'datetime' and 'close' columns
    file_path = 'C:/Users/Richard/Desktop/数据/沪深300/SSE.000300.csv'  # Ensure the path is correct
    df = load_and_prepare_data(file_path)
    # Set parameters
    specific_date = '2018-01-09'
    window_size = 5  # 5-day window
    num_bins = 50    # 20 bins for histogram
    
    # Calculate metrics
    ambiguity_metric, risk = calculate_ambiguity_and_risk(
        df, 
        specific_date, 
        window_size=window_size, 
        num_bins=num_bins
    )
    
    # Print results
    print(f"Ambiguity Metric on {specific_date}: {ambiguity_metric}")
    print(f"Risk (Standard Deviation of Daily Returns) on {specific_date}: {risk}")
    #print("Standard deviation across bins:")
    #print(interval_std)
    
    
# Import additional required libraries for visualization and analysis
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For statistical visualizations
from scipy import stats  # For statistical tests
from typing import Tuple, List, Union  # For type hints
from datetime import date

def calculate_metrics_for_date_range(df: pd.DataFrame, 
                                   start_date: str,
                                   window_size: int = 5,
                                   num_bins: int = 50) -> pd.DataFrame:
    """
    Calculate ambiguity and risk metrics for a range of dates.
    
    Args:
        df: DataFrame with 'datetime' and 'close' columns
        start_date: Start date for analysis (format: 'YYYY-MM-DD')
        window_size: Number of days in rolling window
        num_bins: Number of bins for histogram
        
    Returns:
        DataFrame with columns: date, ambiguity_metric, risk
    """
    # Prepare the data using existing function
    df = prepare_data(df)
    
    # Get unique dates and sort them
    unique_dates = sorted(df['date'].unique())
    
    # Convert start_date to datetime.date
    start_date = pd.to_datetime(start_date).date()
    
    # Filter dates from start_date onwards
    analysis_dates = [d for d in unique_dates if d >= start_date]
    
    # Initialize lists to store results
    results = []
    
    # Calculate metrics for each date
    for current_date in analysis_dates:
        try:
            # Use existing function to calculate metrics
            ambiguity_metric, risk = calculate_ambiguity_and_risk(
                df,
                current_date,
                window_size=window_size,
                num_bins=num_bins
            )
            
            # Store results
            results.append({
                'date': current_date,
                'ambiguity_metric': ambiguity_metric,
                'risk': risk
            })
            
        except ValueError as e:
            # Skip dates where we don't have enough data
            print(f"Skipping {current_date}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Remove rows with NaN values
    results_df = results_df.dropna()
    
    return results_df

def analyze_correlation(results_df: pd.DataFrame) -> None:
    """
    Analyze and visualize the correlation between ambiguity and risk metrics.
    
    Args:
        results_df: DataFrame containing date, ambiguity_metric, and risk columns
    """
    # Calculate correlation coefficient
    correlation = results_df['ambiguity_metric'].corr(results_df['risk'])
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Time series plot
    plt.subplot(2, 2, 1)
    plt.plot(results_df['date'], results_df['ambiguity_metric'], label='Ambiguity')
    plt.plot(results_df['date'], results_df['risk'], label='Risk')
    plt.title('Ambiguity and Risk Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 2. Scatter plot with regression line
    plt.subplot(2, 2, 2)
    sns.regplot(data=results_df, x='ambiguity_metric', y='risk')
    plt.title(f'Correlation: {correlation:.3f}')
    plt.xlabel('Ambiguity Metric')
    plt.ylabel('Risk')
    
    # 3. Histogram of ambiguity metric
    plt.subplot(2, 2, 3)
    sns.histplot(results_df['ambiguity_metric'], kde=True)
    plt.title('Distribution of Ambiguity Metric')
    plt.xlabel('Ambiguity Metric')
    
    # 4. Histogram of risk
    plt.subplot(2, 2, 4)
    sns.histplot(results_df['risk'], kde=True)
    plt.title('Distribution of Risk')
    plt.xlabel('Risk')
    
    plt.tight_layout()
    plt.savefig('ambiguity_risk_analysis_1.png')
    plt.close()
    
    # Print statistical summary
    print("\nStatistical Summary:")
    print(results_df.describe())
    
    # Print correlation analysis
    print("\nCorrelation Analysis:")
    print(f"Pearson Correlation Coefficient: {correlation:.3f}")
    
    # Perform statistical test
    t_stat, p_value = stats.pearsonr(results_df['ambiguity_metric'], results_df['risk'])
    print(f"P-value: {p_value:.3e}")
    print(f"T-statistic: {t_stat:.3f}")

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    file_path = 'C:/Users/Richard/Desktop/数据/沪深300/SSE.000300.csv'
    df = load_and_prepare_data(file_path)
    
    # Set parameters
    start_date = '2018-01-09'
    window_size = 5
    num_bins = 50
    
    # Calculate metrics for the date range
    print("Calculating metrics...")
    results_df = calculate_metrics_for_date_range(
        df,
        start_date,
        window_size=window_size,
        num_bins=num_bins
    )
    
    # Save results to CSV
    results_df.to_csv('daily_ambiguity_risk_metrics_2.csv', index=False)
    print("\nResults saved to '300_daily_ambiguity_risk_metrics_2.csv'")
    
    # Analyze correlation and create visualizations
    print("\nAnalyzing correlation...")
    analyze_correlation(results_df)
    
    # Display first few rows of results
    print("\nFirst few rows of results:")
    print(results_df.head())
    
    # Calculate rolling correlation (30-day window)
    rolling_corr = results_df['ambiguity_metric'].rolling(window=30).corr(results_df['risk'])
    
    # Plot rolling correlation
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['date'], rolling_corr)
    plt.title('30-Day Rolling Correlation between Ambiguity and Risk')
    plt.xlabel('Date')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.savefig('rolling_correlation_1.png')
    plt.close()
    
    print("\nAnalysis complete. Check the generated CSV and PNG files for detailed results.")
    
    
    
import itertools

# Define parameter ranges
window_sizes = list(range(5, 21, 3))   # 5, 8, 11, 14, 17, 20
num_bins_list = list(range(20, 101, 10))  # 20, 30, ..., 100

# Prepare data
file_path = 'C:/Users/Richard/Desktop/数据/沪深300/SSE.000300.csv'
df = load_and_prepare_data(file_path)
df = prepare_data(df)
unique_dates = sorted(df['date'].unique())

# Only use dates where all window sizes are possible
min_window = min(window_sizes)
analysis_dates = unique_dates[min_window-1:]

# Prepare result storage
results = {'date': analysis_dates}

# For each parameter combination, calculate ambiguity metric for all dates
for window_size, num_bins in itertools.product(window_sizes, num_bins_list):
    col_name = f"{window_size}d{num_bins}b"
    metrics = []
    for current_date in analysis_dates:
        try:
            ambiguity_metric, _ = calculate_ambiguity_and_risk(
                df, current_date, window_size=window_size, num_bins=num_bins
            )
        except Exception:
            ambiguity_metric = float('nan')
        metrics.append(ambiguity_metric)
    results[col_name] = metrics

# Convert to DataFrame and save
import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv('daily_ambiguity_var_para_2.csv', index=False)
print("Saved to daily_ambiguity_var_para_2.csv")
