"""
Ambiguity Measurement Module - Cross-Entropy Ambiguity (CEA) Index
Implements the A_CEA_t measure based on KL divergence between intraday return distributions
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class AmbiguityMeasurement:
    """
    Compute Cross-Entropy Ambiguity (A_CEA_t) index for stock returns

    The measure quantifies model uncertainty using KL divergence between
    empirical intraday return distributions and adaptive benchmark distributions
    """

    def __init__(self, n_bins=202, return_range=(-0.201, 0.201),
                 window_size=20, n_clusters=4, epsilon=1e-10):
        """
        Initialize the ambiguity measurement

        Parameters:
        -----------
        n_bins : int
            Number of bins for discretizing returns (default: 202)
        return_range : tuple
            Range for binning returns (default: (-0.201, 0.201))
        window_size : int
            Size of rolling window for benchmark selection (default: 20)
        n_clusters : int
            Number of clusters for regime identification (default: 4)
        epsilon : float
            Small constant for numerical stability (default: 1e-10)
        """
        self.n_bins = n_bins
        self.return_range = return_range
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.epsilon = epsilon

        # Create bin edges
        self.bin_edges = np.linspace(return_range[0], return_range[1], n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def discretize_returns(self, returns):
        """
        Discretize returns into histogram bins

        Parameters:
        -----------
        returns : array-like
            Intraday returns for a single day

        Returns:
        --------
        pdf : numpy array
            Probability density function (normalized histogram)
        """
        hist, _ = np.histogram(returns, bins=self.bin_edges, density=True)
        # Normalize to ensure sum = 1
        pdf = hist / np.sum(hist)
        return pdf

    def compute_kl_divergence(self, p, q):
        """
        Compute Kullback-Leibler divergence D(p || q)

        Parameters:
        -----------
        p : numpy array
            True distribution
        q : numpy array
            Reference distribution

        Returns:
        --------
        kl_div : float
            KL divergence D(p || q)
        """
        # Add small constant for numerical stability
        p_safe = p + self.epsilon
        q_safe = np.maximum(q, self.epsilon)

        # Compute KL divergence
        kl_div = np.sum(p_safe * np.log(p_safe / q_safe))
        return kl_div

    def fit_benchmark_distributions(self, pdfs_window):
        """
        Fit cluster-based benchmark distributions using K-means

        Parameters:
        -----------
        pdfs_window : list of numpy arrays
            PDFs for all days in the window

        Returns:
        --------
        benchmarks : dict
            Dictionary mapping cluster labels to benchmark distributions
        """
        # Convert to matrix for clustering
        pdf_matrix = np.array(pdfs_window)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pdf_matrix)

        # Compute cluster centroids as benchmarks
        benchmarks = {}
        for k in range(self.n_clusters):
            cluster_pdfs = pdf_matrix[labels == k]
            benchmark = np.mean(cluster_pdfs, axis=0)
            benchmarks[k] = benchmark

        return benchmarks, kmeans

    def select_benchmark(self, out_of_sample_pdf, benchmarks):
        """
        Select the best benchmark distribution by minimizing KL divergence

        Parameters:
        -----------
        out_of_sample_pdf : numpy array
            PDF for the out-of-sample day
        benchmarks : dict
            Dictionary of benchmark distributions

        Returns:
        --------
        best_benchmark : numpy array
            Selected benchmark distribution
        """
        min_kl = float('inf')
        best_benchmark = None

        for k, benchmark in benchmarks.items():
            kl_div = self.compute_kl_divergence(out_of_sample_pdf, benchmark)
            if kl_div < min_kl:
                min_kl = kl_div
                best_benchmark = benchmark

        return best_benchmark

    def compute_ambiguity_for_stock(self, intraday_returns):
        """
        Compute A_CEA_t for a single stock over time

        Parameters:
        -----------
        intraday_returns : pandas Series or DataFrame
            Minute-level returns with DatetimeIndex

        Returns:
        --------
        ambiguity_series : pandas Series
            A_CEA_t values indexed by date
        """
        # Group by date and compute daily PDFs
        daily_pdfs = {}
        for date, group in intraday_returns.groupby(intraday_returns.index.date):
            daily_pdfs[date] = self.discretize_returns(group.values)

        # Convert to sorted list
        dates = sorted(daily_pdfs.keys())
        pdfs = [daily_pdfs[date] for date in dates]

        # Initialize storage
        ambiguity_values = []
        benchmark = None

        # Rolling window analysis
        for i in range(len(dates)):
            if i < self.window_size:
                # Use first window's PDF as initial benchmark
                if i == 0:
                    window_pdfs = pdfs[:self.window_size]
                    benchmarks, kmeans = self.fit_benchmark_distributions(window_pdfs)
                    # Use first cluster as initial benchmark
                    benchmark = benchmarks[0]

                # For early days, use simple KL from initial benchmark
                kl_div = self.compute_kl_divergence(pdfs[i], benchmark)
                ambiguity_values.append(kl_div)

            elif i % self.window_size == 0:
                # Window boundary: update benchmark
                window_start = i - self.window_size
                window_end = i
                window_pdfs = pdfs[window_start:window_end]

                # Fit new benchmarks
                benchmarks, kmeans = self.fit_benchmark_distributions(window_pdfs)

                # Select best benchmark using out-of-sample day
                if i + 1 < len(pdfs):
                    out_of_sample = pdfs[i + 1]
                    benchmark = self.select_benchmark(out_of_sample, benchmarks)

                # Compute ambiguity for current day
                kl_div = self.compute_kl_divergence(pdfs[i], benchmark)
                ambiguity_values.append(kl_div)

            else:
                # Within window: use current benchmark
                kl_div = self.compute_kl_divergence(pdfs[i], benchmark)
                ambiguity_values.append(kl_div)

        # Create Series
        ambiguity_series = pd.Series(ambiguity_values, index=dates)
        return ambiguity_series

    def compute_ambiguity_cross_section(self, returns_data):
        """
        Compute A_CEA_t for multiple stocks in cross-section

        Parameters:
        -----------
        returns_data : pandas DataFrame
            DataFrame with columns for each stock and DatetimeIndex

        Returns:
        --------
        ambiguity_df : pandas DataFrame
            DataFrame of A_CEA_t values for all stocks
        """
        ambiguity_dict = {}

        for stock_id in returns_data.columns:
            stock_returns = returns_data[stock_id].dropna()
            if len(stock_returns) > self.window_size:
                ambiguity_series = self.compute_ambiguity_for_stock(stock_returns)
                ambiguity_dict[stock_id] = ambiguity_series

        ambiguity_df = pd.DataFrame(ambiguity_dict)
        return ambiguity_df


def compute_peer_ambiguity(ambiguity_df, industry_mapping):
    """
    Compute peer-based ambiguity as instrumental variable

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        A_CEA_t values for all stocks
    industry_mapping : dict
        Mapping from stock_id to industry

    Returns:
    --------
    peer_ambiguity_df : pandas DataFrame
        Leave-one-out industry average ambiguity
    """
    peer_ambiguity_df = ambiguity_df.copy()

    for date in ambiguity_df.index:
        for stock in ambiguity_df.columns:
            if stock in industry_mapping:
                industry = industry_mapping[stock]
                # Get all stocks in same industry
                industry_stocks = [s for s in ambiguity_df.columns
                                 if s in industry_mapping and industry_mapping[s] == industry]
                # Compute leave-one-out average
                peer_stocks = [s for s in industry_stocks if s != stock]
                if len(peer_stocks) > 0:
                    peer_ambiguity_df.loc[date, stock] = ambiguity_df.loc[date, peer_stocks].mean()

    return peer_ambiguity_df


def compute_control_variables(returns_data):
    """
    Compute control variables: RV, Skewness, Kurtosis, Turnover

    Parameters:
    -----------
    returns_data : pandas DataFrame
        Minute-level returns for all stocks

    Returns:
    --------
    controls : dict of pandas DataFrames
        Dictionary containing RV, Skewness, Kurtosis, Turnover Rate
    """
    controls = {}

    # Resample to daily frequency
    daily_returns = returns_data.resample('D').apply(
        lambda x: np.log(x.iloc[-1] / x.iloc[0]) if len(x) > 0 else np.nan
    )

    # Realized Volatility
    controls['RV'] = returns_data.resample('D').apply(
        lambda x: np.sqrt(np.mean(x**2)) if len(x) > 0 else np.nan
    )

    # Skewness
    controls['Skewness'] = returns_data.resample('D').apply(
        lambda x: x.skew() if len(x) > 0 else np.nan
    )

    # Kurtosis
    controls['Kurtosis'] = returns_data.resample('D').apply(
        lambda x: x.kurtosis() if len(x) > 0 else np.nan
    )

    # Turnover Rate (requires volume data - placeholder)
    controls['Turnover'] = returns_data.resample('D').apply(
        lambda x: len(x) if len(x) > 0 else np.nan  # Placeholder
    )

    return controls


if __name__ == "__main__":
    # Example usage
    print("Ambiguity Measurement Module")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-05-24', freq='B')
    n_stocks = 10
    n_minutes_per_day = 240

    # Create synthetic intraday returns
    sample_data = {}
    for stock in range(n_stocks):
        stock_returns = []
        for date in dates:
            daily_minutes = np.random.normal(0, 0.001, n_minutes_per_day)
            stock_returns.extend(daily_minutes)
        sample_data[f'Stock_{stock}'] = stock_returns

    # Create DataFrame
    index = pd.date_range(dates[0], periods=len(dates) * n_minutes_per_day, freq='1min')
    returns_df = pd.DataFrame(sample_data, index=index)

    # Compute ambiguity
    ambiguity_measure = AmbiguityMeasurement()
    ambiguity_df = ambiguity_measure.compute_ambiguity_cross_section(returns_df)

    print(f"\nComputed ambiguity for {len(ambiguity_df.columns)} stocks")
    print(f"Date range: {ambiguity_df.index[0]} to {ambiguity_df.index[-1]}")
    print(f"\nSample statistics:")
    print(ambiguity_df.describe())
