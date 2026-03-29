"""
Ambiguity Measurement Module for Chinese Energy Market
Implements Cross-Entropy Ambiguity (CEA) Index with PCA-based Composite Measures
Designed for the paper: "Pricing the Unknown: Ambiguity Premiums in China's Green vs. Brown Energy Markets"

This module computes multiple levels of ambiguity measures:
1. Firm-level ambiguity (CEA_i,t)
2. Sector ambiguity (Brown vs. Green energy)
3. Composite Energy Ambiguity (PCA-based systematic factor)
4. Policy ambiguity (from energy indices)
5. Geopolitical ambiguity (from defense stocks and gold futures)

Author: Research Team
Date: 2024
Paper Reference: causal_ambi_china.tex
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class AmbiguityMeasurement:
    """
    Compute Cross-Entropy Ambiguity (A_CEA_t) index for Chinese A-share energy stocks

    Based on Hansen and Sargent's (2001) multiplier-preference model, this class quantifies
    model uncertainty using Kullback-Leibler divergence between empirical intraday return
    distributions and adaptive benchmark distributions.

    Key Features:
    - Handles Chinese A-share market structure (limit days, price limits)
    - Supports multiple ambiguity measures (firm, sector, composite, policy, geopolitical)
    - Implements PCA-based systematic ambiguity extraction
    - Optimized for energy sector classification (Brown vs. Green)
    """

    def __init__(self, n_bins=202, return_range=(-0.201, 0.201),
                 window_size=20, n_clusters=4, epsilon=1e-10):
        """
        Initialize the ambiguity measurement

        Parameters:
        -----------
        n_bins : int
            Number of bins for discretizing returns (default: 202 as per paper)
        return_range : tuple
            Range for binning returns covering 99.7% of intraday returns
        window_size : int
            Size of rolling window for benchmark selection (default: 20 trading days)
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

        Implements Equation from paper:
        q_{i,t}(x_j) = (1/N_{i,t}) * sum(1(r_{i,t,k} in bin_j))

        Parameters:
        -----------
        returns : array-like
            Intraday returns for a single day (1-minute or 5-minute data)

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

        Implements: D_KL(p||q) = sum(p * log(p/q))

        Parameters:
        -----------
        p : numpy array
            True distribution (empirical)
        q : numpy array
            Reference distribution (benchmark)

        Returns:
        --------
        kl_div : float
            KL divergence measuring information loss
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
        kmeans : KMeans object
            Fitted K-means model for benchmark selection
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

        Implements: P_{w+1} = argmin_k D_KL(q_{i,Ww+1} || p_{w,k})

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

    def compute_ambiguity_for_stock(self, intraday_returns, limit_days=None):
        """
        Compute A_CEA_t for a single stock over time

        Implements the dynamic algorithm from Section 3.2 of causal_ambi_china.tex

        Parameters:
        -----------
        intraday_returns : pandas Series or DataFrame
            Minute-level returns with DatetimeIndex
        limit_days : list of dates or None
            Dates with limit-up/limit-down movements to exclude (for Chinese A-share)

        Returns:
        --------
        ambiguity_series : pandas Series
            A_CEA_t values indexed by date
        """
        # Group by date and compute daily PDFs
        daily_pdfs = {}
        for date, group in intraday_returns.groupby(intraday_returns.index.date):
            # Skip limit days if specified
            if limit_days and date in limit_days:
                continue
            if len(group) > 10:  # Require minimum observations
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
                # Use first window's PDFs as initial benchmarks
                if i == 0 and len(pdfs) >= self.window_size:
                    window_pdfs = pdfs[:self.window_size]
                    benchmarks, kmeans = self.fit_benchmark_distributions(window_pdfs)
                    # Use first cluster as initial benchmark
                    benchmark = benchmarks[0]

                # For early days, use simple KL from initial benchmark
                if benchmark is not None:
                    kl_div = self.compute_kl_divergence(pdfs[i], benchmark)
                    ambiguity_values.append(kl_div)
                else:
                    ambiguity_values.append(0.0)

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
                else:
                    benchmark = benchmarks[0]

                # Compute ambiguity for current day
                kl_div = self.compute_kl_divergence(pdfs[i], benchmark)
                ambiguity_values.append(kl_div)

            else:
                # Within window: use current benchmark
                if benchmark is not None:
                    kl_div = self.compute_kl_divergence(pdfs[i], benchmark)
                    ambiguity_values.append(kl_div)
                else:
                    ambiguity_values.append(0.0)

        # Create Series
        ambiguity_series = pd.Series(ambiguity_values, index=dates)
        return ambiguity_series

    def compute_ambiguity_cross_section(self, returns_data, limit_days_dict=None):
        """
        Compute A_CEA_t for multiple stocks in cross-section

        Parameters:
        -----------
        returns_data : pandas DataFrame
            DataFrame with columns for each stock and DatetimeIndex
        limit_days_dict : dict or None
            Dictionary mapping stock_id to list of limit days

        Returns:
        --------
        ambiguity_df : pandas DataFrame
            DataFrame of A_CEA_t values for all stocks
        """
        ambiguity_dict = {}

        for stock_id in returns_data.columns:
            stock_returns = returns_data[stock_id].dropna()
            limit_days = limit_days_dict.get(stock_id, None) if limit_days_dict else None

            if len(stock_returns) > self.window_size:
                ambiguity_series = self.compute_ambiguity_for_stock(stock_returns, limit_days)
                if len(ambiguity_series) > 0:
                    ambiguity_dict[stock_id] = ambiguity_series

        ambiguity_df = pd.DataFrame(ambiguity_dict)
        return ambiguity_df


class EnergySectorAmbiguity:
    """
    Compute sector-level and composite ambiguity measures for Chinese energy market

    This class implements the hierarchy of CEA measures from Section 3.3 of causal_ambi_china.tex:
    - Firm-level ambiguity (CEA_i,t)
    - Sector ambiguity (Brown vs. Green)
    - Composite Energy Ambiguity (PCA-based)
    - Policy ambiguity (from energy indices)
    - Geopolitical ambiguity (from defense stocks and gold)
    """

    def __init__(self, energy_classification, market_cap_data=None):
        """
        Initialize energy sector ambiguity calculator

        Parameters:
        -----------
        energy_classification : dict
            Dictionary mapping stock_id to 'Brown', 'Green', or 'Grey'
        market_cap_data : pandas DataFrame or None
            Market capitalization data for value-weighting
        """
        self.classification = energy_classification
        self.market_cap_data = market_cap_data

    def compute_sector_ambiguity(self, ambiguity_df):
        """
        Compute value-weighted sector ambiguity measures

        Implements Equation:
        CEA_{Brown,t} = sum_{i in Brown} w_{i,t} * CEA_{i,t}
        CEA_{Green,t} = sum_{i in Green} w_{i,t} * CEA_{i,t}

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            Firm-level CEA values (stocks × dates)

        Returns:
        --------
        sector_ambiguity : dict
            Dictionary with 'Brown' and 'Green' sector ambiguity series
        """
        sector_ambiguity = {}

        # Get stocks by sector
        brown_stocks = [s for s in ambiguity_df.columns if self.classification.get(s) == 'Brown']
        green_stocks = [s for s in ambiguity_df.columns if self.classification.get(s) == 'Green']

        # Compute sector ambiguity
        for sector_name, stocks in [('Brown', brown_stocks), ('Green', green_stocks)]:
            if len(stocks) > 0:
                sector_data = ambiguity_df[stocks]

                if self.market_cap_data is not None:
                    # Value-weighted
                    weights = self.market_cap_data[stocks].div(
                        self.market_cap_data[stocks].sum(axis=1), axis=0
                    )
                    sector_ambiguity[sector_name] = (sector_data * weights).sum(axis=1)
                else:
                    # Equal-weighted
                    sector_ambiguity[sector_name] = sector_data.mean(axis=1)

        return sector_ambiguity

    def compute_composite_ambiguity(self, ambiguity_df, n_components=1):
        """
        Compute Composite Energy Ambiguity using PCA

        Implements Equation:
        CEA_{Composite,t} = sum_{i=1}^{N} lambda_i * CEA_{i,t}

        This extracts the common component of model uncertainty across the industry,
        distinguishing systematic ambiguity from firm-specific idiosyncratic ambiguity.

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            Firm-level CEA values (stocks × dates)
        n_components : int
            Number of principal components (default: 1 for first PC)

        Returns:
        --------
        composite_ambiguity : pandas Series
            Composite Energy Ambiguity index
        pca_loadings : pandas Series
            Principal component loadings for each stock
        """
        # Fill missing values with forward fill then backward fill
        ambiguity_filled = ambiguity_df.fillna(method='ffill').fillna(method='bfill')

        # Standardize (z-score) before PCA
        ambiguity_standardized = (
            (ambiguity_filled - ambiguity_filled.mean()) / ambiguity_filled.std()
        )

        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(ambiguity_standardized.T)  # Transpose: stocks as features, dates as observations

        # Get first principal component scores (this gives us the composite ambiguity over time)
        composite_scores = pca.transform(ambiguity_standardized.T)[:, 0]

        # Create Series
        composite_ambiguity = pd.Series(composite_scores, index=ambiguity_df.index)

        # Get loadings
        pca_loadings = pd.Series(pca.components_[0], index=ambiguity_df.columns)

        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_[0]

        return composite_ambiguity, pca_loadings, explained_variance

    def compute_policy_ambiguity(self, index_returns, ambiguity_measure=None):
        """
        Compute Policy Ambiguity from energy index returns

        Uses CEA on CSI 300 Energy Index ETF or Shanghai Crude Oil Futures (INE SC)

        Parameters:
        -----------
        index_returns : pandas Series
            Intraday returns for energy index
        ambiguity_measure : AmbiguityMeasurement or None
            Ambiguity measurement object (creates new if None)

        Returns:
        --------
        policy_ambiguity : pandas Series
            Policy ambiguity time series
        """
        if ambiguity_measure is None:
            ambiguity_measure = AmbiguityMeasurement()

        policy_ambiguity = ambiguity_measure.compute_ambiguity_for_stock(index_returns)

        return policy_ambiguity

    def compute_geopolitical_ambiguity(self, defense_returns, gold_returns,
                                      ambiguity_measure=None):
        """
        Compute Geopolitical Ambiguity from defense stocks and gold futures

        Implements Equation:
        GeoAmbiguity_t = PC1(Defense CEA_t, Gold CEA_t)

        Parameters:
        -----------
        defense_returns : pandas Series
            Intraday returns for CSI National Defense Index
        gold_returns : pandas Series
            Intraday returns for SHFE Gold futures
        ambiguity_measure : AmbiguityMeasurement or None
            Ambiguity measurement object (creates new if None)

        Returns:
        --------
        geo_ambiguity : pandas Series
            Geopolitical ambiguity time series (first PC)
        """
        if ambiguity_measure is None:
            ambiguity_measure = AmbiguityMeasurement()

        # Compute CEA for defense and gold
        defense_cea = ambiguity_measure.compute_ambiguity_for_stock(defense_returns)
        gold_cea = ambiguity_measure.compute_ambiguity_for_stock(gold_returns)

        # Align dates
        geo_data = pd.DataFrame({
            'Defense': defense_cea,
            'Gold': gold_cea
        }).dropna()

        if len(geo_data) > 0:
            # Take first principal component
            pca = PCA(n_components=1)
            geo_ambiguity_values = pca.fit_transform(geo_data.values).flatten()
            geo_ambiguity = pd.Series(geo_ambiguity_values, index=geo_data.index)
            explained_variance = pca.explained_variance_ratio_[0]
        else:
            geo_ambiguity = pd.Series(dtype=float)
            explained_variance = 0.0

        return geo_ambiguity, explained_variance


def compute_peer_ambiguity(ambiguity_df, sector_mapping):
    """
    Compute peer-based ambiguity as instrumental variable

    Implements leave-one-out industry average:
    PeerAmbiguity_{i,t} = (1/(N_sector(i)-1)) * sum_{j in sector(i), j≠i} CEA_{j,t}

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        CEA_t values for all stocks
    sector_mapping : dict
        Mapping from stock_id to sector/industry

    Returns:
    --------
    peer_ambiguity_df : pandas DataFrame
        Leave-one-out sector average ambiguity
    """
    peer_ambiguity_df = ambiguity_df.copy()

    for date in ambiguity_df.index:
        for stock in ambiguity_df.columns:
            if stock in sector_mapping:
                sector = sector_mapping[stock]
                # Get all stocks in same sector
                sector_stocks = [
                    s for s in ambiguity_df.columns
                    if s in sector_mapping and sector_mapping[s] == sector
                ]
                # Compute leave-one-out average
                peer_stocks = [s for s in sector_stocks if s != stock]
                if len(peer_stocks) > 0:
                    peer_ambiguity_df.loc[date, stock] = ambiguity_df.loc[date, peer_stocks].mean()

    return peer_ambiguity_df


def compute_control_variables(returns_data, volume_data=None, bid_ask_data=None):
    """
    Compute control variables: RV, Skewness, Kurtosis, Turnover, Spread

    Parameters:
    -----------
    returns_data : pandas DataFrame
        Minute-level returns for all stocks
    volume_data : pandas DataFrame or None
        Trading volume data (for turnover rate)
    bid_ask_data : dict or None
        Dictionary with 'bid' and 'ask' DataFrames

    Returns:
    --------
    controls : dict of pandas DataFrames
        Dictionary containing RV, Skewness, Kurtosis, Turnover Rate, Spread
    """
    controls = {}

    # Resample to daily frequency for all calculations
    # Realized Volatility (RV)
    controls['RV'] = returns_data.resample('D').apply(
        lambda x: np.sqrt(np.mean(x**2)) if len(x) > 0 else np.nan
    )

    # Skewness (third moment)
    controls['Skewness'] = returns_data.resample('D').apply(
        lambda x: x.skew() if len(x) > 0 else np.nan
    )

    # Kurtosis (fourth moment minus 3 for excess kurtosis)
    controls['Kurtosis'] = returns_data.resample('D').apply(
        lambda x: x.kurtosis() if len(x) > 0 else np.nan
    )

    # Turnover Rate
    if volume_data is not None:
        # This would use actual volume and shares outstanding
        # For now, use sum of absolute returns as proxy
        controls['Turnover'] = returns_data.resample('D').apply(
            lambda x: np.sum(np.abs(x)) if len(x) > 0 else np.nan
        )
    else:
        controls['Turnover'] = returns_data.resample('D').apply(
            lambda x: np.sum(np.abs(x)) if len(x) > 0 else np.nan
        )

    # Bid-Ask Spread (if data provided)
    if bid_ask_data is not None:
        bid = bid_ask_data['bid'].resample('D').mean()
        ask = bid_ask_data['ask'].resample('D').mean()
        controls['Spread'] = ((ask - bid) / ((ask + bid) / 2))
    else:
        # Placeholder - in practice you'd need actual bid-ask data
        controls['Spread'] = returns_data.resample('D').apply(
            lambda x: 0.001 if len(x) > 0 else np.nan  # Placeholder
        )

    return controls


if __name__ == "__main__":
    # Example usage for Chinese energy market
    print("Ambiguity Measurement Module for Chinese Energy Market")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-05-24', freq='B')
    n_stocks = 20
    n_minutes_per_day = 240

    # Create synthetic intraday returns for energy stocks
    sample_data = {}
    energy_classification = {}
    for i in range(n_stocks):
        stock_name = f'Energy_{i}'
        stock_returns = []
        for date in dates:
            daily_minutes = np.random.normal(0, 0.001, n_minutes_per_day)
            stock_returns.extend(daily_minutes)
        sample_data[stock_name] = stock_returns

        # Classify as Brown or Green
        energy_classification[stock_name] = 'Brown' if i < 10 else 'Green'

    # Create DataFrame
    index = pd.date_range(dates[0], periods=len(dates) * n_minutes_per_day, freq='1min')
    returns_df = pd.DataFrame(sample_data, index=index)

    # Compute firm-level ambiguity
    print("\nComputing firm-level ambiguity...")
    ambiguity_measure = AmbiguityMeasurement()
    ambiguity_df = ambiguity_measure.compute_ambiguity_cross_section(returns_df)

    print(f"Computed ambiguity for {len(ambiguity_df.columns)} stocks")
    print(f"Date range: {ambiguity_df.index[0]} to {ambiguity_df.index[-1]}")

    # Compute sector ambiguity
    print("\nComputing sector ambiguity...")
    sector_ambiguity = EnergySectorAmbiguity(energy_classification)
    sector_results = sector_ambiguity.compute_sector_ambiguity(ambiguity_df)

    print(f"Brown Energy Ambiguity: mean = {sector_results['Brown'].mean():.6f}")
    print(f"Green Energy Ambiguity: mean = {sector_results['Green'].mean():.6f}")

    # Compute composite ambiguity
    print("\nComputing composite energy ambiguity (PCA)...")
    composite_ambiguity, loadings, explained_var = sector_ambiguity.compute_composite_ambiguity(ambiguity_df)

    print(f"Composite Energy Ambiguity: mean = {composite_ambiguity.mean():.6f}")
    print(f"First PC explains: {explained_var:.2%} of variance")

    print("\n" + "=" * 70)
    print("Example usage complete!")
