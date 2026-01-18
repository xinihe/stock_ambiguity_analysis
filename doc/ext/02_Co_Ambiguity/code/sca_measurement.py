"""
Systemic Co-Ambiguity (SCA) Measurement Module
Implements the SCA index construction and related computations
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class SystemicCoAmbiguity:
    """
    Compute Systemic Co-Ambiguity (SCA) index measuring synchronization
    of uncertainty (ambiguity) across assets in the market
    """

    def __init__(self, corr_window=60, weighted=False):
        """
        Initialize the SCA calculator

        Parameters:
        -----------
        corr_window : int
            Rolling window for computing pairwise correlations (default: 60 days)
        weighted : bool
            Whether to compute market-cap-weighted SCA (default: False)
        """
        self.corr_window = corr_window
        self.weighted = weighted

    def compute_sca(self, ambiguity_df, market_caps=None):
        """
        Compute Systemic Co-Ambiguity Index

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            Individual ambiguity indices (stocks × dates)
            Columns: stock identifiers, Index: dates
        market_caps : pandas DataFrame or None
            Market capitalizations (stocks × dates), required if weighted=True

        Returns:
        --------
        sca_series : pandas Series
            SCA time series indexed by date
        """
        dates = ambiguity_df.index
        sca_values = []

        for i, date in enumerate(dates):
            # Get historical window
            start_date = max(dates[0], date - pd.Timedelta(days=self.corr_window))
            window_data = ambiguity_df.loc[start_date:date]

            # Skip if insufficient data
            if len(window_data) < self.corr_window // 2:
                sca_values.append(np.nan)
                continue

            # Compute pairwise correlation matrix
            corr_matrix = window_data.corr()

            # Handle NaN values
            corr_matrix = corr_matrix.fillna(0)

            if self.weighted:
                if market_caps is None:
                    raise ValueError("market_caps required for weighted SCA")
                # Get current market caps
                current_caps = market_caps.loc[date]
                # Normalize weights
                weights = current_caps / current_caps.sum()
                # Compute weighted SCA
                sca = self._compute_weighted_sca(corr_matrix, weights)
            else:
                # Compute unweighted SCA
                sca = self._compute_unweighted_sca(corr_matrix)

            sca_values.append(sca)

        sca_series = pd.Series(sca_values, index=dates)
        return sca_series

    def _compute_unweighted_sca(self, corr_matrix):
        """
        Compute unweighted SCA as average pairwise correlation

        Formula: SCA = (2 / (N*(N-1))) * sum_{i<j} Corr(A_i, A_j)

        Parameters:
        -----------
        corr_matrix : pandas DataFrame
            N×N correlation matrix of ambiguity indices

        Returns:
        --------
        sca : float
            Systemic Co-Ambiguity value
        """
        n = corr_matrix.shape[0]

        # Extract upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices(n, k=1)
        correlations = corr_matrix.values[upper_tri_indices]

        # Compute average
        if len(correlations) > 0:
            sca = np.mean(correlations)
        else:
            sca = np.nan

        return sca

    def _compute_weighted_sca(self, corr_matrix, weights):
        """
        Compute market-cap-weighted SCA

        Formula: SCA_weighted = sum_{i≠j} w_i * w_j * Corr(A_i, A_j)

        Parameters:
        -----------
        corr_matrix : pandas DataFrame
            N×N correlation matrix of ambiguity indices
        weights : pandas Series
            Market cap weights for each stock

        Returns:
        --------
        sca : float
            Weighted Systemic Co-Ambiguity value
        """
        # Convert to numpy arrays
        corr = corr_matrix.values
        w = weights.values.reshape(-1, 1)

        # Compute weighted sum: w' * R * w - sum(w_i^2)
        # Subtract diagonal elements (self-correlation = 1)
        sca = w.T @ corr @ w - np.sum(w**2)
        sca = sca[0, 0]  # Extract scalar

        return sca

    def compute_sca_efficient(self, ambiguity_df, market_caps=None):
        """
        Efficient vectorized computation of SCA

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            Individual ambiguity indices (stocks × dates)
        market_caps : pandas DataFrame or None
            Market capitalizations (stocks × dates), required if weighted=True

        Returns:
        --------
        sca_series : pandas Series
            SCA time series indexed by date
        """
        dates = ambiguity_df.index
        sca_values = []

        # Pre-compute standardized ambiguity series
        ambiguity_std = ambiguity_df.rolling(
            window=self.corr_window, min_periods=self.corr_window//2
        ).apply(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
        )

        for date in dates:
            # Get current standardized values
            current_std = ambiguity_std.loc[date]

            # Skip if insufficient data
            if current_std.isna().all():
                sca_values.append(np.nan)
                continue

            # Compute correlation matrix using outer product
            # For standardized series, corr = (1/T) * sum(r_i * r_j)
            valid_stocks = current_std.dropna().index
            n_valid = len(valid_stocks)

            if n_valid < 2:
                sca_values.append(np.nan)
                continue

            # Extract valid standardized values
            r = current_std.loc[valid_stocks].values.reshape(-1, 1)

            # Compute correlation matrix estimate
            corr_estimate = r @ r.T / n_valid

            if self.weighted:
                if market_caps is None:
                    raise ValueError("market_caps required for weighted SCA")
                current_caps = market_caps.loc[date, valid_stocks]
                weights = current_caps / current_caps.sum()
                sca = self._compute_weighted_sca(
                    pd.DataFrame(corr_estimate, index=valid_stocks, columns=valid_stocks),
                    weights
                )
            else:
                sca = self._compute_unweighted_sca(
                    pd.DataFrame(corr_estimate, index=valid_stocks, columns=valid_stocks)
                )

            sca_values.append(sca)

        sca_series = pd.Series(sca_values, index=dates)
        return sca_series

    def compute_network_metrics(self, ambiguity_df):
        """
        Compute additional network-based co-ambiguity metrics

        Parameters:
        -----------
        ambiguity_df : pandas DataFrame
            Individual ambiguity indices (stocks × dates)

        Returns:
        --------
        metrics : dict of pandas Series
            Dictionary containing network metrics
        """
        dates = ambiguity_df.index
        metrics = {
            'eigenvalue_centrality': [],
            'network_density': [],
            'clustering_coefficient': []
        }

        for date in dates:
            # Get correlation matrix
            start_date = max(dates[0], date - pd.Timedelta(days=self.corr_window))
            window_data = ambiguity_df.loc[start_date:date]

            if len(window_data) < self.corr_window // 2:
                for key in metrics:
                    metrics[key].append(np.nan)
                continue

            corr_matrix = window_data.corr().fillna(0)

            # Eigenvalue centrality (largest eigenvalue)
            eigenvalues = np.linalg.eigvals(corr_matrix.values)
            max_eigenvalue = np.max(np.real(eigenvalues))
            metrics['eigenvalue_centrality'].append(max_eigenvalue)

            # Network density (average absolute correlation)
            density = np.mean(np.abs(corr_matrix.values))
            metrics['network_density'].append(density)

            # Clustering coefficient (simplified)
            # Average correlation of correlations
            triu_indices = np.triu_indices(corr_matrix.shape[0], k=1)
            triu_values = corr_matrix.values[triu_indices]
            clustering = np.mean(triu_values**2) if len(triu_values) > 0 else np.nan
            metrics['clustering_coefficient'].append(clustering)

        # Convert to Series
        for key in metrics:
            metrics[key] = pd.Series(metrics[key], index=dates)

        return metrics


def compute_cross_market_sca(ambiguity_dict, market_names=None):
    """
    Compute SCA across different markets/asset classes

    Parameters:
    -----------
    ambiguity_dict : dict of pandas DataFrames
        Dictionary mapping market names to ambiguity DataFrames
    market_names : list or None
        Names of markets (uses keys if None)

    Returns:
    --------
    cross_market_sca : pandas DataFrame
        Cross-market SCA matrix (markets × dates)
    """
    if market_names is None:
        market_names = list(ambiguity_dict.keys())

    dates = ambiguity_dict[market_names[0]].index
    cross_market_sca = pd.DataFrame(index=dates, columns=market_names)

    # Compute market-level ambiguity (average across stocks)
    market_ambiguity = {}
    for market in market_names:
        market_ambiguity[market] = ambiguity_dict[market].mean(axis=1)

    # Convert to DataFrame
    market_ambiguity_df = pd.DataFrame(market_ambiguity)

    # Compute pairwise correlations
    for date in dates:
        # Get historical window
        start_date = max(dates[0], date - pd.Timedelta(days=60))
        window_data = market_ambiguity_df.loc[start_date:date]

        if len(window_data) < 30:
            cross_market_sca.loc[date] = np.nan
            continue

        # Compute correlation matrix
        corr_matrix = window_data.corr()

        # Store diagonal (auto-correlation) as market-specific SCA
        for market in market_names:
            cross_market_sca.loc[date, market] = corr_matrix.loc[market, market]

    return cross_market_sca


if __name__ == "__main__":
    # Example usage
    print("Systemic Co-Ambiguity Measurement Module")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-05-24', freq='D')
    n_stocks = 300

    # Generate ambiguity data with time-varying correlation
    # Normal periods: low correlation
    # Crisis periods: high correlation
    ambiguity_data = np.random.randn(len(dates), n_stocks) * 0.1

    # Add correlated shocks during crisis periods
    crisis_periods = [
        ('2018-02-01', '2018-04-01'),
        ('2020-02-01', '2020-04-01'),
        ('2022-02-01', '2022-04-01')
    ]

    for start, end in crisis_periods:
        mask = (dates >= start) & (dates <= end)
        common_shock = np.random.randn(len(dates[mask])) * 0.3
        ambiguity_data[mask] += common_shock.reshape(-1, 1)

    ambiguity_df = pd.DataFrame(
        ambiguity_data,
        index=dates,
        columns=[f'Stock_{i}' for i in range(n_stocks)]
    )

    # Add market caps
    market_caps = pd.DataFrame(
        np.random.rand(len(dates), n_stocks) * 1e10 + 1e9,
        index=dates,
        columns=[f'Stock_{i}' for i in range(n_stocks)]
    )

    # Compute SCA
    print("\nComputing unweighted SCA...")
    sca_calculator = SystemicCoAmbiguity(corr_window=60, weighted=False)
    sca_unweighted = sca_calculator.compute_sca_efficient(ambiguity_df)
    print(f"Mean SCA: {sca_unweighted.mean():.6f}")
    print(f"Std SCA: {sca_unweighted.std():.6f}")

    print("\nComputing weighted SCA...")
    sca_calculator_w = SystemicCoAmbiguity(corr_window=60, weighted=True)
    sca_weighted = sca_calculator_w.compute_sca_efficient(ambiguity_df, market_caps)
    print(f"Mean Weighted SCA: {sca_weighted.mean():.6f}")
    print(f"Std Weighted SCA: {sca_weighted.std():.6f}")

    print("\nComputing network metrics...")
    network_metrics = sca_calculator.compute_network_metrics(ambiguity_df)
    print(f"Mean Eigenvalue Centrality: {network_metrics['eigenvalue_centrality'].mean():.6f}")
    print(f"Mean Network Density: {network_metrics['network_density'].mean():.6f}")

    print("\n" + "=" * 60)
    print("Computation complete!")
