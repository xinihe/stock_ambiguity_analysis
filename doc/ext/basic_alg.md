# Basic Algorithm: Cross-Entropy Ambiguity (CEA) Index

This document outlines the basic idea and algorithm for measuring ambiguity in stock returns, denoted as $\mathcal{A}^{CEA}_t$. The measure is based on the Kullback-Leibler (KL) divergence within a multiplier-preference framework.

## 1. Core Concept

The algorithm quantifies **ambiguity** as the divergence between the *observed* daily return distribution (empirical data) and a *benchmark* distribution (expected model).
*   **Risk**: Uncertainty about outcomes given a known probability distribution.
*   **Ambiguity**: Uncertainty about the probability distribution itself (model uncertainty).

The measure captures how much the daily return distribution deviates from a "best estimate" benchmark, reflecting informational instability.

## 2. Data Preparation

*   **Input**: High-frequency (e.g., one-minute) trading data for stock $s$ on day $t$.
*   **Discretization**:
    *   The daily range of returns is discretized into **202 equally spaced bins** covering the interval $[-0.201, +0.201]$.
    *   For each day $t$, a probability vector (distribution) $q_t$ is constructed by calculating the frequency of minute-level returns falling into each bin.
    *   Zero probabilities are handled by adding a small constant $\epsilon = 10^{-10}$ to avoid numerical instability in logarithmic calculations.

## 3. Algorithm Steps

The algorithm operates in a rolling window fashion, processing data in segments (e.g., 20-day windows).

### Step 1: Segmentation and Clustering (Training)

1.  **Partition**: The full time series of trading days is partitioned into consecutive segments (e.g., 20 days per segment).
2.  **Clustering**: Within a segment $j$, the daily return distributions $\{q_d\}_{d \in \text{segment } j}$ are grouped into **4 classes** (regimes) based on distributional similarity (using K-means clustering).
3.  **Representative Distributions**: For each class $k \in \{1, 2, 3, 4\}$, compute the representative distribution (centroid) $p_{j,k}$ as the average of all distributions in that class:
    $$
    p_{j,k} = \frac{1}{N_{j,k}} \sum_{\ell \in \text{class}_{j,k}} q_{\ell}
    $$
    where $N_{j,k}$ is the number of days in class $k$. These $p_{j,k}$ serve as the set of "candidate" benchmark distributions.

### Step 2: Benchmark Selection (Transition)

At the boundary between segment $j$ and the next segment $j+1$, we use the first day of the new segment (day $20j+21$, denoted as $q_{\text{out}}$) to select the most appropriate benchmark.

1.  **Compare**: Calculate the KL divergence between the out-of-sample distribution $q_{\text{out}}$ and each candidate $p_{j,k}$:
    $$
    D_{\mathrm{KL}}(q_{\text{out}} \parallel p_{j,k}) = \sum_{x} q_{\text{out}}(x) \log \frac{q_{\text{out}}(x)}{p_{j,k}(x)}
    $$
2.  **Select**: Choose the candidate that minimizes this divergence to be the **Standard Benchmark Distribution** ($P_{j+1}$) for the next window:
    $$
    P_{j+1} = \arg\min_{k=1,\dots,4} D_{\mathrm{KL}}(q_{\text{out}} \parallel p_{j,k})
    $$
    This step identifies the historical regime that best fits the current market state.

### Step 3: Ambiguity Measurement (Testing)

For each trading day $i$ in the current window (segment $j+1$), the ambiguity index $\mathcal{A}^{CEA}(q_i)$ is calculated as the KL divergence between the day's empirical distribution $q_i$ and the selected benchmark $P_{j+1}$.

$$
\mathcal{A}^{CEA}(q_i) = D_{\mathrm{KL}}(q_i \parallel P_{j+1}) = \sum_{x} q_i(x) \log \frac{q_i(x)}{P_{j+1}(x)}
$$

## 4. Mathematical Summary

The overall formula for ambiguity on day $i$ (belonging to window $j+1$ with boundary day determined by $q_{\text{out}}$) is:

$$
\mathcal{A}^{CEA}(q_i) = \sum_{x} q_i(x) \log \frac{q_i(x)}{\arg\min_{k} \left( \sum_{y} q_{\text{out}}(y) \log \frac{q_{\text{out}}(y)}{p_{j,k}(y)} \right)}
$$

Where:
*   $q_i(x)$ is the probability of return bin $x$ on day $i$.
*   $q_{\text{out}}$ is the distribution of the first day of the window (used for selection).
*   $p_{j,k}$ are the candidate distributions derived from the previous window's clusters.

## 5. Interpretation

*   **$\mathcal{A}^{CEA}_t$**: Represents the "information loss" or "surprise" experienced when modeling the current day's returns using the best-fitting historical benchmark.
*   **High Ambiguity**: Indicates that the current return distribution deviates significantly from historical norms (even the best-fitting ones), suggesting high model uncertainty or a shift in market regime.
*   **Low Ambiguity**: Indicates that the current market behavior is well-explained by existing models (benchmarks).
