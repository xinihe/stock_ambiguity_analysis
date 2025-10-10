# Combined Global Uncertainty Index: Methodology and Implementation

## Rationale for the Combined Global Uncertainty Index

The core objective of this research is to robustly link global systemic uncertainty to the adaptive ambiguity aversion index in the Chinese capital market. Achieving this requires creating a composite index that accurately reflects the full complexity of external shocks, addressing two significant methodological challenges:

1. **Data Frequency Mismatch**
2. **Risk of Multicollinearity**

## Addressing Methodological Challenges

### Data Frequency Mismatch

In our implementation, we have:
- **Daily Climate Risk Data**: Containing physical risk, transition risk, policy risk, and market sentiment risk components
- **Monthly Geopolitical Risk (GPR) Data**: Aggregated by country (China, Hong Kong Special Administrative Region, Japan, and US)

To address this mismatch, we:
1. Aggregate daily climate risk data to monthly frequency
2. Apply a MIDAS-inspired time-weighting scheme to incorporate recency effects
3. Ensure both components are normalized for comparability

### Mitigating Multicollinearity

By constructing a single composite index that combines both risk dimensions, we avoid the multicollinearity issues that would arise from including them separately in the state transition equation for the ambiguity aversion index.

## Implemented Methodology

### Step 1: Data Preparation and Aggregation

1. **Load and Process Raw Data**:
   - Read daily climate risk series and monthly GPR country data
   - Convert date columns to date-time format

2. **Aggregate Climate Risk to Monthly Frequency**:
   - Resample daily climate risk data to monthly frequency using arithmetic mean
   - Create a composite climate risk index by averaging all climate risk components

3. **Create Composite GPR Index**:
   - Average GPR values across all countries to create a global GPR index

### Step 2: Normalization of Indices

Normalize both climate risk and GPR indices to ensure comparability:

- **Standardization (z-score normalization)**:
  ```
  normalized_value = (original_value - mean) / standard_deviation
  ```
- Alternative: **Scaling to mean of 100**:
  ```
  normalized_value = (original_value / mean) * 100
  ```

### Step 3: MIDAS-Inspired Weighting Approach

Implement a time-decaying weighting scheme using a Beta polynomial function:

```
w(k, θ, m) = [(k/m)^(θ₁-1) * ((m-k)/m)^(θ₂-1)] / Σ[(k/m)^(θ₁-1) * ((m-k)/m)^(θ₂-1)]
```

Where:
- `k` is the position in the window (1 to m)
- `m` is the window size (e.g., 12 months)
- `θ` is a parameter vector controlling the shape and decay of weights

This ensures that more recent observations receive higher weights, preserving the informational content of recent shocks.

### Step 4: Constructing the Final Composite Index

The Combined Global Uncertainty Index (CGUI) is calculated as a weighted average of the normalized components:

```
CGUI_t = w_climate * CR_t + w_gpr * GPR_t
```

Where:
- `CR_t` is the normalized climate risk index at time t
- `GPR_t` is the normalized and MIDAS-weighted GPR index at time t
- `w_climate` and `w_gpr` are structural weights summing to 1.0, determined using Correlation-Adjusted Weighting by default

## Implementation Details

The Python implementation (`create_combined_uncertainty_index.py`) follows this methodology with the following key features:

1. Object-oriented design for modularity and extensibility
2. Flexible weighting schemes with Correlation-Adjusted Weighting as the default approach
3. Visualization of the components and final index
4. Comprehensive data validation and error handling
5. Support for multiple weighting methods: Correlation-Adjusted, Inverse Variance, EWMA, PCA-based, equal weighting, and custom weights

## Weighting Methodology

### Correlation-Adjusted Weighting - Recommended Default Approach

This balanced approach combines statistical properties with domain knowledge:

1. **Base Weights**: Start with inverse variance weights to reduce volatility impact
2. **Correlation Adjustment**: Modify weights based on the correlation between components
3. **Balance Targeting**: Move toward more balanced weights when components are highly correlated
4. **Normalization**: Ensure final weights sum to 1.0

This method provides a better balance than PCA (which tends to dominate one component) while still being data-driven.

### Inverse Variance Weighting

Assigns lower weights to more volatile components:

1. **Calculate Variances**: Compute the variance of each normalized component
2. **Inverse Variances**: Use the reciprocal of variance as weights
3. **Normalize**: Scale weights to sum to 1.0

This approach naturally reduces the impact of components with high variability.

### Exponentially Weighted Moving Average (EWMA) Dynamic Weights

Time-varying weights that adapt to changing market conditions:

1. **Rolling Variances**: Calculate 6-month rolling variances for each component
2. **Dynamic Ratios**: Compute time-varying variance ratios
3. **Smoothing**: Apply EWMA to create stable but adaptive weights
4. **Normalization**: Ensure weights sum to 1.0 at each time point

This method provides dynamic adaptability while maintaining reasonable balance.

### Alternative Weighting Schemes

The implementation also supports:

- **Principal Component Analysis (PCA)**: Weights based on variance contribution (can be dominated by one component)
- **Equal Weighting (0.5/0.5)**: Simple but potentially ignores important statistical properties
- **Custom Weights**: User-specified weights based on domain expertise

### Advantages of Correlation-Adjusted Weighting

- **Better Balance**: Avoids domination by a single component
- **Statistical Foundation**: Still incorporates important data properties
- **Improved Interpretability**: More intuitive weights that balance both risk dimensions
- **Reduced Volatility**: More stable index through balanced contributions
- **Domain Relevance**: Acknowledges the theoretical importance of both climate and geopolitical risks

## Usage Notes

The resulting Combined Global Uncertainty Index serves as a robust measure of global systemic uncertainty that can be used as a driving factor in the state transition equation for the ambiguity aversion index, allowing for precise measurement of how global systemic uncertainty dynamically shifts Chinese market sentiment.
