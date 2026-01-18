# Systemic Co-Ambiguity: An Early Warning Signal for Financial Market Crises

## 1. Executive Summary & Research Objective
This research module develops **Systemic Co-Ambiguity (SCA)**, a novel indicator that quantifies the "Synchronization of Uncertainty" across the market. The objective is to validate $SCA_t$ as a leading indicator for financial crises, providing an early warning signal superior to traditional volatility or correlation metrics.

**Core Hypothesis**: Financial crises are preceded by a transition from *idiosyncratic* uncertainty (diversifiable) to *systemic* uncertainty (non-diversifiable), causing a simultaneous liquidity freeze.

---

## 2. Economic Logic: Why Co-Ambiguity Predicts Crashes

While standard correlations measure how asset *prices* move together (contagion of valuation), **Co-Ambiguity** measures how asset *uncertainties* move together (contagion of information failure).

*   **Normal State**: Uncertainty is idiosyncratic (Asset A is ambiguous, Asset B is clear). Market makers can hedge inventory risk across assets.
*   **Pre-Crisis State**: Structural ambiguity (e.g., pandemic, geopolitical shift) affects *all* valuation models simultaneously.
*   **Mechanism**: When ambiguity synchronizes, diversification fails. Market makers retreat globally, causing a systemic liquidity freeze.
*   **Prediction**: A spike in the correlation of ambiguity across stocks ($SCA_t$) predicts future market-wide drawdowns.

---

## 3. Signal Construction

### A. The Ambiguity Index
Let $\mathcal{A}_{i,t}$ be the Ambiguity Index for asset $i$ at time $t$.

### B. Systemic Co-Ambiguity Index ($SCA_t$)
Defined as the average pairwise correlation of ambiguity indices across the market universe ($N$ stocks) over a rolling window $W$ (e.g., 60 days):

$$
SCA_t = \frac{2}{N(N-1)} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \text{Corr}_t(\mathcal{A}_{i}, \mathcal{A}_{j})
$$

*   **Refinement**: Weighted $SCA_t$ using Market Cap weights ($w_i$) to capture systemic importance:
    $$
    SCA_t^{weighted} = \sum_{i \neq j} w_i w_j \text{Corr}_t(\mathcal{A}_{i}, \mathcal{A}_{j})
    $$

---

## 4. Empirical Validation Strategy

### A. In-Sample Analysis: Explaining Past Crashes
*   **Objective**: Show that $SCA_t$ spikes *before* historical market crashes.
*   **Methodology**:
    1.  **Event Study**: Plot $SCA_t$ around major crisis events (e.g., 2008 GFC, 2015 China Crash, 2020 COVID).
    2.  **Predictive Regression**:
        $$
        \text{Crash}_{t+k} = \alpha + \beta_1 SCA_t + \beta_2 \text{VIX}_t + \beta_3 \text{PriceCorr}_t + \varepsilon
        $$
        *   **Dependent Variable**: Binary dummy (1 if market drops >5% in next $k$ days) or continuous drawdown.
        *   **Controls**: VIX (Fear), Average Correlation of Returns (Standard Contagion).
    3.  **Metric**: Incremental $R^2$ or Pseudo-$R^2$ (Logit) added by $SCA_t$.

### B. Out-of-Sample (OOS) Analysis: Trading Signal Quality
*   **Objective**: Test if a trading strategy based on $SCA_t$ avoids losses in unseen data.
*   **Signal Design**:
    *   **Warning Signal**: If $SCA_t > \text{Threshold}$ (e.g., 90th percentile of rolling 2-year history), switch from Equity to Cash/Bonds.
*   **Performance Metrics**:
    1.  **Signal Efficiency**:
        *   **False Positive Rate (Type I)**: Signal "Crash" $\rightarrow$ Market Up (Cost of hedging).
        *   **False Negative Rate (Type II)**: Signal "Safe" $\rightarrow$ Market Crash (Disaster).
        *   **ROC Curve & AUC**: Area Under Curve > 0.5 implies predictive power; > 0.7 is strong.
    2.  **Portfolio Metrics**:
        *   **Calmar Ratio**: Annualized Return / Maximum Drawdown.
        *   **Sortino Ratio**: Downside risk-adjusted return.

### C. Economic Significance (The "So What?")
*   **Stress Testing**: Compare $SCA_t$ against **SRISK** (Engle et al.) or **CoVaR** (Adrian & Brunnermeier).
    *   *Argument*: SRISK/CoVaR rely on lagging price data. $SCA$ relies on distributional uncertainty, which reacts faster to news.
*   **Lead-Lag Analysis**: Granger Causality test between $SCA_t$ and VIX. Does Uncertainty Synchronization lead Volatility?

---

## 5. Execution Roadmap (Doable Steps)

1.  **Data Prep**:
    *   Compute daily $\mathcal{A}^{CEA}_t$ for all index constituents (e.g., S&P 500 or CSI 300).
    *   Clean data for delisted stocks to avoid survivorship bias.
2.  **Computation**:
    *   Implement efficient rolling correlation matrix calculation (parallelized).
    *   Construct the daily $SCA_t$ time series (2010–2024).
3.  **Validation**:
    *   **Visual Check**: Overlay $SCA_t$ with market index drawdowns.
    *   **Regression**: Run Logit models for 1-month ahead crash prediction.
    *   **Backtest**: Simulate the "Co-Ambiguity Hedging Strategy" (Equity/Cash switch) and compute Calmar Ratios.

---

## Code Folder Overview & Usage Instructions

This section provides comprehensive documentation for the Python code implementing the Systemic Co-Ambiguity analysis framework described above.

### Folder Location

The code for this project is located at:
```
@ext/02_Co_Ambiguity/code/
```

### Python Code Functionality

The `code/` folder contains the following key Python scripts:

1. **`sca_measurement.py`**
   - **Purpose**: Computes the Systemic Co-Ambiguity (SCA) index
   - **Key functionality**:
     - Computes pairwise ambiguity correlations across stocks
     - Implements efficient vectorized SCA calculation
     - Supports both unweighted and market-cap-weighted versions
     - Outputs: Daily SCA time series for the market

2. **`hypothesis_testing.py`**
   - **Purpose**: Implements statistical tests for all five research hypotheses
   - **Key functionality**:
     - `test_hypothesis_1_leading_indicator()`: Tests if SCA leads crises
     - `test_hypothesis_2_incremental_power()`: Tests incremental predictive power
     - `test_hypothesis_3_liquidity_channel()`: Tests liquidity deterioration mechanism
     - `test_hypothesis_4_structural_change()`: Tests regime-dependent effects
     - `test_hypothesis_5_volatility_lead()`: Tests Granger causality with volatility

3. **`backtest_analysis.py`**
   - **Purpose**: Implements trading strategy backtesting and validation
   - **Key functionality**:
     - `generate_dynamic_threshold_signals()`: Creates warning signals based on rolling percentiles
     - `backtest_strategy()`: Evaluates strategy performance metrics
     - `compute_signal_efficiency()`: Calculates ROC/AUC for signal quality
     - `walk_forward_validation()`: Performs out-of-sample validation

4. **`main_pipeline.py`**
   - **Purpose**: Orchestrates the complete co-ambiguity analysis pipeline
   - **Key functionality**:
     - Loads or generates sample data matching the paper's specifications
     - Computes SCA indices for all stocks in the universe
     - Runs all five hypothesis tests with comprehensive diagnostics
     - Performs strategy backtesting with multiple signal variants
     - Generates visualizations and comprehensive research reports

### Data Connection Instructions

This project assumes the availability of high-frequency trading data as described in `@draft/paper_overleaf/QuantAmbi2_article.tex`. Below are step-by-step instructions for connecting your data to the code.

#### 1. File Path Requirements

Place your data files in the following location:
```
@ext/02_Co_Ambiguity/code/data/
```

Create this `data/` subdirectory within the `code/` folder if it does not already exist.

#### 2. Data Format Specifications

The code expects data in **CSV format** (comma-separated values) with the following specifications:

**Ambiguity Indices Data** (`ambiguity_indices.csv`):
- Required columns:
  - `date`: Trading date (format: YYYY-MM-DD)
  - `stock_id`: Unique stock identifier
  - `ambiguity`: Computed CEA index value
- Data types: date (string), stock_id (string), ambiguity (float)
- Example:
```csv
date,stock_id,ambiguity
2018-01-01,Stock0001,0.0234
2018-01-01,Stock0002,0.0156
...
```

**Returns Data** (`returns.csv`):
- Required columns: `date`, `stock_id`, `return` (daily log return)
- Optional columns: `volume`, `close` (closing price)

**Market Index Data** (`market_index.csv`):
- Required columns: `date`, `close` (closing price)
- Optional columns: `open`, `high`, `low`, `volume`

**Liquidity Data** (`liquidity_measures.csv`):
- Required columns: `date`, `stock_id`, `spread` (bid-ask spread)
- Optional columns: `turnover` (turnover rate), `depth` (market depth)

**VIX Data** (`vix_index.csv`):
- Required columns: `date`, `vix_value`

**CoVaR/SRISK Data** (`systemic_risk_measures.csv`):
- Required columns: `date`, `covar`, `srisk` (optional)

#### 3. Modifying Input Parameters

To point the code to your local data files, update the data paths in the configuration section of `main_pipeline.py`:

**Example code modification**:
```python
# In main_pipeline.py, locate the DATA_PATH configuration
DATA_PATH = 'data/'  # Default: uses data/ subdirectory

# To use a custom path, change to:
DATA_PATH = '/path/to/your/data/'

# Then update data loading calls:
ambiguity_data = pd.read_csv(DATA_PATH + 'ambiguity_indices.csv')
returns_data = pd.read_csv(DATA_PATH + 'returns.csv')
market_data = pd.read_csv(DATA_PATH + 'market_index.csv')
```

The code automatically looks for files in the `data/` subdirectory relative to where you run the script.

#### 4. Universal Data Loading Example

Here is a project-agnostic example of how to load your data in Python (adaptable to this project's needs):

```python
import pandas as pd
import os

# Define data directory
DATA_DIR = 'data/'  # Relative to code/ folder

# Load ambiguity indices
ambiguity_df = pd.read_csv(
    os.path.join(DATA_DIR, 'ambiguity_indices.csv'),
    parse_dates=['date']
)

# Pivot to wide format if needed
ambiguity_matrix = ambiguity_df.pivot(
    index='date',
    columns='stock_id',
    values='ambiguity'
)

# Verify data structure
print("Data shape:", ambiguity_matrix.shape)
print("Date range:", ambiguity_matrix.index.min(), "to", ambiguity_matrix.index.max())
print("Missing values:\n", ambiguity_matrix.isnull().sum())

# Handle missing values
ambiguity_matrix = ambiguity_matrix.fillna(method='ffill').fillna(0)
```

**Data validation checklist**:
- [ ] All required columns present
- [ ] Date column parsed correctly
- [ ] No duplicate (date, stock_id) pairs
- [ ] Numeric columns have appropriate data types
- [ ] Sufficient observations (minimum 252 trading days recommended)
- [ ] Survivorship bias addressed (delisted stocks included)

### Code Access & Understanding

#### Accessing the Code Folder

The code is accessible at:
```
@ext/02_Co_Ambiguity/code/
```

This folder contains four main Python files:
- `sca_measurement.py` (SCA computation)
- `hypothesis_testing.py` (statistical testing)
- `backtest_analysis.py` (strategy validation)
- `main_pipeline.py` (pipeline orchestrator)

#### Documentation Within the Code

**Docstrings and Comments**:
- All core functions contain comprehensive docstrings explaining:
  - Function purpose and parameters
  - Return values and data types
  - Algorithm steps and mathematical formulas
  - Usage examples

**Comment Conventions**:
- **Block comments**: Major algorithm sections begin with `# === Section Name ===`
- **Inline comments**: Key mathematical operations include formula references
- **Variable naming**: Follows Python PEP 8 guidelines (lowercase_with_underscores)

**Example function documentation** (from `sca_measurement.py`):
```python
def compute_sca_efficient(self, ambiguity_df):
    """
    Compute Systemic Co-Ambiguity using vectorized operations

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        DataFrame with dates as index and stocks as columns
        Values are individual ambiguity indices

    Returns:
    --------
    sca_series : pandas Series
        Time series of Systemic Co-Ambiguity values
    """
    # Standardize ambiguity within rolling window
    ambiguity_std = self._standardize_ambiguity(ambiguity_df)

    # Compute efficient pairwise correlations
    corr_matrix = ambiguity_std.rolling(
        window=self.corr_window
    ).corr()

    # Extract upper triangle and average
    sca_series = self._compute_unweighted_sca(corr_matrix)
    return sca_series
```

#### Steps to Run the Code

**Prerequisites**:
- Python version: 3.8 or higher
- Operating system: Windows, macOS, or Linux

**Installation**:
1. Open a terminal or command prompt
2. Navigate to the code folder:
   ```bash
   cd @ext/02_Co_Ambiguity/code/
   ```
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

If `requirements.txt` is not available, install manually:
```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn
```

**Basic Execution Commands**:
- **Run the complete pipeline**:
  ```bash
  python main_pipeline.py
  ```
- **Run individual modules**:
  ```bash
  python sca_measurement.py
  python hypothesis_testing.py
  python backtest_analysis.py
  ```

**Expected outputs**:
- Console output showing progress through analysis stages
- Generated visualizations (PNG files) saved in the working directory
- Research report (TXT file) with all hypothesis test results
- Backtest performance metrics for all strategies

#### Troubleshooting Common Issues

**Universal Issues**:

1. **Missing Dependencies**
   - **Symptom**: `ModuleNotFoundError: No module named 'xxx'`
   - **Solution**: Install missing packages:
     ```bash
     pip install xxx
     ```
   - **Prevention**: Always use a virtual environment and install from `requirements.txt`

2. **Data Format Errors**
   - **Symptom**: `ParserError: Error tokenizing data` or `KeyError: 'xxx'`
   - **Solution**: Verify your CSV files match the format specifications above
   - **Check**: Use `pd.read_csv('data/your_file.csv').head()` to inspect column names

3. **Path Mismatches**
   - **Symptom**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/xxx.csv'`
   - **Solution**: Ensure data files are in the `code/data/` subdirectory
   - **Check**: Run `ls data/` (Unix) or `dir data` (Windows) to list files

**Co-Ambiguity-Specific Issues**:

1. **Memory Error with Large Universes**
   - **Symptom**: `MemoryError: Unable to allocate array` when computing SCA
   - **Solution**: Use fewer stocks or reduce correlation window:
     ```python
     # Reduce universe size
     ambiguity_subset = ambiguity_df.iloc[:, :100]

     # Use shorter window
     sca_calc = SystemicCoAmbiguity(corr_window=30)
     ```

2. **Insufficient Observations for Rolling Windows**
   - **Symptom**: `ValueError: Cannot construct rolling window with insufficient observations`
   - **Solution**: Increase data length or reduce window size
   - **Adjustment**: Ensure you have at least 2× the correlation window length

3. **Granger Causality Test Failures**
   - **Symptom**: `ValueError: Insufficient observations for Granger causality`
   - **Solution**: Increase sample size or reduce maximum lag length
   - **Adjustment**: Modify `max_lag` parameter in function call

4. **Perfect Separation in Logistic Regression**
   - **Symptom**: `PerfectSeparationError: Perfect separation detected`
   - **Solution**: Use Firth logistic regression or add regularization
   - **Workaround**: The code includes regularization; check warnings in output

5. **Slow SCA Computation**
   - **Symptom**: Long computation times for large universes
   - **Solution**: Use the efficient vectorized method instead of naive approach:
     ```python
     # Use this (fast)
     sca = sca_calc.compute_sca_efficient(ambiguity_df)

     # Not this (slow)
     sca = sca_calc.compute_sca(ambiguity_df)
     ```

**Getting Additional Help**:
- Check the docstrings: Each function includes detailed documentation
- Review the `README.md` in the `code/` folder for additional technical details
- Examine the generated research report for diagnostic information
- Verify data format using the validation checklist above

---

## Contact & Support

For questions about the research methodology or code implementation, please refer to the research paper located in `@ext/02_Co_Ambiguity/draft/co_ambiguity_paper.tex` for theoretical details and the code documentation above for implementation specifics.
