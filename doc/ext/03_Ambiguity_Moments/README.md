# Ambiguity vs. Higher-Order Moments: Distinguishing Model Uncertainty from Tail Risk in Financial Markets

## 1. Executive Summary & Research Objective
This research module distinguishes **Ambiguity** from **Higher-Order Moments** (Skewness/Kurtosis) both theoretically and empirically. The goal is to prove that Ambiguity captures "Second-Order Uncertainty" (uncertainty *about* the distribution) distinct from Tail Risk (properties *of* the distribution), and that their **Interaction** provides a superior crash prediction signal.

**Core Hypothesis**: Ambiguity amplifies Tail Risk. A market with negative skewness is fragile ("thin ice"), but high ambiguity ("fog") makes it impossible to assess the risk, leading to panic.

---

## 2. Theoretical Distinction

To publish in top-tier journals, the conceptual difference must be clear using the **Knightian Uncertainty** framework:

*   **Risk (Higher-Order Moments)**: Describes a *known* probability distribution $P$.
    *   **Skewness (3rd Moment)**: Asymmetry. "I know the distribution is left-skewed, so a -10% drop is more likely than a +10% gain."
    *   **Kurtosis (4th Moment)**: Tail thickness. "I know extreme events are frequent."
    *   *Key*: The investor trusts the model $P$.
*   **Ambiguity (Entropy/Divergence)**: Describes the *confidence in* distribution $P$.
    *   **Ambiguity**: "I observe distribution $P$, but I don't trust it because the market regime is shifting. The true distribution might be $Q$."
    *   *Key*: Uncertainty *about* the probability (Second-Order Probability).

---

## 3. Empirical Verification: Proving Orthogonality

We must demonstrate that Ambiguity contains unique information not captured by Skewness or Kurtosis.

### A. Correlation Analysis
*   **Method**: Calculate time-series correlation matrix for Ambiguity ($\mathcal{A}^{CEA}_t$), Realized Volatility (RV), Skewness, and Kurtosis.
*   **Hypothesis**: Correlation should be low (e.g., $< 0.3$). Skewness/Kurtosis are driven by price jumps; Ambiguity is driven by distributional *instability*.

### B. Regression Orthogonality Test
*   **Model**: Regress Ambiguity on contemporaneous moments.
    $$
    \mathcal{A}^{CEA}_t = \alpha + \beta_1 \text{RV}_t + \beta_2 \text{Skewness}_t + \beta_3 \text{Kurtosis}_t + \varepsilon_t
    $$
*   **Metric**: Examine $R^2$. Low $R^2$ ($< 10\%$) proves that most variation in Ambiguity is unexplained by moments. $\varepsilon_t$ is "Pure Ambiguity."

### C. Principal Component Analysis (PCA)
*   **Method**: Run PCA on $[\mathcal{A}^{CEA}, \text{RV}, \text{Skew}, \text{Kurt}]$.
*   **Hypothesis**: Ambiguity loads on a distinct factor separate from the "Tail Risk" factor.

---

## 4. Crash Prediction Research Design: The Interaction Effect

### A. Econometric Model (Logit/Probit)
Define **Crash Event** ($Y_{t+1}$) as binary (e.g., Market Return $< -5\%$).

$$
\text{Prob}(Y_{t+1}=1) = \Phi \left( \alpha + \beta_1 \text{Skew}_t + \beta_2 \text{Kurt}_t + \beta_3 \mathcal{A}^{CEA}_t + \beta_4 (\text{Skew}_t \times \mathcal{A}^{CEA}_t) + \text{Controls} \right)
$$

*   **Key Coefficients**:
    *   $\beta_1$ (Skew): Negative (Lower skew $\to$ Higher crash prob).
    *   $\beta_3$ (Ambiguity): Positive (Higher ambiguity $\to$ Higher crash prob).
    *   **$\beta_4$ (Interaction)**: **Crucial Test**. Significant Negative coefficient means *High Ambiguity makes Negative Skewness even more dangerous*.

### B. Evaluation Metrics
1.  **Likelihood Ratio Test**: Compare full model (with interaction) vs. restricted model.
2.  **AUC (Area Under ROC Curve)**: Does adding Interaction term increase AUC significantly?

---

## 5. Portfolio Implication: Double-Sorting Strategy

To prove economic value (profitability):

### A. Strategy Design
1.  **First Sort**: Sort stocks into 5 quintiles based on **Skewness**.
2.  **Second Sort**: Within the lowest Skewness quintile (High Crash Risk), sort into 2 groups based on **Ambiguity** (High vs. Low).

### B. Prediction
*   **"Toxic" Portfolio**: Low Skewness + High Ambiguity. Should have lowest returns/highest drawdown.
*   **"Stable" Portfolio**: High Skewness + Low Ambiguity.

### C. Execution
*   **Long-Short Strategy**: Short "Toxic", Long "Stable".
*   **Test**: Calculate Alpha relative to Fama-French 5-factor model.

---

## 6. Execution Roadmap (Doable Steps)

1.  **Data Prep**:
    *   Compute daily Skewness, Kurtosis, and $\mathcal{A}^{CEA}_t$ for all stocks.
2.  **Orthogonality Check**:
    *   Run correlation matrix and PCA in Python/R.
3.  **Crash Prediction**:
    *   Define "Crash" (e.g., rolling 3-sigma drop).
    *   Run Logit regression with Interaction term.
4.  **Portfolio Backtest**:
    *   Implement Double-Sort logic.
    *   Compute monthly returns and alphas.

---

## Code Folder Overview & Usage Instructions

This section provides comprehensive documentation for the Python code implementing the ambiguity vs. higher-order moments analysis framework described above.

### Folder Location

The code for this project is located at:
```
@ext/03_Ambiguity_Moments/code/
```

### Python Code Functionality

The `code/` folder contains the following key Python scripts:

1. **`moments_analysis.py`**
   - **Purpose**: Core module implementing all five hypothesis tests
   - **Key functionality**:
     - `test_hypothesis_1_correlation()`: Tests correlation orthogonality
     - `test_hypothesis_2_regression()`: Tests regression orthogonality
     - `test_hypothesis_3_interaction()`: Tests interaction effect for crash prediction
     - `test_hypothesis_4_pca()`: Tests factor structure distinctness
     - `test_hypothesis_5_portfolio()`: Tests portfolio value through double-sorting
   - Outputs: Comprehensive test results for each hypothesis

2. **`main_pipeline.py`**
   - **Purpose**: Orchestrates the complete analysis pipeline
   - **Key functionality**:
     - Loads or generates sample data matching the paper's specifications
     - Computes ambiguity measures and higher-order moments
     - Runs all five hypothesis tests with detailed diagnostics
     - Generates visualizations (correlation heatmaps, PCA plots, portfolio performance)
     - Creates comprehensive research reports

### Data Connection Instructions

This project assumes the availability of high-frequency trading data as described in `@draft/paper_overleaf/QuantAmbi2_article.tex`. Below are step-by-step instructions for connecting your data to the code.

#### 1. File Path Requirements

Place your data files in the following location:
```
@ext/03_Ambiguity_Moments/code/data/
```

Create this `data/` subdirectory within the `code/` folder if it does not already exist.

#### 2. Data Format Specifications

The code expects data in **CSV format** (comma-separated values) with the following specifications:

**Intraday Returns Data** (`intraday_returns.csv`):
- Required columns:
  - `date`: Trading date (format: YYYY-MM-DD)
  - `time`: Intraday timestamp (format: HH:MM:SS or Unix timestamp)
  - `stock_id`: Unique stock identifier
  - `return`: One-minute log return
- Data types: date (string), time (string), stock_id (string), return (float)
- Example:
```csv
date,time,stock_id,return
2018-01-01,09:30:00,Stock0001,0.000234
2018-01-01,09:31:00,Stock0001,-0.000156
...
```

**Daily Returns Data** (`daily_returns.csv`):
- Required columns: `date`, `stock_id`, `return` (daily log return)
- This can be computed from intraday data if not directly available

**Market Index Data** (`market_index.csv`):
- Required columns: `date`, `close` (closing price)
- Optional columns: `open`, `high`, `low`, `volume`

**Fama-French Factors** (`fama_french_factors.csv`):
- Required columns: `date`, `MKT_RF`, `SMB`, `HML`, `RMW`, `CMA` (five-factor model)
- Used for portfolio alpha computation

#### 3. Modifying Input Parameters

To point the code to your local data files, update the data paths in the configuration section of `main_pipeline.py`:

**Example code modification**:
```python
# In main_pipeline.py, locate the DATA_PATH configuration
DATA_PATH = 'data/'  # Default: uses data/ subdirectory

# To use a custom path, change to:
DATA_PATH = '/path/to/your/data/'

# Then update data loading calls:
returns_data = pd.read_csv(DATA_PATH + 'daily_returns.csv')
market_data = pd.read_csv(DATA_PATH + 'market_index.csv')
ff_factors = pd.read_csv(DATA_PATH + 'fama_french_factors.csv')
```

The code automatically looks for files in the `data/` subdirectory relative to where you run the script.

#### 4. Universal Data Loading Example

Here is a project-agnostic example of how to load your data in Python (adaptable to this project's needs):

```python
import pandas as pd
import numpy as np
import os

# Define data directory
DATA_DIR = 'data/'  # Relative to code/ folder

# Load daily returns
returns_df = pd.read_csv(
    os.path.join(DATA_DIR, 'daily_returns.csv'),
    parse_dates=['date']
)

# Pivot to wide format (stocks as columns)
returns_matrix = returns_df.pivot(
    index='date',
    columns='stock_id',
    values='return'
)

# Compute higher-order moments
def compute_moments(returns_df, window=20):
    """Compute realized volatility, skewness, and kurtosis"""
    moments_dict = {
        'RV': returns_df.rolling(window=window).std(),
        'Skew': returns_df.rolling(window=window).skew(),
        'Kurt': returns_df.rolling(window=window).kurtosis()
    }
    return moments_dict

# Compute moments
moments = compute_moments(returns_matrix)

# Verify data structure
print("Returns shape:", returns_matrix.shape)
print("Date range:", returns_matrix.index.min(), "to", returns_matrix.index.max())
print("Missing values:\n", returns_matrix.isnull().sum().sum())

# Handle missing values
returns_matrix = returns_matrix.fillna(method='ffill').fillna(0)
```

**Data validation checklist**:
- [ ] All required columns present
- [ ] Date column parsed correctly
- [ ] No duplicate (date, stock_id) pairs
- [ ] Numeric columns have appropriate data types
- [ ] Sufficient observations (minimum 252 trading days recommended for moment computation)
- [ ] Returns are in log format (or can be converted)

### Code Access & Understanding

#### Accessing the Code Folder

The code is accessible at:
```
@ext/03_Ambiguity_Moments/code/
```

This folder contains two main Python files:
- `moments_analysis.py` (core hypothesis testing)
- `main_pipeline.py` (pipeline orchestrator)

#### Documentation Within the Code

**Docstrings and Comments**:
- All core functions contain comprehensive docstrings explaining:
  - Function purpose and parameters
  - Return values and data types
  - Algorithm steps and mathematical formulas
  - Expected output formats

**Comment Conventions**:
- **Block comments**: Major algorithm sections begin with `# === Section Name ===`
- **Inline comments**: Key statistical operations include formula references
- **Variable naming**: Follows Python PEP 8 guidelines (lowercase_with_underscores)

**Example function documentation** (from `moments_analysis.py`):
```python
def test_hypothesis_1_correlation(ambiguity_df, moments_dict):
    """
    Test H1: Ambiguity and moments exhibit low correlation

    Parameters:
    -----------
    ambiguity_df : pandas DataFrame
        Daily ambiguity indices (dates x stocks)
    moments_dict : dict of pandas DataFrames
        Dictionary with keys 'RV', 'Skew', 'Kurt'

    Returns:
    --------
    results : dict
        Dictionary containing for each moment:
        - mean_correlation: Average correlation across time
        - ci_lower: 95% CI lower bound
        - ci_upper: 95% CI upper bound
        - p_value: P-value for H0: correlation >= 0.3
        - orthogonality_confirmed: Boolean
    """
    # Implementation details...
```

#### Steps to Run the Code

**Prerequisites**:
- Python version: 3.8 or higher
- Operating system: Windows, macOS, or Linux

**Installation**:
1. Open a terminal or command prompt
2. Navigate to the code folder:
   ```bash
   cd @ext/03_Ambiguity_Moments/code/
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
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn
```

**Basic Execution Commands**:
- **Run the complete pipeline**:
  ```bash
  python main_pipeline.py
  ```
- **Run individual modules**:
  ```bash
  python moments_analysis.py
  ```

**Expected outputs**:
- Console output showing progress through all five hypothesis tests
- Generated visualizations:
  - Correlation heatmap
  - PCA factor loading plots
  - Portfolio performance charts
  - ROC curves for crash prediction
- Research report (TXT file) with comprehensive test results

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

**Ambiguity Moments-Specific Issues**:

1. **Insufficient Data for Moment Computation**
   - **Symptom**: `ValueError: Cannot compute moments with insufficient observations`
   - **Solution**: Increase sample size or reduce rolling window size
   - **Adjustment**: Ensure you have at least 2× the window length for reliable moment estimates

2. **High Correlation Between Moments**
   - **Symptom**: Warning about high multicollinearity in regression
   - **Solution**: This is expected; moments are naturally correlated
   - **Note**: The code handles this using VIF diagnostics and PCA

3. **Perfect Separation in Logistic Regression**
   - **Symptom**: `PerfectSeparationError: Perfect separation detected in crash prediction`
   - **Solution**: The code includes regularization; check warnings in output
   - **Workaround**: Reduce the number of predictors or use Firth logistic regression

4. **Portfolio Formation with Small Sample**
   - **Symptom**: Warning about insufficient stocks in quintile groups
   - **Solution**: Use fewer quantiles (terciles instead of quintiles) or increase stock universe
   - **Adjustment**: Modify `n_quantiles` parameter in portfolio formation function

5. **Memory Issues with Large Datasets**
   - **Symptom**: `MemoryError: Unable to allocate array` during correlation computation
   - **Solution**: Process data in batches or use fewer stocks
   - **Optimization**:
     ```python
     # Reduce universe size for testing
     ambiguity_subset = ambiguity_df.iloc[:, :100]
     moments_subset = {k: v.iloc[:, :100] for k, v in moments_dict.items()}
     ```

6. **Skewness/Kurtosis Computation Warnings**
   - **Symptom**: `InvalidValueError: Skewness/kurtosis computation requires at least 3 observations`
   - **Solution**: Increase rolling window size or remove assets with insufficient data
   - **Adjustment**: The code automatically handles this with minimum observation checks

**Getting Additional Help**:
- Check the docstrings: Each function includes detailed documentation
- Review the `README.md` in the `code/` folder for additional technical details
- Examine the generated research report for diagnostic information
- Verify data format using the validation checklist above

---

## Contact & Support

For questions about the research methodology or code implementation, please refer to the research paper located in `@ext/03_Ambiguity_Moments/draft/ambiguity_moments_paper.tex` for theoretical details and the code documentation above for implementation specifics.
