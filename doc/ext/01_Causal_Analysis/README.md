# Causal Inference in Economic Decision-Making: Ambiguity as a Driver of Asset Returns

## 1. Executive Summary & Research Objective
This research module establishes **Ambiguity** (Model Uncertainty) as a causal determinant of **Asset Returns**, distinct from traditional risk factors. The objective is to move beyond correlation to causality by identifying specific transmission mechanisms and employing rigorous identification strategies (Instrumental Variables).

**Core Hypothesis**: Ambiguity represents a distinct state of "Model Uncertainty" that forces agents to alter their pricing (demanding a premium) and trading behavior (withdrawing liquidity), leading to predictable return patterns.

---

## 2. Theoretical Framework (The "Why")

The causal chain operates through two primary economic channels:

### A. The Ambiguity Premium Channel (Asset Pricing)
*   **Economic Logic**: Based on the Ellsberg Paradox, investors are ambiguity-averse. They prefer known risks to unknown probabilities. When the "model" for an asset becomes uncertain (High Ambiguity), investors cannot maximize expected utility.
*   **Causal Chain**:
    1.  **Shock**: Ambiguity increases (e.g., conflicting news, structural break).
    2.  **Reaction**: Investors demand an additional "Ambiguity Premium" to hold the asset, effectively increasing the discount rate.
    3.  **Price Impact**: Current price $P_t$ falls to accommodate the higher required rate of return.
    4.  **Consequence**: Future realized return $r_{t+1}$ increases as the price recovers or the premium is realized.
*   **Testable Prediction**: Positive relationship between lagged Ambiguity and Future Returns.

### B. The Liquidity Provision Channel (Market Microstructure)
*   **Economic Logic**: Market makers (MMs) rely on inventory and risk models to set quotes. High ambiguity implies "Model Failure," making these models unreliable.
*   **Causal Chain**:
    1.  **Shock**: Ambiguity spikes.
    2.  **Reaction**: MMs widen bid-ask spreads or reduce depth to protect against "unknown" adverse selection (risk of trading against informed agents with superior models).
    3.  **Liquidity Dry-Up**: Trading becomes costly; order flow becomes toxic.
    4.  **Consequence**: This illiquidity is priced. The "illiquidity discount" leads to a short-term price drop, followed by a reversal (return) when liquidity normalizes.

---

## 3. Empirical Research Design

To isolate the effect of Ambiguity, the regression specification must control for "Risk" (known unknowns) to prove that Ambiguity (unknown unknowns) is a distinct pricing factor.

### A. Data & Variable Construction
*   **Dependent Variable**: Future Realized Returns ($r_{t+1}$).
*   **Independent Variable**: Ambiguity Index ($\mathcal{A}^{CEA}_t$).
*   **Control Variables (The "Must-Haves")**:
    1.  **Fundamental Risk**: Realized Volatility (RV), Beta.
    2.  **Tail Risk**: Skewness (crash risk), Kurtosis (fat tails). *Crucial distinction: Skewness measures the shape; Ambiguity measures uncertainty about the shape.*
    3.  **Liquidity**: Turnover Rate, Amihud Illiquidity, Bid-Ask Spread. *Controlling for liquidity helps distinguish direct vs. indirect effects.*
    4.  **Behavioral Factors**: Momentum (past returns), Reversal.
    5.  **Information Environment**: Analyst Coverage, Institutional Ownership ratio.

### B. Econometric Specification
$$
r_{i,t+1} = \alpha + \beta_1 \mathcal{A}^{CEA}_{i,t} + \gamma \mathbf{Controls}_{i,t} + \text{FixedEffects} + \varepsilon_{i,t+1}
$$

---

## 4. Identification Strategy (Addressing Endogeneity)

Does Ambiguity cause Returns, or do falling prices cause Ambiguity (Reverse Causality)? We employ Instrumental Variables (IVs) to isolate exogenous variation.

### A. Instrumental Variable Candidates
1.  **Peer-Based Ambiguity (Granular IV)**
    *   *Construction*: Instrument stock $i$'s ambiguity with the average ambiguity of other stocks in the *same industry* (excluding $i$).
    *   *Logic*: Industry-wide ambiguity shocks (e.g., regulatory uncertainty for Tech) affect stock $i$'s ambiguity but are exogenous to stock $i$'s idiosyncratic return shocks.
2.  **Policy Uncertainty Interaction**
    *   *Construction*: Interaction of **Economic Policy Uncertainty (EPU) Index** $\times$ **Firm's Sensitivity to Policy** (e.g., subsidy dependence).
    *   *Logic*: EPU is a macro shock. Its differential impact is an exogenous shifter of firm-level ambiguity.
3.  **"Non-Fundamental" Information Shocks**
    *   *Construction*: Unexpected length/complexity (file size) of regulatory filings (e.g., 8-K) that *do not* contain earnings surprises.
    *   *Logic*: Complex disclosures increase ambiguity (processing cost) without immediately changing fundamental value.

### B. Robustness Checks
*   **Granger Causality**: Test if Ambiguity leads Returns (and not vice versa).
*   **Placebo Tests**: Test if Ambiguity predicts returns in "safe" assets (should be insignificant) vs. "ambiguous" assets.

---

## 5. Mechanism Analysis (The "How" & "When")

### A. Mediation Analysis (The "How")
Test if the effect runs *through* liquidity or sentiment.
*   **Test**: Mediation analysis (Sobel test).
    *   Path: Ambiguity $\rightarrow$ Bid-Ask Spread $\rightarrow$ Return.
    *   If the direct effect of Ambiguity disappears after controlling for Spreads, the Liquidity Channel is dominant.

### B. Moderation Analysis (The "When")
When is the causal effect strongest?
1.  **Market Regime**: Bull vs. Bear Markets. (Ambiguity aversion is often asymmetric; stronger in Bear markets).
2.  **Volatility Regime**: High vs. Low Volatility.
3.  **Investor Sophistication**: Institutional Ownership.
4.  **Asset Characteristics**: Hard-to-value stocks (high R&D, intangible assets).

---

## 6. Execution Roadmap (Doable Steps)

1.  **Data Gathering**:
    *   Collect daily stock returns, volume, and intraday data (for RV, Skew, Kurt).
    *   Construct the $\mathcal{A}^{CEA}_t$ index for the full universe.
    *   Match with Analyst Coverage and Institutional Ownership data.
2.  **IV Construction**:
    *   Compute industry-average ambiguity (leave-one-out mean).
    *   Download EPU Index data and compute firm sensitivity betas.
3.  **Regression Analysis**:
    *   Run Panel OLS with fixed effects.
    *   Run 2SLS (Two-Stage Least Squares) using the identified IVs.
4.  **Mechanism Testing**:
    *   Run interaction regressions for Moderation Analysis.
    *   Perform Sobel tests for Mediation Analysis.

---

## Code Folder Overview & Usage Instructions

This section provides comprehensive documentation for the Python code implementing the causal analysis framework described above.

### Folder Location

The code for this project is located at:
```
@ext/01_Causal_Analysis/code/
```

### Python Code Functionality

The `code/` folder contains the following key Python scripts:

1. **`ambiguity_measurement.py`**
   - **Purpose**: Computes the Cross-Entropy Ambiguity (CEA) index $\mathcal{A}^{CEA}_t$ for individual stocks
   - **Key functionality**:
     - Discretizes intraday returns into histogram bins
     - Computes Kullback-Leibler divergence between empirical and benchmark distributions
     - Implements dynamic benchmark selection using K-means clustering
     - Outputs: Daily ambiguity indices for all stocks in the universe

2. **`causal_analysis.py`**
   - **Purpose**: Implements causal inference methods to establish ambiguity as a causal determinant of returns
   - **Key functionality**:
     - `baseline_ols()`: Panel regression with fixed effects
     - `instrumental_variables_2sls()`: Two-stage least squares estimation using instrumental variables
     - `granger_causality_test()`: Tests temporal precedence between ambiguity and returns
     - `mediation_analysis()`: Sobel tests for liquidity channel mediation
     - `heterogeneity_analysis()`: Tests effect modification across market regimes

3. **`main_analysis.py`**
   - **Purpose**: Orchestrates the complete causal analysis pipeline
   - **Key functionality**:
     - Loads or generates sample data matching the paper's specifications
     - Computes ambiguity measures for all stocks
     - Generates instrumental variables (peer-based, EPU interaction, filing complexity)
     - Runs all hypothesis tests and generates visualizations
     - Produces comprehensive research reports

### Data Connection Instructions

This project assumes the availability of high-frequency trading data as described in `@draft/paper_overleaf/QuantAmbi2_article.tex`. Below are step-by-step instructions for connecting your data to the code.

#### 1. File Path Requirements

Place your data files in the following location:
```
@ext/01_Causal_Analysis/code/data/
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

**Market Index Data** (`market_index.csv`):
- Required columns: `date`, `close` (closing price)
- Optional columns: `open`, `high`, `low`, `volume`

**Liquidity Data** (`liquidity_measures.csv`):
- Required columns: `date`, `stock_id`, `spread` (bid-ask spread), `turnover` (turnover rate), `depth` (market depth)

**EPU Index Data** (`epu_index.csv`):
- Required columns: `date`, `epu_value`

**Filing Complexity Data** (`filing_complexity.csv`):
- Required columns: `date`, `stock_id`, `file_size` (file size in bytes or pages)

#### 3. Modifying Input Parameters

To point the code to your local data files, update the data paths in the configuration section of `main_analysis.py`:

**Example code modification**:
```python
# In main_analysis.py, locate the DATA_PATH configuration
DATA_PATH = 'data/'  # Default: uses data/ subdirectory

# To use a custom path, change to:
DATA_PATH = '/path/to/your/data/'

# Then update data loading calls:
ambiguity_data = pd.read_csv(DATA_PATH + 'intraday_returns.csv')
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

# Load intraday returns
returns_df = pd.read_csv(os.path.join(DATA_DIR, 'intraday_returns.csv'),
                        parse_dates=['date'])

# Verify data structure
print("Data shape:", returns_df.shape)
print("Columns:", returns_df.columns.tolist())
print("Date range:", returns_df['date'].min(), "to", returns_df['date'].max())

# Check for missing values
print("Missing values:\n", returns_df.isnull().sum())
```

**Data validation checklist**:
- [ ] All required columns present
- [ ] Date column parsed correctly
- [ ] No duplicate rows
- [ ] Numeric columns have appropriate data types
- [ ] Sufficient observations (minimum 252 trading days recommended)

### Code Access & Understanding

#### Accessing the Code Folder

The code is accessible at:
```
@ext/01_Causal_Analysis/code/
```

This folder contains three main Python files:
- `ambiguity_measurement.py` (core ambiguity computation)
- `causal_analysis.py` (causal inference methods)
- `main_analysis.py` (pipeline orchestrator)

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

**Example function documentation** (from `ambiguity_measurement.py`):
```python
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
        KL divergence measuring information loss
    """
    # Add epsilon for numerical stability
    p_safe = p + self.epsilon
    q_safe = np.maximum(q, self.epsilon)

    # Compute KL divergence
    kl_div = np.sum(p_safe * np.log(p_safe / q_safe))
    return kl_div
```

#### Steps to Run the Code

**Prerequisites**:
- Python version: 3.8 or higher
- Operating system: Windows, macOS, or Linux

**Installation**:
1. Open a terminal or command prompt
2. Navigate to the code folder:
   ```bash
   cd @ext/01_Causal_Analysis/code/
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
pip install numpy pandas scikit-learn scipy statsmodels linearmodels matplotlib seaborn
```

**Basic Execution Commands**:
- **Run the complete pipeline**:
  ```bash
  python main_analysis.py
  ```
- **Run individual modules**:
  ```bash
  python ambiguity_measurement.py
  python causal_analysis.py
  ```

**Expected outputs**:
- Console output showing progress through analysis stages
- Generated visualizations (PNG files) saved in the working directory
- Research report (TXT file) with all hypothesis test results

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

**Causal Analysis-Specific Issues**:

1. **Instrumental Variable Weakness**
   - **Symptom**: First-stage F-statistic < 10 in 2SLS output
   - **Solution**: Use stronger instruments or combine multiple instruments
   - **Diagnostic**: Check the `first_stage_f_statistic` value in results

2. **Convergence Warnings in Logistic Regression**
   - **Symptom**: `PerfectSeparationError: Perfect separation detected`
   - **Solution**: Use Firth logistic regression or add regularization
   - **Workaround**: The code handles this automatically; check warnings in output

3. **Granger Causality Test Failures**
   - **Symptom**: `ValueError: Insufficient observations for Granger causality`
   - **Solution**: Increase sample size or reduce maximum lag length
   - **Adjustment**: Modify `max_lag` parameter in function call

4. **Memory Issues with Large Datasets**
   - **Symptom**: `MemoryError: Unable to allocate array`
   - **Solution**: Process data in batches or use data types with lower memory footprint
   - **Optimization**: Reduce `window_size` or `n_clusters` parameters in ambiguity computation

**Getting Additional Help**:
- Check the docstrings: Each function includes detailed documentation
- Review the `README.md` in the `code/` folder for additional technical details
- Examine the generated research report for diagnostic information

---

## Contact & Support

For questions about the research methodology or code implementation, please refer to the research paper located in `@ext/01_Causal_Analysis/draft/causal_ambiguity_paper.tex` for theoretical details and the code documentation above for implementation specifics.
