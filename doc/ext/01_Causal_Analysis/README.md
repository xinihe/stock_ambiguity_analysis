# China Energy Market Ambiguity Analysis: Complete Documentation

## Paper: "Pricing the Unknown: Ambiguity Premiums in China's Green vs. Brown Energy Markets"

---

## Table of Contents
1. [Research Overview](#1-research-overview)
2. [Database Connection](#2-database-connection)
3. [Code Structure](#3-code-structure)
4. [Installation Guide](#4-installation-guide)
5. [Quick Start](#5-quick-start)
6. [Detailed Usage Instructions](#6-detailed-usage-instructions)
7. [Data Requirements](#7-data-requirements)
8. [Output Files](#8-output-files)
9. [Troubleshooting](#9-troubleshooting)
10. [Theoretical Background](#10-theoretical-background)

---

## 1. Research Overview

### Research Question
How does **ambiguity** (model uncertainty) affect asset returns in China's energy market during the "Dual Carbon" transition (Peaking Carbon by 2030, Carbon Neutrality by 2060)?

### Core Hypothesis
Ambiguity represents a distinct state of "model uncertainty" that:
1. Causes investors to demand an **ambiguity premium** (positive relationship between ambiguity and future returns)
2. Creates a **liquidity dry-up** as market makers withdraw during uncertain times
3. Affects **Brown vs. Green** energy stocks differentially (Green discount due to policy support)
4. Responds to **policy shocks** (DiD analysis around major announcements)
5. Transmits through **multiple channels** (direct pricing and indirect liquidity)

### Energy Sector Classification
- **Brown Energy**: Traditional fossil fuels (Coal, Oil & Gas, Thermal Power)
- **Green Energy**: Renewables (Solar, Wind, EV, Batteries, Hydro)
- **Grey Energy**: Utilities and grid infrastructure

### Key Policy Dates (Natural Experiments)
- **2020-09-22**: Xi's UN speech (2060 carbon neutrality pledge)
- **2021-03-15**: 14th Five-Year Plan approval
- **2021-07-16**: National carbon market launch
- **2021-10-24**: "Dual Carbon" policy documents (1+N framework)

---

## 2. Database Connection

### PostgreSQL Database Details

**IMPORTANT**: All high-frequency stock data is stored in a PostgreSQL database that you need to connect to.

```
Server IP:     10.28.255.30
Host Name:     Ubuntu server (research team internal)
Database:      stock_hf
Port:          5432 (default PostgreSQL port)
```

### Database Schema

The database contains a main table with high-frequency stock data:

**Table: `stock_hf`**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `code` | VARCHAR | Stock identifier | '601857.SH' (PetroChina), '300750.SZ' (CATL) |
| `closeprice` | DECIMAL | Closing price at timestamp | 45.23 |
| `datetime` | TIMESTAMP | Date and time of observation | '2020-01-01 09:30:00' |

**Primary Keys**: (code, datetime)
**Indexes**: datetime (for time queries), code (for stock queries)

### Stock Code Convention
Chinese A-share stocks follow this naming pattern:
- **Shanghai Main Board**: 6XXXXX.SH (e.g., 601857.SH = PetroChina)
- **Shenzhen Main Board**: 0XXXXXX.SZ (e.g., 000001.SZ = Ping An)
- **ChiNext (Growth Enterprise)**: 3XXXXXX.SZ (e.g., 300750.SZ = CATL)
- **STAR Market**: 688XXX.SH (tech innovation board)

### Setting Up Database Connection

#### Method 1: Environment Variables (Recommended)
Set up environment variables in your shell or IDE:

```bash
# Linux/Mac (add to ~/.bashrc or ~/.zshrc)
export DB_USER='your_username'
export DB_PASSWORD='your_password'

# Windows (set in System Environment Variables)
setx DB_USER "your_username"
setx DB_PASSWORD "your_password"
```

Then run the scripts - they will automatically use these variables.

#### Method 2: Hardcoded (Not Recommended for Security)
Modify the `__init__` call in scripts to pass credentials:

```python
loader = ChinaEnergyDataLoader(
    db_host='10.28.255.30',
    db_name='stock_hf',
    db_user='your_username',
    db_password='your_password'
)
```

**WARNING**: Do not commit hardcoded credentials to version control!

#### Method 3: Configuration File
Create a `db_config.ini` file (add to `.gitignore`):

```ini
[database]
host = 10.28.255.30
port = 5432
dbname = stock_hf
user = your_username
password = your_password
```

Then load in Python:
```python
import configparser
config = configparser.ConfigParser()
config.read('db_config.ini')
```

### Testing Database Connection

Test your connection before running the full pipeline:

```python
from data_loader import ChinaEnergyDataLoader
import os

# Set credentials
os.environ['DB_USER'] = 'your_username'
os.environ['DB_PASSWORD'] = 'your_password'

# Initialize loader
loader = ChinaEnergyDataLoader(
    db_host='10.28.255.30',
    db_name='stock_hf'
)

# Test connection
if loader.db_engine is not None:
    print("✓ Database connection successful!")
    stock_universe = loader.load_stock_universe_from_db()
    print(f"✓ Found {len(stock_universe)} stocks")
else:
    print("✗ Connection failed. Check credentials and network.")
```

### Network Requirements

To connect to the database:
1. **VPN Access**: If the server is behind a firewall, connect to your institution's VPN first
2. **SSH Tunnel** (alternative method):
   ```bash
   ssh -L 5432:localhost:5432 user@10.28.255.30
   ```
   Then connect to `localhost:5432` instead of `10.28.255.30`

3. **Firewall Rules**: Ensure port 5432 is open on your network

### Common Database Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Connection timeout** | "Could not connect to server" | Check VPN, ping 10.28.255.30, verify firewall |
| **Authentication failed** | "password authentication failed" | Verify username/password, check account status |
| **Database doesn't exist** | "database 'stock_hf' does not exist" | Verify database name with admin |
| **Slow queries** | Queries take >10 minutes | Add date filters, reduce stock list, add indexes |
| **Memory error** | "out of memory" | Process data in chunks, use smaller date range |

### Query Examples

**Basic query for one stock**:
```sql
SELECT code, datetime, closeprice
FROM stock_hf
WHERE code = '601857.SH'
  AND datetime >= '2020-01-01'
  AND datetime <= '2020-12-31'
ORDER BY datetime;
```

**Query for multiple stocks**:
```sql
SELECT code, datetime, closeprice
FROM stock_hf
WHERE code IN ('601857.SH', '300750.SZ', '601012.SH')
  AND datetime >= '2020-01-01'
  AND datetime <= '2020-12-31'
ORDER BY datetime, code;
```

**Get all available stocks**:
```sql
SELECT DISTINCT code
FROM stock_hf
ORDER BY code;
```

---

## 3. Code Structure

### File Organization

```
@ext/01_Causal_Analysis/
├── code/
│   ├── data_loader.py           # Database connection and data loading
│   ├── ambiguity_measurement.py  # CEA calculation algorithms
│   ├── causal_analysis.py        # Causal inference methods
│   ├── main_china_energy.py      # Main pipeline orchestrator
│   └── data/                     # Local data directory (fallback)
│       ├── epu_china.csv
│       ├── policy_sensitivity.csv
│       ├── geopolitical_indices.csv
│       └── ...
├── draft/
│   ├── causal_ambi_china.tex     # Main paper
│   └── energy_industry_proposal.md
└── README.md                     # This file
```

### Module Descriptions

#### 1. `data_loader.py` (1,800+ lines)
**Purpose**: Connect to PostgreSQL database and load all required data

**Key Classes**:
- `ChinaEnergyDataLoader`: Main data loading class
  - Connects to database at 10.28.255.30
  - Queries high-frequency price data
  - Calculates log returns
  - Classifies energy stocks
  - Loads control variables

**Key Functions**:
```python
load_intraday_returns_from_db(stock_list, start_date, end_date)
    # Main method to load stock data from database
    # Returns: DataFrame with datetime index, stocks as columns

classify_energy_stocks(stock_list, sector_mapping=None)
    # Classifies stocks as Brown/Green/Grey
    # Returns: dict {stock_id: 'Brown'/'Green'/'Grey'}

load_limit_days(stock_list, start_date, end_date)
    # Identifies limit-up/limit-down days
    # Returns: dict {stock_id: [dates]}

load_epu_data(start_date, end_date)
    # Loads Economic Policy Uncertainty index
    # Returns: Series with dates as index

load_policy_sensitivity(stock_list, start_date, end_date)
    # Loads firm-level policy dependence scores
    # Returns: DataFrame (dates × stocks)

load_geopolitical_data(start_date, end_date)
    # Loads defense and gold returns
    # Returns: dict {'defense': Series, 'gold': Series}

load_control_data(start_date, end_date)
    # Loads oil and carbon prices
    # Returns: dict {'oil_returns': Series, 'ets_returns': Series}
```

**Usage Example**:
```python
from data_loader import ChinaEnergyDataLoader
import os

# Set credentials
os.environ['DB_USER'] = 'your_username'
os.environ['DB_PASSWORD'] = 'your_password'

# Initialize
loader = ChinaEnergyDataLoader(
    db_host='10.28.255.30',
    db_name='stock_hf'
)

# Load data
returns = loader.load_intraday_returns_from_db(
    stock_list=['601857.SH', '300750.SZ'],
    start_date='2020-01-01',
    end_date='2022-12-31'
)
```

#### 2. `ambiguity_measurement.py` (600+ lines)
**Purpose**: Calculate Cross-Entropy Ambiguity (CEA) indices

**Key Classes**:
- `AmbiguityMeasurement`: Firm-level CEA calculation
- `EnergySectorAmbiguity`: Sector and composite ambiguity

**Key Algorithms**:
```python
compute_ambiguity_for_stock(intraday_returns, limit_days=None)
    # Calculates CEA for a single stock
    # Uses KL divergence between empirical and benchmark distributions
    # Returns: Series of daily CEA values

compute_sector_ambiguity(ambiguity_df)
    # Calculates value-weighted sector CEA
    # Returns: dict {'Brown': Series, 'Green': Series, 'Grey': Series}

compute_composite_ambiguity(ambiguity_df, n_components=1)
    # Extracts systematic ambiguity via PCA
    # First PC = common ambiguity factor across all stocks
    # Returns: Series of composite CEA

compute_policy_ambiguity(index_returns, ambiguity_measure=None)
    # Calculates CEA on policy-sensitive index
    # Returns: Series of policy ambiguity

compute_geopolitical_ambiguity(defense_returns, gold_returns)
    # Calculates geopolitical ambiguity from defense/gold
    # Returns: Series of geopolitical CEA
```

**Mathematical Foundation**:
```
CEA_t = KL(P_t || Q_t) = Σ P_t(x) log(P_t(x) / Q_t(x))

Where:
- P_t: Empirical distribution of intraday returns at time t
- Q_t: Benchmark distribution (expected model)
- KL: Kullback-Leibler divergence
```

#### 3. `causal_analysis.py` (700+ lines)
**Purpose**: Implement causal inference methods

**Key Class**:
- `CausalAmbiguityAnalysis`: All causal testing methods

**Key Methods**:
```python
baseline_panel_ols(dependent_var, ambiguity_var, green_var)
    # Equation (10): Tests H1 (Ambiguity Premium)
    # r_{i,t+1} = α + β₁CEA_{i,t} + β₂CEA×Green + Controls
    # Returns: regression results

instrumental_variables_2sls(dependent_var, endogenous_var, instruments)
    # Equations (11-12): Three-instrument IV strategy
    # Instruments: Peer CEA, EPU×PolicySens, Geopolitical CEA
    # Returns: First-stage and second-stage results

difference_in_differences(policy_shock_dates, window_days=30)
    # Equation (13): DiD around policy shocks
    # Treatment: Green energy, Control: Brown energy
    # Returns: DiD estimator and p-values

mediation_analysis(independent_var, mediator_var, dependent_var)
    # Equations (14-15): Tests liquidity channel
    # Path: CEA → Liquidity → Returns
    # Returns: Direct, indirect, and total effects

moderation_analysis(moderator_var)
    # Equation (16): Tests Green vs. Brown differential
    # Returns: Main effect and interaction effect

granger_causality_test(ambiguity_var, returns_var, max_lag=5)
    # Tests temporal precedence
    # Returns: F-statistics and p-values
```

#### 4. `main_china_energy.py` (500+ lines)
**Purpose**: Orchestrate complete analysis pipeline

**Key Class**:
- `ChinaEnergyPipeline`: Main pipeline controller

**Pipeline Steps**:
1. `load_data()`: Load all data from database
2. `compute_ambiguity_measures()`: Calculate all CEA indices
3. `prepare_analysis_dataset()`: Merge for causal analysis
4. `run_causal_analysis()`: Run all hypothesis tests
5. `visualize_results()`: Generate all figures
6. `generate_report()`: Create text report

**Usage**:
```python
from main_china_energy import ChinaEnergyPipeline

# Initialize
pipeline = ChinaEnergyPipeline(
    data_path='data/',
    output_path='output/'
)

# Run full pipeline
pipeline.run_full_pipeline()
```

---

## 4. Installation Guide

### System Requirements

**Operating System**:
- Linux (Ubuntu 20.04+ recommended)
- macOS 10.15+
- Windows 10+ (with WSL2 recommended)

**Python Version**:
- Python 3.8 or higher
- Python 3.10+ recommended

**Hardware**:
- Minimum 8GB RAM (16GB+ recommended for full dataset)
- 10GB free disk space
- Internet connection for database and package downloads

### Step-by-Step Installation

#### Step 1: Install Python

**Linux/macOS**:
```bash
# Check if Python is installed
python3 --version

# If not installed (Ubuntu)
sudo apt update
sudo apt install python3.10 python3-pip python3-venv
```

**Windows**:
1. Download from https://www.python.org/downloads/
2. Run installer (check "Add Python to PATH")
3. Verify: `python --version`

#### Step 2: Create Virtual Environment

**Linux/macOS**:
```bash
cd /path/to/01_Causal_Analysis/code
python3 -m venv venv
source venv/bin/activate
```

**Windows**:
```cmd
cd C:\path\to\01_Causal_Analysis\code
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Install Dependencies

Create a `requirements.txt` file in the `code/` directory:

```txt
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Database
psycopg2-binary>=2.9.0
sqlalchemy>=1.4.0

# Machine Learning
scikit-learn>=1.0.0

# Econometrics
statsmodels>=0.13.0
linearmodels>=4.20.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
python-dotenv>=0.19.0
tqdm>=4.62.0
```

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Install PostgreSQL Client (Optional)

If you need command-line database access:

**Linux**:
```bash
sudo apt install postgresql-client
```

**macOS**:
```bash
brew install postgresql
```

**Windows**: Download from https://www.enterprisedb.com/downloads/postgres-postgresql-downloads

Test connection:
```bash
psql -h 10.28.255.30 -U your_username -d stock_hf
```

#### Step 5: Configure Environment Variables

Create a `.env` file in the `code/` directory:

```bash
# Database credentials
DB_USER=your_username
DB_PASSWORD=your_password

# Optional: Database settings
DB_HOST=10.28.255.30
DB_PORT=5432
DB_NAME=stock_hf
```

Add to `.gitignore`:
```
.env
venv/
*.pyc
__pycache__/
output/
*.log
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()

import os
db_user = os.environ.get('DB_USER')
```

---

## 5. Quick Start

### 5-Minute Setup

1. **Clone/Download the code**:
   ```bash
   cd /path/to/research/Ambiguity/doc/ext/01_Causal_Analysis/code
   ```

2. **Set up credentials**:
   ```bash
   export DB_USER='your_username'
   export DB_PASSWORD='your_password'
   ```

3. **Run the pipeline**:
   ```bash
   python main_china_energy.py
   ```

That's it! The pipeline will:
- Connect to the database
- Load all required data
- Calculate ambiguity measures
- Run causal analysis
- Generate figures and reports

### Expected Output

**Console output**:
```
======================================================================
CHINA ENERGY MARKET AMBIGUITY ANALYSIS
Full Pipeline Execution
======================================================================

STEP 1: Loading Data
======================================================================
Loading Intraday Returns from Database
...
✓ Loaded 1,234,567 price observations from database
✓ Stocks classified: 150
  - Brown energy: 45
  - Green energy: 78
  - Grey energy: 27

STEP 2: Computing Ambiguity Measures
...
[continues through all steps]

✓ PIPELINE COMPLETED SUCCESSFULLY
All outputs saved to: /path/to/output/
```

**Generated files** in `output/`:
```
cea_timeseries.png
cea_distribution.png
pca_components.png
policy_shocks.png
results_summary.png
analysis_report.txt
```

---

## 6. Detailed Usage Instructions

### Step 1: Test Database Connection

Before running the full analysis, verify your database connection works:

```python
# test_connection.py
from data_loader import ChinaEnergyDataLoader
import os

# Set credentials
os.environ['DB_USER'] = 'your_username'
os.environ['DB_PASSWORD'] = 'your_password'

# Initialize loader
loader = ChinaEnergyDataLoader(
    db_host='10.28.255.30',
    db_name='stock_hf'
)

# Test connection
if loader.db_engine:
    print("✓ Connected!")

    # Load stock universe
    stocks = loader.load_stock_universe_from_db()
    print(f"✓ Found {len(stocks)} stocks")

    # Test loading data for one stock
    test_data = loader.load_intraday_returns_from_db(
        stock_list=['601857.SH'],
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    print(f"✓ Loaded test data: {test_data.shape}")
else:
    print("✗ Connection failed!")
```

Run: `python test_connection.py`

### Step 2: Load Data for Specific Stocks

If you want to analyze specific stocks:

```python
from data_loader import ChinaEnergyDataLoader, prepare_analysis_data
import os

os.environ['DB_USER'] = 'your_username'
os.environ['DB_PASSWORD'] = 'your_password'

# Define your stock list
my_stocks = [
    '601857.SH',  # PetroChina (Brown)
    '300750.SZ',  # CATL (Green)
    '601012.SH',  # LONGi Green Energy (Green)
    '600900.SH',  # Yangtze Power (Grey)
]

# Initialize loader
loader = ChinaEnergyDataLoader(
    db_host='10.28.255.30',
    db_name='stock_hf'
)

# Prepare data
data = prepare_analysis_data(
    data_loader=loader,
    stock_list=my_stocks,
    start_date='2020-01-01',
    end_date='2022-12-31'
)

print(f"Loaded data for {len(my_stocks)} stocks")
```

### Step 3: Calculate Ambiguity Measures

```python
from ambiguity_measurement import AmbiguityMeasurement, EnergySectorAmbiguity

# Initialize
ambiguity_meas = AmbiguityMeasurement()
sector_ambiguity = EnergySectorAmbiguity()

# Get returns from data_dict
returns = data['intraday_returns']
limit_days = data['limit_days']

# Calculate firm-level CEA
firm_cea = ambiguity_meas.compute_ambiguity_cross_section(
    returns_data=returns,
    limit_days_dict=limit_days
)

# Calculate sector CEA
sector_cea = sector_ambiguity.compute_sector_ambiguity(firm_cea)

# Calculate composite CEA (PCA)
composite_cea = sector_ambiguity.compute_composite_ambiguity(firm_cea)

print(f"firm_cea: {firm_cea.shape}")
print(f"sector_cea: {list(sector_cea.keys())}")
print(f"composite_cea: {len(composite_cea)} periods")
```

### Step 4: Run Causal Analysis

```python
from causal_analysis import CausalAmbiguityAnalysis

# Prepare analysis dataset
analysis_df = prepare_analysis_dataset(data, firm_cea, composite_cea)

# Initialize
causal = CausalAmbiguityAnalysis(analysis_df)

# Run baseline OLS
ols_results = causal.baseline_panel_ols(
    dependent_var='forward_return',
    ambiguity_var='CEA',
    green_var='Green_Dummy'
)

print(f"OLS β_CEA: {ols_results['params']['CEA']:.4f}")
print(f"t-stat: {ols_results['tstats']['CEA']:.2f}")
print(f"p-value: {ols_results['pvalues']['CEA']:.4f}")

# Run 2SLS
instruments = {
    'peer_ambiguity': calculate_peer_ambiguity(analysis_df),
    'epu': data['epu_series'],
    'policy_sensitivity': 'policy_sensitivity'
}

iv_results = causal.instrumental_variables_2sls(
    dependent_var='forward_return',
    endogenous_var='CEA',
    instruments=instruments
)
```

### Step 5: Run Complete Pipeline

```python
from main_china_energy import ChinaEnergyPipeline

# Initialize
pipeline = ChinaEnergyPipeline(
    data_path='data/',
    output_path='output/'
)

# Run everything
pipeline.run_full_pipeline()
```

### Customizing the Pipeline

**Change date range**:
```python
# In main_china_energy.py, modify the run_full_pipeline method
data_dict = prepare_analysis_data(
    data_loader=self.data_loader,
    start_date='2019-01-01',  # Custom start
    end_date='2023-12-31'      # Custom end
)
```

**Filter stocks by sector**:
```python
# Get all green energy stocks
green_stocks = [
    s for s, t in data['energy_classification'].items()
    if t == 'Green'
]

# Load only green stocks
data = prepare_analysis_data(
    data_loader=loader,
    stock_list=green_stocks
)
```

**Skip certain analyses**:
```python
# Run only steps 1-3
pipeline.load_data()
pipeline.compute_ambiguity_measures()
analysis_df = pipeline.prepare_analysis_dataset()

# Skip to visualization
pipeline.visualize_results()
```

---

## 7. Data Requirements

### Primary Data: High-Frequency Stock Prices

**Source**: PostgreSQL database `stock_hf` at 10.28.255.30

**Required Fields**:
- `code`: Stock identifier (VARCHAR)
- `closeprice`: Closing price (DECIMAL)
- `datetime`: Timestamp (TIMESTAMP)

**Frequency**:
- Preferred: 1-minute intervals
- Minimum: 5-minute intervals
- Format: YYYY-MM-DD HH:MM:SS

**Trading Hours** (Chinese A-share):
- Morning: 9:30-11:30
- Afternoon: 13:00-15:00
- Lunch break excluded

**Data Quality Checks**:
```python
# Check for missing data
missing_pct = returns.isnull().sum() / len(returns) * 100
print(f"Missing data: {missing_pct}%")

# Check for outliers
z_scores = (returns - returns.mean()) / returns.std()
outliers = (np.abs(z_scores) > 5).sum()
print(f"Outliers (>5 std): {outliers}")

# Check limit days
limit_days = loader.load_limit_days(stock_list)
total_limits = sum(len(dates) for dates in limit_days.values())
print(f"Limit days: {total_limits}")
```

### Secondary Data: External Variables

#### 1. Economic Policy Uncertainty (EPU)

**File**: `data/epu_china.csv`

**Format**:
```csv
date,epu_value
2018-01-01,120.5
2018-01-02,125.3
...
```

**Source**: www.policyuncertainty.com/china_monthly.html

**Frequency**: Monthly (forward-filled to daily)

#### 2. Policy Sensitivity

**File**: `data/policy_sensitivity.csv`

**Format**:
```csv
date,stock_id,policy_sensitivity
2018-01-01,601857.SH,0.25
2018-01-01,300750.SZ,0.45
...
```

**Calculation**:
```
policy_sensitivity = (gov_subsidies + gov_contracts) / total_revenue
```

**Range**: 0-1 (higher = more policy-dependent)

#### 3. Geopolitical Indicators

**File**: `data/geopolitical_indices.csv`

**Format**:
```csv
date,time,defense_index_return,gold_futures_return
2018-01-01,09:30:00,0.001,0.0005
...
```

**Indicators**:
- CSI National Defense Index (399967.SZ)
- SHFE Gold Futures

#### 4. Control Variables

**INE Crude Oil Futures** (`ine_crude_oil.csv`):
```csv
date,return
2018-01-01,0.02
...
```

**China ETS** (`china_ets.csv`):
```csv
date,return
2021-07-16,0.01
...
```

### Limit Days Data

**File**: `data/limit_days.csv` (optional)

**Format**:
```csv
date,stock_id,limit_type
2020-03-09,601857.SH,limit_up
2020-03-10,300750.SZ,limit_down
...
```

**Note**: If not provided, will be detected from database

---

## 8. Output Files

### Directory Structure

```
output/
├── figures/
│   ├── cea_timeseries.png
│   ├── cea_distribution.png
│   ├── pca_components.png
│   ├── policy_shocks.png
│   └── results_summary.png
├── tables/
│   ├── ols_results.csv
│   ├── iv_results.csv
│   ├── did_results.csv
│   └── summary_statistics.csv
└── reports/
    ├── analysis_report.txt
    └── hypothesis_tests.txt
```

### Figure Descriptions

#### 1. `cea_timeseries.png`
**Content**: Time series of CEA by energy type
- Three panels: Brown, Green, Grey
- Red vertical lines: Policy shock dates
- X-axis: Date
- Y-axis: Average CEA

**Interpretation**:
- Rising lines = Increasing ambiguity
- Spikes at policy dates = Market reaction
- Divergence = Differential impact

#### 2. `cea_distribution.png`
**Content**: Kernel density plots of CEA distribution
- Three colored distributions: Brown, Green, Grey
- X-axis: CEA value
- Y-axis: Density

**Interpretation**:
- Rightward shift = Higher average ambiguity
- Wider distribution = More volatility
- Overlap = Similar ambiguity levels

#### 3. `pca_components.png`
**Content**: PCA analysis of composite ambiguity
- Left: Scree plot (variance by component)
- Right: Cumulative variance

**Interpretation**:
- First PC = Systematic ambiguity factor
- High variance = Common factor explains most variation
- Used for composite CEA construction

#### 4. `policy_shocks.png`
**Content**: Event study around 4 policy dates
- 4 panels (one per policy date)
- Two lines: Brown vs. Green
- X-axis: Days relative to event
- Y-axis: CEA

**Interpretation**:
- Divergence at event = Differential impact
- Pre-trend check = Parallel trends assumption
- Post-event convergence = Reversion

#### 5. `results_summary.png`
**Content**: Summary of all hypothesis tests
- 4 panels: OLS vs. 2SLS, DiD, Mediation, Moderation
- Bar charts with error bars
- Significance stars

**Interpretation**:
- Positive β = Ambiguity premium (H1 supported)
- Negative interaction = Green discount (H2 supported)
- Significant DiD = Policy effects (H3 supported)
- Indirect effect > 0 = Liquidity channel (H4 supported)

### Report Files

#### `analysis_report.txt`

**Structure**:
```
================================================================================
CHINA ENERGY MARKET AMBIGUITY ANALYSIS REPORT
================================================================================

1. DATA SUMMARY
   - Date range: ...
   - Total observations: ...
   - Brown/Green/Grey counts: ...

2. AMBIGUITY MEASURES SUMMARY
   - Firm-level CEA mean: ...
   - Sector CEA means: ...
   - Composite CEA statistics: ...

3. CAUSAL ANALYSIS RESULTS
   3.1 Baseline OLS
       - CEA coefficient: ...
       - t-statistic: ...
       - p-value: ...

   3.2 2SLS
       - First stage F-stat: ...
       - Second stage coefficient: ...

   3.3 DiD
       - DiD estimator: ...
       - 95% CI: [...]

   3.4 Mediation
       - Direct effect: ...
       - Indirect effect: ...

   3.5 Moderation
       - Main effect: ...
       - Interaction effect: ...

4. HYPOTHESIS TESTING SUMMARY
   H1 (Ambiguity Premium): SUPPORTED/NOT SUPPORTED
   H2 (Green Discount): SUPPORTED/NOT SUPPORTED
   H3 (Policy Shocks): SUPPORTED/NOT SUPPORTED
   H4 (Liquidity Channel): SUPPORTED/NOT SUPPORTED
   H5 (Regime Dependence): SUPPORTED/NOT SUPPORTED
```

---

## 9. Troubleshooting

### Common Issues and Solutions

#### Issue 1: Database Connection Failed

**Symptoms**:
```
⚠ Warning: Could not connect to database: connection refused
⚠ Warning: No database credentials provided.
```

**Solutions**:
1. **Check VPN**: Are you connected to the institutional VPN?
   ```bash
   ping 10.28.255.30
   ```
   If timeout, connect to VPN first.

2. **Check Credentials**: Verify username/password
   ```python
   print(os.environ.get('DB_USER'))  # Should print your username
   ```

3. **Check Firewall**: Is port 5432 open?
   ```bash
   telnet 10.28.255.30 5432
   ```

4. **Test with psql**:
   ```bash
   psql -h 10.28.255.30 -U your_username -d stock_hf
   ```

#### Issue 2: Memory Error

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solutions**:
1. **Reduce date range**:
   ```python
   returns = loader.load_intraday_returns_from_db(
       start_date='2022-01-01',  # Instead of 2018
       end_date='2022-12-31'
   )
   ```

2. **Reduce stock list**:
   ```python
   stocks = stock_universe[:50]  # Only first 50 stocks
   ```

3. **Process in chunks**:
   ```python
   for date_range in [('2020', '2021'), ('2022', '2023')]:
       data = load_data_for_period(date_range)
       analyze(data)
   ```

4. **Use lower frequency**:
   ```python
   # Resample to 5-minute
   returns = returns.resample('5min').last().diff()
   ```

#### Issue 3: Slow Query Performance

**Symptoms**:
```
Query taking > 10 minutes...
```

**Solutions**:
1. **Add date filters**:
   ```sql
   -- Always filter by date
   WHERE datetime >= '2020-01-01' AND datetime <= '2020-12-31'
   ```

2. **Request database indexes**:
   ```sql
   -- Ask DB admin to create
   CREATE INDEX idx_datetime ON stock_hf(datetime);
   CREATE INDEX idx_code_datetime ON stock_hf(code, datetime);
   ```

3. **Use LIMIT for testing**:
   ```python
   test_stocks = stock_universe[:10]  # Test with 10 stocks
   ```

#### Issue 4: Module Not Found

**Symptoms**:
```
ModuleNotFoundError: No module named 'xxx'
```

**Solutions**:
1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install missing package**:
   ```bash
   pip install xxx
   ```

3. **Reinstall all dependencies**:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

#### Issue 5: Weak Instruments in 2SLS

**Symptoms**:
```
First stage F-statistic: 3.2 (should be > 10)
```

**Solutions**:
1. **Combine instruments**:
   ```python
   instruments = {
       'peer_ambiguity': peer_cea,
       'epu_x_sensitivity': epu * policy_sens,
       'geopolitical': geo_cea
   }
   ```

2. **Use alternative instrument**:
   ```python
   # Try industry-level CEA instead of peer
   industry_cea = calculate_industry_cea(returns)
   ```

3. **Check instrument relevance**:
   ```python
   # Correlation between instrument and endogenous variable
   print(instrument.corr(endogenous_var))
   # Should be > 0.3
   ```

#### Issue 6: Parallel Trends Violation in DiD

**Symptoms**:
```
Pre-treatment trends are not parallel
```

**Solutions**:
1. **Check pre-trends visually**:
   ```python
   # Plot pre-period trends
   pre_period = analysis_df[analysis_df['date'] < shock_date]
   ```

2. **Use different control group**:
   ```python
   # Try Grey energy as control instead of Brown
   ```

3. **Add time trends**:
   ```python
   # Include time fixed effects
   analysis_df['time_trend'] = range(len(analysis_df))
   ```

#### Issue 7: Data Type Errors

**Symptoms**:
```
TypeError: unsupported operand type(s) for /: 'str' and 'float'
```

**Solutions**:
1. **Check data types**:
   ```python
   print(returns.dtypes)
   # Should be float64, not object
   ```

2. **Convert to numeric**:
   ```python
   returns = returns.apply(pd.to_numeric, errors='coerce')
   ```

3. **Handle string dates**:
   ```python
   analysis_df['date'] = pd.to_datetime(analysis_df['date'])
   ```

### Getting Help

If issues persist:

1. **Check the logs**:
   ```python
   import logging
   logging.basicConfig(filename='analysis.log', level=logging.DEBUG)
   ```

2. **Verify data manually**:
   ```python
   # Check database directly
   psql -h 10.28.255.30 -U user -d stock_hf -c "SELECT COUNT(*) FROM stock_hf;"
   ```

3. **Contact database administrator**:
   - Server: 10.28.255.30
   - Database: stock_hf
   - Issue: [description]

4. **Consult documentation**:
   - Paper: `causal_ambi_china.tex`
   - Proposal: `energy_industry_proposal.md`
   - This README

---

## 10. Theoretical Background

### Ambiguity vs. Risk

**Risk** (Knightian Risk):
- Known probabilities
- Example: 50% chance of rain
- Measured by variance, standard deviation

**Ambiguity** (Knightian Uncertainty):
- Unknown probabilities
- Example: "I don't know the probability of rain"
- Measured by CEA (Cross-Entropy Ambiguity)

### The CEA Measure

**Cross-Entropy Ambiguity (CEA)**:
```
CEA_t = KL(P_t || Q_t) = Σ_x P_t(x) log(P_t(x) / Q_t(x))
```

**Intuition**:
- P_t: Actual return distribution today
- Q_t: Expected return distribution (benchmark)
- CEA: How "surprised" we are by today's distribution
- High CEA = High model uncertainty

**Why Intraday Data?**:
- Intraday returns contain rich distributional information
- Daily aggregation loses information
- Ambiguity manifests in distribution shape changes

### Theoretical Mechanisms

**Mechanism 1: Ambiguity Premium**
1. Ambiguity ↑ → Model uncertainty ↑
2. Investors demand higher expected return
3. Current price ↓ to accommodate higher discount rate
4. Future realized return ↑

**Mechanism 2: Liquidity Channel**
1. Ambiguity ↑ → Market maker models fail
2. MMs widen spreads or withdraw
3. Liquidity ↓ → Transaction costs ↑
4. Prices ↓ → Future returns ↑

### Five Hypotheses

**H1: Ambiguity Premium**
- Prediction: β_CEA > 0
- Test: Panel regression of returns on lagged CEA
- Equation: (10) in paper

**H2: Green Discount**
- Prediction: β_Interaction < 0
- Test: CEA × Green dummy interaction
- Rationale: Policy support reduces green energy ambiguity

**H3: Policy Shocks**
- Prediction: DiD estimator significant
- Test: Difference-in-Differences around policy dates
- Treatment: Green energy, Control: Brown energy

**H4: Liquidity Channel**
- Prediction: Indirect effect > 0
- Test: Mediation analysis
- Mediator: Bid-ask spread or turnover

**H5: Regime Dependence**
- Prediction: Effects vary by market conditions
- Test: Moderation by EPU levels, volatility regime

### Econometric Specifications

**Equation (10) - Baseline OLS**:
```
r_{i,t+1} = α + β₁CEA_{i,t} + β₂(CEA_{i,t} × Green_i) + γX_{i,t} + FE + ε_{i,t+1}
```

**Equations (11-12) - 2SLS**:
```
First stage:  CEA_{i,t} = π₀ + π₁Z_{i,t} + π₂X_{i,t} + FE + υ_{i,t}
Second stage: r_{i,t+1} = α + β₁CEÂ_{i,t} + β₂Green_i + γX_{i,t} + FE + ε_{i,t+1}
```

Instruments (Z):
- Peer CEA (industry average excluding stock i)
- EPU × Policy Sensitivity
- Geopolitical CEA

**Equation (13) - DiD**:
```
CEA_{i,t} = α + β₁Post_t × Green_i + β₂Post_t + β₃Green_i + FE + ε_{i,t}
```

**Equations (14-15) - Mediation**:
```
(14) Mediator_{i,t} = α + a·CEA_{i,t} + Controls
(15) r_{i,t+1} = β + c'·CEA_{i,t} + b·Mediator_{i,t} + Controls
```

Indirect effect = a × b
Direct effect = c'

**Equation (16) - Moderation**:
```
r_{i,t+1} = α + β₁CEA_{i,t} + β₂(CEA_{i,t} × Green_i) + γX_{i,t} + FE + ε_{i,t+1}
```

---

## Appendix A: Sample SQL Queries

### Query 1: Get All Energy Stocks
```sql
SELECT DISTINCT code
FROM stock_hf
WHERE code LIKE '601%'  -- Shanghai energy stocks
   OR code LIKE '600%'
   OR code LIKE '300%'  -- ChiNext (green energy)
ORDER BY code;
```

### Query 2: Get Daily Returns for One Stock
```sql
WITH daily_prices AS (
    SELECT
        code,
        DATE(datetime) as date,
        closeprice
    FROM stock_hf
    WHERE code = '601857.SH'
      AND datetime >= '2020-01-01'
      AND datetime <= '2020-12-31'
    GROUP BY code, DATE(datetime), closeprice
),
daily_returns AS (
    SELECT
        code,
        date,
        closeprice,
        LAG(closeprice) OVER (ORDER BY date) as prev_close,
        (closeprice / LAG(closeprice) OVER (ORDER BY date) - 1) as daily_return
    FROM daily_prices
)
SELECT * FROM daily_returns WHERE prev_close IS NOT NULL;
```

### Query 3: Detect Limit Days
```sql
WITH daily_returns AS (
    SELECT
        code,
        DATE(datetime) as date,
        FIRST_VALUE(closeprice) OVER (
            PARTITION BY code, DATE(datetime)
            ORDER BY datetime
        ) as open_price,
        LAST_VALUE(closeprice) OVER (
            PARTITION BY code, DATE(datetime)
            ORDER BY datetime
        ) as close_price,
        LAG(closeprice) OVER (
            PARTITION BY code
            ORDER BY datetime
        ) as prev_close
    FROM stock_hf
    WHERE datetime >= '2020-01-01'
)
SELECT
    code,
    date,
    (close_price - prev_close) / prev_close as daily_return,
    CASE
        WHEN (close_price - prev_close) / prev_close >= 0.095 THEN 'limit_up'
        WHEN (close_price - prev_close) / prev_close <= -0.095 THEN 'limit_down'
        ELSE 'normal'
    END as limit_type
FROM daily_returns
WHERE prev_close IS NOT NULL
  AND ABS((close_price - prev_close) / prev_close) >= 0.095;
```

---

## Appendix B: Key Functions Reference

### data_loader.py

| Function | Purpose | Returns |
|----------|---------|---------|
| `ChinaEnergyDataLoader.__init__()` | Initialize with DB credentials | DataLoader object |
| `load_stock_universe_from_db()` | Get all available stocks | List of stock codes |
| `load_intraday_returns_from_db()` | Main data loading function | DataFrame (datetime × stocks) |
| `classify_energy_stocks()` | Categorize by energy type | Dict {stock: Brown/Green/Grey} |
| `load_limit_days()` | Get limit-up/down days | Dict {stock: [dates]} |
| `load_epu_data()` | Get policy uncertainty | Series (dates) |
| `load_policy_sensitivity()` | Get firm policy dependence | DataFrame (dates × stocks) |
| `load_geopolitical_data()` | Get defense/gold returns | Dict {defense, gold} |
| `load_control_data()` | Get oil/carbon prices | Dict {oil, ets} |

### ambiguity_measurement.py

| Function | Purpose | Returns |
|----------|---------|---------|
| `compute_ambiguity_for_stock()` | Single-stock CEA | Series (daily CEA) |
| `compute_ambiguity_cross_section()` | Multi-stock CEA | DataFrame (dates × stocks) |
| `compute_sector_ambiguity()` | Sector-level CEA | Dict {Brown, Green, Grey} |
| `compute_composite_ambiguity()` | PCA-based CEA | Series (systematic factor) |
| `compute_policy_ambiguity()` | Policy-based CEA | Series (daily) |
| `compute_geopolitical_ambiguity()` | Geo-based CEA | Series (daily) |

### causal_analysis.py

| Function | Purpose | Returns |
|----------|---------|---------|
| `baseline_panel_ols()` | Test H1 (ambiguity premium) | Regression results |
| `instrumental_variables_2sls()` | Address endogeneity | IV results |
| `difference_in_differences()` | Test H3 (policy shocks) | DiD estimator |
| `mediation_analysis()` | Test H4 (liquidity channel) | Direct/indirect effects |
| `moderation_analysis()` | Test H2/H5 (moderation) | Interaction effects |
| `granger_causality_test()` | Test temporal precedence | F-statistics |

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **CEA** | Cross-Entropy Ambiguity - Our measure of model uncertainty |
| **Brown Energy** | Traditional fossil fuels (coal, oil, gas) |
| **Green Energy** | Renewables (solar, wind, EV, batteries) |
| **Grey Energy** | Utilities and grid infrastructure |
| **DiD** | Difference-in-Differences - Causal inference method |
| **2SLS** | Two-Stage Least Squares - IV estimation |
| **EPU** | Economic Policy Uncertainty Index |
| **PCA** | Principal Component Analysis - Dimensionality reduction |
| **IV** | Instrumental Variable - Exogenous variation |
| **KL Divergence** | Kullback-Leibler divergence - Information theory measure |
| **Limit Day** | Day when stock hits daily price limit (±10%) |
| **Dual Carbon** | China's 2030 peak carbon, 2060 carbon neutrality goals |
| **A-share** | Chinese domestic stocks (Shanghai + Shenzhen) |
| **ChiNext** | ChiNext - China's NASDAQ-style board |
| **STAR Market** | Shanghai Tech Innovation Board |

---

## Contact & Support

For questions or issues:
1. Check this README first
2. Review the paper: `causal_ambi_china.tex`
3. Check code comments (very detailed!)
4. Contact database admin for server issues

**Database**: 10.28.255.30 / stock_hf
**Paper**: `draft/causal_ambi_china.tex`
**Proposal**: `draft/energy_industry_proposal.md`

---

## Version History

- **v1.0** (2024): Initial release with PostgreSQL database connection
- **v1.1** (2024): Added detailed comments and documentation
- **v1.2** (2024): Comprehensive README with troubleshooting

---

## License

This code is part of the research project "Pricing the Unknown: Ambiguity Premiums in China's Green vs. Brown Energy Markets". Use with proper citation.

---

**Last Updated**: 2024
**Maintained By**: Research Team
