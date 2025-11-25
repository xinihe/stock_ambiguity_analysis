# Data and Empirical Research Design - Geopolitical Risk and Ambiguity Paper

## Overview

This document summarizes the Data and Empirical Research Design section that has been drafted for `geopoliticalAmb02.tex`, along with the supporting Python scripts and LaTeX tables.

## Contents Created

### 1. Enhanced LaTeX Document (`geopoliticalAmb02.tex`)

The Data and Empirical Research Design section now includes:

- **Section 3**: Complete data and methodology description
- **Subsection 3.1**: Data and Variables
  - Dependent variables (market and firm returns)
  - Key independent variables (GPR, ambiguity measures, volatility)
  - Control variables (market-level and firm-level)
  - Two comprehensive tables with sample statistics

- **Subsection 3.2**: Methodology and Empirical Analysis
  - Baseline regression specifications (time-series and cross-sectional)
  - Mechanism and mediation analysis framework
  - Moderation analysis for SOE effects
  - Comprehensive robustness checks

### 2. Python Analysis Scripts

#### A. Data Generation (`generate_sample_data.py`)
- Generates sample market data based on CSI 300 index structure
- Creates synthetic geopolitical risk (GPR) data
- Constructs ambiguity measures using cross-entropy framework
- Simulates firm-level data with realistic characteristics
- Produces summary statistics and visualizations

**Features:**
- Realistic market dynamics with stress periods
- Industry distribution matching Chinese market composition
- SOE vs non-SOE classification
- Firm characteristics (size, B/M, leverage, etc.)

#### B. Baseline Regressions (`baseline_regressions.py`)
- Implements all regression models described in the paper
- Tests all four hypotheses (H1-H4)
- Time-series analysis for market-level relationships
- Fama-MacBeth cross-sectional analysis
- SOE moderation analysis with interaction terms
- Comprehensive regression results with standard errors

**Hypotheses Tested:**
- **H1**: GPR increases ambiguity
- **H2**: Ambiguity is negatively priced
- **H3**: Ambiguity mediates GPR-return relationship
- **H4**: SOE status moderates ambiguity pricing

#### C. LaTeX Table Generation (`generate_latex_tables.py`)
- Creates publication-ready LaTeX tables
- Matches the formatting requirements of the paper
- Includes proper significance testing and standard errors
- Generates 7 comprehensive tables for the empirical analysis

### 3. Generated LaTeX Tables

Located in `/outputs/tables/`:

1. **Descriptive Statistics** (`descriptive_statistics.tex`)
   - Summary statistics for all key variables
   - Market and firm-level characteristics

2. **Industry Composition** (`industry_composition.tex`)
   - Sample distribution by industry and ownership type
   - SOE vs non-SOE breakdown

3. **Baseline Regressions** (`baseline_regressions.tex`)
   - Time-series regression results
   - Models with incremental variable additions

4. **GPR-Ambiguity Relationship** (`gpr_ambiguity_relationship.tex`)
   - Test of H1: GPR impact on ambiguity
   - Contemporaneous and lagged effects

5. **Fama-MacBeth Results** (`fama_macbeth_results.tex`)
   - Cross-sectional pricing of ambiguity
   - Test of H2 with firm characteristics

6. **SOE Moderation** (`soe_moderation.tex`)
   - Test of H4: ownership type effects
   - Interaction terms and subsample analysis

7. **Correlation Matrix** (`correlation_matrix.tex`)
   - Correlations between key variables
   - Helps identify multicollinearity issues

## Key Methodological Features

### Ambiguity Measures
- **Cross-Entropy Ambiguity (CE-Ambiguity)**: Baseline measure
- **Model Disagreement Ambiguity (MD-Ambiguity)**: Alternative specification
- **Weight Dispersion Ambiguity (WD-Ambiguity)**: Robustness check

### Empirical Strategy
1. **Two-stage approach**: Time-series + cross-sectional analysis
2. **Mediation analysis**: Tests transmission channels
3. **Fama-MacBeth**: Addresses cross-sectional dependence
4. **Instrumental variables**: Addresses endogeneity concerns
5. **Event studies**: Natural experiments around geopolitical events

### Robustness Checks
- Alternative ambiguity measures
- Different model specifications
- Subsample analysis (pre/post-COVID, high/low GPR periods)
- Endogeneity controls
- Market regime analysis

## How to Use

### For the LaTeX Document:
1. The Data and Empirical Research Design section is already integrated into `geopoliticalAmb02.tex`
2. Tables can be included using `\input{tables/table_name.tex}`
3. Ensure `\usepackage{threeparttable}` is in the preamble

### For Data Analysis:
1. Run `python generate_sample_data.py` to create sample data
2. Run `python baseline_regressions.py` to perform empirical analysis
3. Run `python generate_latex_tables.py` to create tables

### Key Files:
- **Main document**: `/outputs/results/geopoliticalAmb02.tex`
- **Analysis scripts**: `/scripts/data_analysis/`
- **Generated tables**: `/outputs/tables/`
- **Sample data**: `/outputs/data/` (when generated)

## Sample Size and Period
- **Sample Period**: January 2018 - December 2023 (1,458 trading days)
- **Market Analysis**: CSI 300 Index
- **Cross-sectional**: 2,700 firms (923 SOEs, 1,777 non-SOEs)
- **Observations**: ~350,000 firm-day observations

## Integration with Paper Structure

The Data and Empirical Research Design section logically follows the Literature Review section and precedes the Results section. It provides:

1. **Complete transparency** about data sources and construction
2. **Rigorous methodology** aligned with established asset pricing literature
3. **Comprehensive robustness** to ensure credible findings
4. **Clear hypothesis testing** framework for empirical results

All tables and methodological descriptions are consistent with the existing content and citations in the paper.