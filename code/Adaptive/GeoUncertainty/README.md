# GeoUncertainty Research Project

## Overview

This research project focuses on creating a combined global uncertainty index that links global systemic uncertainty to adaptive ambiguity aversion in the Chinese capital market. The project addresses methodological challenges including data frequency mismatch and multicollinearity risks by constructing a composite index that combines climate risk and geopolitical risk dimensions.

## Project Structure

```
GeoUncertainty/
├── README.md                    # This file - project documentation
├── scripts/                     # All Python scripts organized by functionality
│   ├── data_processing/         # Data extraction, cleaning, and processing scripts
│   │   ├── extract_data.py      # Extract climate risk and GPR data from Excel files
│   │   ├── clean_and_document_data.py  # Data cleaning and documentation
│   │   ├── filter_gpr_data.py   # Filter and process geopolitical risk data
│   │   └── create_combined_uncertainty_index.py  # Create composite uncertainty index
│   └── analysis/                # Analysis and correlation scripts
│       └── analyze_correlations.py  # Correlation analysis and visualization
├── data/                        # All data files organized by processing stage
│   ├── raw/                     # Original extracted data files
│   │   ├── climate_risk_series_daily.csv  # Daily climate risk data
│   │   └── gpr_countries_data.csv         # GPR data for selected countries
│   └── processed/               # Cleaned and processed data files
│       ├── climate_risk_series_daily_clean.csv  # Cleaned daily climate risk data
│       ├── gpr_countries_data_filtered.csv      # Filtered GPR country data
│       ├── combined_global_uncertainty_index.csv # Final composite index
│       └── rolling_correlations.csv             # Rolling correlation results
├── outputs/                     # Analysis results and visualizations
│   ├── plots/                   # Generated plots and visualizations
│   │   ├── combined_uncertainty_index_plot.png  # Composite index visualization
│   │   ├── correlation_heatmap.png              # Correlation heatmap
│   │   ├── rolling_correlations.png             # Rolling correlation plots
│   │   └── scatter_matrix.png                   # Scatter matrix visualization
│   └── results/                 # Analysis results and intermediate outputs
│       └── analysis/            # Detailed analysis results
│           ├── ambiguity_correlations.png       # Ambiguity correlation analysis
│           ├── combined_data_analysis.csv       # Combined analysis results
│           ├── correlation_heatmap.png          # Additional correlation heatmap
│           ├── granger_causality.png            # Granger causality analysis
│           ├── regression_analysis.png          # Regression analysis results
│           └── time_series_plots.png            # Time series visualizations
├── documentation/               # Project documentation and methodology
│   ├── data_info.md            # Detailed data description and processing steps
│   └── midas.md                # MIDAS methodology and implementation details
└── trash/                      # Placeholder for obsolete files (currently empty)
```

## Methodology

### Core Objective
Robustly link global systemic uncertainty to adaptive ambiguity aversion index in the Chinese capital market through a composite uncertainty index.

### Key Challenges Addressed
1. **Data Frequency Mismatch**: Daily climate risk data vs. monthly geopolitical risk data
2. **Multicollinearity Risk**: Avoiding correlation issues in state transition equations

### Data Sources
- **Climate Risk Data**: Physical risk, transition risk, policy risk, and market sentiment risk components
- **Geopolitical Risk Data**: Country-specific GPR data for China, Hong Kong SAR, Japan, and US

## Workflow

### 1. Data Extraction and Processing
- **Script**: `scripts/data_processing/extract_data.py`
- **Input**: Excel files from `/data/` directory (Climate_Risk_Index.xlsx, data_gpr_export.xls)
- **Output**: Raw CSV files in `data/raw/`

### 2. Data Cleaning and Filtering
- **Scripts**: 
  - `scripts/data_processing/clean_and_document_data.py`
  - `scripts/data_processing/filter_gpr_data.py`
- **Input**: Raw data files
- **Output**: Cleaned data files in `data/processed/`

### 3. Combined Index Creation
- **Script**: `scripts/data_processing/create_combined_uncertainty_index.py`
- **Methodology**: MIDAS-inspired time-weighting scheme with normalization
- **Output**: `data/processed/combined_global_uncertainty_index.csv`

### 4. Correlation Analysis
- **Script**: `scripts/analysis/analyze_correlations.py`
- **Output**: Correlation matrices, rolling correlations, and visualizations in `outputs/`

## Key Features

### MIDAS-Inspired Methodology
- Time-weighting scheme to incorporate recency effects
- Monthly aggregation of daily climate risk data
- Normalization for comparability between risk dimensions

### Comprehensive Analysis
- Static and rolling correlation analysis
- Granger causality testing
- Multiple visualization techniques
- Regression analysis capabilities

## Usage Instructions

### Prerequisites
- Python 3.x
- Required packages: pandas, numpy, matplotlib, seaborn

### Running the Analysis
1. **Data Extraction**: Run `scripts/data_processing/extract_data.py`
2. **Data Processing**: Execute cleaning and filtering scripts
3. **Index Creation**: Run `scripts/data_processing/create_combined_uncertainty_index.py`
4. **Analysis**: Execute `scripts/analysis/analyze_correlations.py`

### Output Interpretation
- **Plots**: Visual representations of uncertainty patterns and correlations
- **CSV Files**: Quantitative results for further analysis
- **Documentation**: Detailed methodology and data descriptions

## File Descriptions

### Data Processing Scripts
- **extract_data.py**: Extracts and formats raw data from Excel sources
- **clean_and_document_data.py**: Performs data cleaning and creates documentation
- **filter_gpr_data.py**: Filters geopolitical risk data for relevant countries
- **create_combined_uncertainty_index.py**: Implements MIDAS methodology for composite index

### Analysis Scripts
- **analyze_correlations.py**: Comprehensive correlation and causality analysis

### Data Files
- **Raw Data**: Original extracted data maintaining source format
- **Processed Data**: Cleaned, filtered, and normalized data ready for analysis
- **Results**: Final composite indices and analysis outputs

### Documentation
- **data_info.md**: Comprehensive data description and processing documentation
- **midas.md**: Detailed methodology explanation and implementation notes

## Research Applications

This organized structure supports:
- **Academic Research**: Clear methodology and reproducible results
- **Policy Analysis**: Systematic uncertainty measurement for decision-making
- **Risk Management**: Composite risk indicators for financial markets
- **Further Development**: Modular structure for methodology extensions

## Maintenance Notes

- **Trash Folder**: Reserved for obsolete files during project evolution
- **Version Control**: Organized structure facilitates git tracking
- **Scalability**: Modular design supports additional data sources and methodologies
- **Documentation**: Comprehensive documentation ensures reproducibility

## Contact and Contributions

This project structure facilitates collaborative research and methodology development. Each component is documented and organized for easy understanding and extension.