import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataAnalysisAndCausality:
    def __init__(self):
        self.sse_data_path = '/Users/tlxy/Research/Ambiguity/data/SSE.000300.csv'
        self.ambiguity_data_path = '/Users/tlxy/Research/Ambiguity/code/Entropy/daily_ambiguity_risk_metrics_2.csv'
        self.uncertainty_data_path = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/combined_global_uncertainty_index.csv'
        self.climate_risk_path = '/Users/tlxy/Research/Ambiguity/data/Climate_Risk_Index.xlsx'
        self.gpr_data_path = '/Users/tlxy/Research/Ambiguity/data/data_gpr_export.xls'
        # Move output to GeoUncertainty folder as requested
        self.output_dir = '/Users/tlxy/Research/Ambiguity/code/Adaptive/GeoUncertainty/analysis'
        self.output_csv_path = f'{self.output_dir}/combined_data_analysis.csv'
        self.combined_data = None
        
        # Set English as default language for plots
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    
    def load_data(self):
        """Load and preprocess all datasets with robust error handling"""
        print("Loading data...")
        
        # Initialize all dataframes to None
        sse_df = None
        ambiguity_df = None
        uncertainty_df = None
        climate_df = None
        gpr_df = None
        
        # Load SSE300 data and convert to daily
        print("Loading SSE300 data and converting to daily...")
        try:
            sse_df = pd.read_csv(self.sse_data_path)
            # Try different datetime column names
            datetime_cols = ['datetime', 'date', 'Date', 'DATE', 'time', 'Time']
            datetime_col = None
            for col in datetime_cols:
                if col in sse_df.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                sse_df['date'] = pd.to_datetime(sse_df[datetime_col], errors='coerce').dt.date
                sse_df = sse_df.dropna(subset=['date'])
                
                # Group by date
                sse_df = sse_df.groupby('date').agg({
                    'SSE.000300.open': 'first',
                    'SSE.000300.close': 'last',
                    'SSE.000300.high': 'max',
                    'SSE.000300.low': 'min',
                    'SSE.000300.volume': 'sum'
                }).reset_index()
                
                # Calculate daily returns with error handling
                if 'SSE.000300.open' in sse_df.columns and 'SSE.000300.close' in sse_df.columns:
                    # Avoid division by zero
                    sse_df['daily_return'] = np.where(
                        sse_df['SSE.000300.open'] != 0,
                        (sse_df['SSE.000300.close'] - sse_df['SSE.000300.open']) / sse_df['SSE.000300.open'],
                        0
                    )
                    sse_df = sse_df[['date', 'daily_return', 'SSE.000300.close']]
                    print(f"SSE300 data loaded successfully with {len(sse_df)} rows")
                else:
                    print("Warning: Missing required price columns in SSE300 data")
            else:
                print("Warning: No datetime column found in SSE300 data")
        except Exception as e:
            print(f"Error loading SSE300 data: {e}")
        
        # Load ambiguity data with robust handling
        print("Loading ambiguity data...")
        try:
            ambiguity_df = pd.read_csv(self.ambiguity_data_path)
            # Convert date with error handling
            if 'date' in ambiguity_df.columns:
                ambiguity_df['date'] = pd.to_datetime(ambiguity_df['date'], errors='coerce').dt.date
                ambiguity_df = ambiguity_df.dropna(subset=['date'])
                print(f"Ambiguity data loaded successfully with {len(ambiguity_df)} rows")
            else:
                print("Warning: No date column found in ambiguity data")
                ambiguity_df = None
        except Exception as e:
            print(f"Error loading ambiguity data: {e}")
            ambiguity_df = None
        
        # Load uncertainty data with robust handling
        print("Loading uncertainty data and converting to daily...")
        try:
            uncertainty_df = pd.read_csv(self.uncertainty_data_path)
            # Try different date column names
            date_cols = ['Date', 'date', 'DATE', 'time', 'Time']
            date_col = None
            for col in date_cols:
                if col in uncertainty_df.columns:
                    date_col = col
                    break
            
            if date_col:
                uncertainty_df['date'] = pd.to_datetime(uncertainty_df[date_col], errors='coerce').dt.date
                uncertainty_df = uncertainty_df.dropna(subset=['date'])
                
                # Select available columns
                selected_cols = ['date']
                for col in ['Correlation_Adjusted_Index', 'Equal_Weighted_Index', 'Climate_Risk_Component', 'GPR_Component']:
                    if col in uncertainty_df.columns:
                        selected_cols.append(col)
                
                uncertainty_df = uncertainty_df[selected_cols]
                print(f"Uncertainty data loaded successfully with {len(uncertainty_df)} rows")
            else:
                print("Warning: No date column found in uncertainty data")
                uncertainty_df = None
        except Exception as e:
            print(f"Error loading uncertainty data: {e}")
            uncertainty_df = None
        
        # Load climate risk data with improved handling
        print("Loading climate risk data...")
        try:
            # Try to read with different approaches
            try:
                # Skip potential header rows
                climate_df = pd.read_excel(self.climate_risk_path, skiprows=1)
            except:
                # Try without skipping rows
                climate_df = pd.read_excel(self.climate_risk_path)
            
            # Debug: Print first few rows and columns to understand structure
            print(f"Climate data columns: {climate_df.columns.tolist()}")
            print(f"Climate data sample (first 3 rows):\n{climate_df.head(3)}")
            
            # Try multiple date column detection strategies
            date_found = False
            # Try common date column names
            date_cols = ['Date', 'date', 'DATE', 'time', 'Time', 'Year', 'year', 'month', 'Month']
            for col in date_cols:
                if col in climate_df.columns:
                    try:
                        climate_df['date'] = pd.to_datetime(climate_df[col], errors='coerce')
                        if climate_df['date'].notna().sum() > 0:  # If any date was successfully parsed
                            climate_df['date'] = climate_df['date'].dt.date
                            date_found = True
                            print(f"Found date column: {col}")
                            break
                    except:
                        continue
            
            # If no date column found, try to parse first column
            if not date_found and not climate_df.empty:
                try:
                    climate_df['date'] = pd.to_datetime(climate_df.iloc[:, 0], errors='coerce')
                    if climate_df['date'].notna().sum() > 0:
                        climate_df['date'] = climate_df['date'].dt.date
                        date_found = True
                        print("Parsed date from first column")
                except:
                    print("Warning: Could not parse any date column in climate data")
            
            if date_found:
                climate_df = climate_df.dropna(subset=['date'])
                
                # Find climate risk index column by checking column names
                risk_cols = []
                for col in climate_df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['risk', 'index', 'climate', 'value', 'score']):
                        risk_cols.append(col)
                
                if risk_cols:
                    # Use the first relevant column found
                    climate_df = climate_df[['date', risk_cols[0]]].rename(columns={risk_cols[0]: 'Climate_Risk_Index'})
                    print(f"Found climate risk column: {risk_cols[0]} -> Climate_Risk_Index")
                    print(f"Climate data loaded successfully with {len(climate_df)} rows")
                else:
                    print("Warning: No climate risk column found")
                    climate_df = None
            else:
                climate_df = None
        except Exception as e:
            print(f"Error loading climate risk data: {e}")
            climate_df = None
        
        # Load GPR country data with improved handling
        print("Loading country-specific GPR data...")
        try:
            gpr_df = pd.read_excel(self.gpr_data_path)
            
            # Debug: Print columns to understand structure
            print(f"GPR data columns: {gpr_df.columns.tolist()}")
            
            # Handle date column
            if 'month' in gpr_df.columns:
                gpr_df['date'] = pd.to_datetime(gpr_df['month'], errors='coerce')
            elif 'date' in gpr_df.columns:
                gpr_df['date'] = pd.to_datetime(gpr_df['date'], errors='coerce')
            else:
                # Try first column as date
                gpr_df['date'] = pd.to_datetime(gpr_df.iloc[:, 0], errors='coerce')
            
            if gpr_df['date'].notna().sum() > 0:
                gpr_df['date'] = gpr_df['date'].dt.date
                gpr_df = gpr_df.dropna(subset=['date'])
                
                # Find all GPR country columns
                gpr_columns = {}
                country_mapping = {
                    'CHN': 'GPR_China',
                    'JPN': 'GPR_Japan',
                    'USA': 'GPR_US',
                    'US': 'GPR_US'
                }
                
                for col in gpr_df.columns:
                    col_upper = col.upper()
                    for country_code, country_name in country_mapping.items():
                        if country_code in col_upper and col_upper.startswith('GPR'):
                            gpr_columns[col] = country_name
                
                # Debug: Show found columns
                print(f"Found GPR columns: {gpr_columns}")
                
                # Select date and GPR columns
                if gpr_columns:
                    gpr_df = gpr_df[['date'] + list(gpr_columns.keys())].rename(columns=gpr_columns)
                    print(f"GPR data loaded successfully with {len(gpr_df)} rows")
                else:
                    print("Warning: No country-specific GPR columns found")
                    gpr_df = None
            else:
                print("Warning: Could not parse date column in GPR data")
                gpr_df = None
        except Exception as e:
            print(f"Error loading GPR data: {e}")
            gpr_df = None
        
        # Only proceed if we have at least SSE and ambiguity data
        if sse_df is None or ambiguity_df is None:
            print("Warning: Missing essential data (SSE or ambiguity). Cannot create date range.")
            self.combined_data = pd.DataFrame()
            return
        
        # Create a complete date range with error handling
        try:
            # Calculate valid date range from available dataframes
            min_dates = []
            max_dates = []
            
            if sse_df is not None and not sse_df.empty:
                min_dates.append(sse_df['date'].min())
                max_dates.append(sse_df['date'].max())
            
            if ambiguity_df is not None and not ambiguity_df.empty:
                min_dates.append(ambiguity_df['date'].min())
                max_dates.append(ambiguity_df['date'].max())
            
            if uncertainty_df is not None and not uncertainty_df.empty:
                min_dates.append(uncertainty_df['date'].min())
                max_dates.append(uncertainty_df['date'].max())
            
            if min_dates and max_dates:
                min_date = max(min_dates)
                max_date = min(max_dates)
                
                print(f"Creating date range from {min_date} to {max_date}")
                
                # Create date range
                date_range = pd.DataFrame({
                    'date': pd.date_range(start=min_date, end=max_date).date
                })
                
                # Merge all datasets with robust handling
                print("Merging datasets...")
                merged_df = date_range.merge(sse_df, on='date', how='left')
                merged_df = merged_df.merge(ambiguity_df, on='date', how='left')
                
                # Process uncertainty data (monthly to daily)
                if uncertainty_df is not None:
                    print("Processing uncertainty data to daily frequency...")
                    uncertainty_daily = []
                    for idx, row in merged_df.iterrows():
                        # Find the latest uncertainty data up to this date
                        month_uncertainty = uncertainty_df[uncertainty_df['date'] <= row['date']].sort_values('date', ascending=False)
                        if not month_uncertainty.empty:
                            month_uncertainty = month_uncertainty.iloc[0]
                            entry = {'date': row['date']}
                            for col in ['Correlation_Adjusted_Index', 'Equal_Weighted_Index', 'Climate_Risk_Component', 'GPR_Component']:
                                if col in month_uncertainty:
                                    entry[col] = month_uncertainty[col]
                                else:
                                    entry[col] = np.nan
                            uncertainty_daily.append(entry)
                        else:
                            uncertainty_daily.append({
                                'date': row['date'],
                                'Correlation_Adjusted_Index': np.nan,
                                'Equal_Weighted_Index': np.nan,
                                'Climate_Risk_Component': np.nan,
                                'GPR_Component': np.nan
                            })
                    
                    uncertainty_daily_df = pd.DataFrame(uncertainty_daily)
                    merged_df = merged_df.merge(uncertainty_daily_df, on='date', how='left')
                
                # Add climate risk data if available
                if climate_df is not None:
                    print("Adding climate risk data...")
                    merged_df = merged_df.merge(climate_df, on='date', how='left')
                
                # Add GPR country data if available
                if gpr_df is not None:
                    print("Adding GPR country data...")
                    # Forward fill GPR data to daily
                    gpr_daily = []
                    for idx, row in merged_df.iterrows():
                        # Find the latest GPR data up to this date
                        gpr_data = gpr_df[gpr_df['date'] <= row['date']].sort_values('date', ascending=False)
                        if not gpr_data.empty:
                            gpr_data = gpr_data.iloc[0]
                            gpr_entry = {'date': row['date']}
                            # Add all GPR country columns found
                            for col in gpr_df.columns:
                                if col != 'date':
                                    gpr_entry[col] = gpr_data[col]
                            gpr_daily.append(gpr_entry)
                        else:
                            gpr_entry = {'date': row['date']}
                            # Add all GPR country columns with NaN
                            for col in gpr_df.columns:
                                if col != 'date':
                                    gpr_entry[col] = np.nan
                            gpr_daily.append(gpr_entry)
                    
                    gpr_daily_df = pd.DataFrame(gpr_daily)
                    merged_df = merged_df.merge(gpr_daily_df, on='date', how='left')
            else:
                print("Warning: Could not determine valid date range")
                merged_df = pd.DataFrame()
        except Exception as e:
            print(f"Error creating date range or merging data: {e}")
            merged_df = pd.DataFrame()
        
        # Add transformed variables only if merged_df is not empty
        if not merged_df.empty:
            print("Adding enhanced variables for correlation analysis...")
            # Rolling statistics with error handling
            ambiguity_cols = [col for col in merged_df.columns if 'ambiguity' in col.lower()]
            correlation_cols = [col for col in merged_df.columns if 'correlation' in col.lower()]
            
            for window in [10, 20, 30]:
                # Add rolling statistics for ambiguity if column exists
                if ambiguity_cols:
                    main_ambiguity_col = ambiguity_cols[0]  # Use first ambiguity column found
                    try:
                        merged_df[f'ambiguity_rolling_std_{window}'] = merged_df[main_ambiguity_col].rolling(window=window).std()
                        merged_df[f'ambiguity_rolling_mean_{window}'] = merged_df[main_ambiguity_col].rolling(window=window).mean()
                    except:
                        print(f"Warning: Could not calculate rolling stats for ambiguity with window {window}")
                
                # Add rolling statistics for correlation index if column exists
                if correlation_cols:
                    main_corr_col = correlation_cols[0]  # Use first correlation column found
                    try:
                        merged_df[f'correlation_index_rolling_std_{window}'] = merged_df[main_corr_col].rolling(window=window).std()
                    except:
                        print(f"Warning: Could not calculate rolling stats for correlation index with window {window}")
            
            # Lagged variables with error handling
            for lag in [1, 3, 5, 10]:
                # Add lagged variables for ambiguity if column exists
                if ambiguity_cols:
                    main_ambiguity_col = ambiguity_cols[0]
                    try:
                        merged_df[f'ambiguity_lag_{lag}'] = merged_df[main_ambiguity_col].shift(lag)
                    except:
                        print(f"Warning: Could not calculate lag {lag} for ambiguity")
                
                # Add lagged variables for correlation index if column exists
                if correlation_cols:
                    main_corr_col = correlation_cols[0]
                    try:
                        merged_df[f'correlation_index_lag_{lag}'] = merged_df[main_corr_col].shift(lag)
                    except:
                        print(f"Warning: Could not calculate lag {lag} for correlation index")
        
        # Store the combined data
        self.combined_data = merged_df.copy() if merged_df is not None else pd.DataFrame()
        
        # Print final summary statistics
        if not self.combined_data.empty:
            print(f"Final combined data shape: {self.combined_data.shape}")
            print(f"Date range: {self.combined_data['date'].min()} to {self.combined_data['date'].max()}")
            print(f"Available columns: {self.combined_data.columns.tolist()}")
        else:
            print("Warning: Combined data is empty after processing")
            # Don't call dropna() here to preserve some data for analysis
    
    def save_combined_data(self):
        """Save the combined data with all variables"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Rename columns to English
        english_column_names = {
            'date': 'Date',
            'daily_return': 'Daily_Return',
            'SSE.000300.close': 'SSE300_Close',
            'ambiguity_metric': 'Ambiguity_Metric',
            'risk': 'Risk_Metric',
            'Climate_Risk_Component': 'Climate_Risk_Component',
            'GPR_Component': 'Global_GPR_Component',
            'Correlation_Adjusted_Index': 'Correlation_Adjusted_Index',
            'Equal_Weighted_Index': 'Equal_Weighted_Index',
            'Climate_Risk_Index': 'Climate_Risk_Index',
            'GPR_China': 'GPR_China',
            'GPR_Japan': 'GPR_Japan',
            'GPR_US': 'GPR_US'
        }
        
        # Rename columns
        export_data = self.combined_data.rename(columns=english_column_names)
        
        # Select columns with data
        export_columns = ['Date', 'Daily_Return', 'SSE300_Close', 'Ambiguity_Metric', 'Risk_Metric']
        
        # Add other columns if they exist
        for col in ['Climate_Risk_Component', 'Global_GPR_Component', 'Correlation_Adjusted_Index', 
                   'Equal_Weighted_Index', 'Climate_Risk_Index', 'GPR_China', 'GPR_Japan', 'GPR_US']:
            if col in export_data.columns:
                export_columns.append(col)
        
        export_data[export_columns].to_csv(self.output_csv_path, index=False)
        print(f"Combined data saved to: {self.output_csv_path}")
        print(f"Saved columns: {export_columns}")
    
    def calculate_correlations(self):
        """Calculate enhanced correlations between variables"""
        print("\nCalculating enhanced correlations...")
        
        # Create a copy of the data for analysis
        analysis_df = self.combined_data.copy()
        
        # Print all available columns for debugging
        print(f"All available columns in combined_data: {analysis_df.columns.tolist()}")
        
        # Define column mappings (capitalized -> lowercase or keep as is)
        column_mappings = {
            'Daily_Return': 'daily_return',
            'Ambiguity_Metric': 'ambiguity_metric',
            'Risk_Metric': 'risk',
            'Correlation_Adjusted_Index': 'Correlation_Adjusted_Index',
            'Equal_Weighted_Index': 'Equal_Weighted_Index',
            'GPR_China': 'GPR_China',
            'GPR_Japan': 'GPR_Japan',
            'GPR_US': 'GPR_US',
            'Climate_Risk_Index': 'Climate_Risk_Index'
        }
        
        # Create a renamed dataframe with consistent column names
        renamed_df = pd.DataFrame()
        
        # First check for capitalized versions (as saved in the CSV)
        for source_col, target_col in column_mappings.items():
            if source_col in analysis_df.columns and not analysis_df[source_col].isna().all():
                renamed_df[target_col] = analysis_df[source_col]
                print(f"Added column: {source_col} -> {target_col}")
        
        # Then check for lowercase versions if needed
        lowercase_alternatives = {
            'daily_return': 'daily_return',
            'ambiguity_metric': 'ambiguity_metric',
            'risk': 'risk'
        }
        
        for source_col, target_col in lowercase_alternatives.items():
            if source_col in analysis_df.columns and target_col not in renamed_df.columns and not analysis_df[source_col].isna().all():
                renamed_df[target_col] = analysis_df[source_col]
                print(f"Added lowercase column: {source_col} -> {target_col}")
        
        # Build base columns list with error handling
        base_cols = []
        
        # Core variables with their expected names in renamed_df
        core_vars = ['daily_return', 'ambiguity_metric', 'risk', 
                    'Correlation_Adjusted_Index', 'Equal_Weighted_Index']
        for var in core_vars:
            if var in renamed_df.columns and not renamed_df[var].isna().all():
                base_cols.append(var)
        
        # GPR variables
        gpr_vars = ['GPR_China', 'GPR_Japan', 'GPR_US']
        for var in gpr_vars:
            if var in renamed_df.columns and not renamed_df[var].isna().all():
                base_cols.append(var)
        
        # Climate risk variable
        if 'Climate_Risk_Index' in renamed_df.columns and not renamed_df['Climate_Risk_Index'].isna().all():
            base_cols.append('Climate_Risk_Index')
        
        print(f"Selected columns for correlation analysis: {base_cols}")
        
        # Check if we have any data to analyze
        if not base_cols:
            print("\nWarning: No valid data columns found for correlation analysis.")
            return pd.DataFrame()
        
        # Calculate correlation matrix only with valid columns
        corr_matrix = renamed_df[base_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # Focus on ambiguity correlations
        ambiguity_col = 'ambiguity_metric' if 'ambiguity_metric' in corr_matrix.columns else None
        if ambiguity_col:
            ambiguity_corr = corr_matrix[ambiguity_col].sort_values(ascending=False)
            print("\nCorrelations with Ambiguity Metric:")
            print(ambiguity_corr.round(4))
        else:
            print("\nWarning: Ambiguity metric not found in correlation matrix.")
            return corr_matrix
        
        # Plot enhanced correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   fmt='.3f', mask=mask, square=True, linewidths=.5, cbar_kws={'label': 'Correlation'})
        plt.title('Enhanced Correlation Matrix of Risk and Uncertainty Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300)
        plt.close()
        
        # Create a focused plot on ambiguity correlations
        plt.figure(figsize=(10, 6))
        # Only attempt to plot if we have the ambiguity column and other correlations
        if len(ambiguity_corr.dropna()) > 1:
            ambiguity_corr.drop(ambiguity_col).plot(kind='bar', color='skyblue')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Correlations with Ambiguity Metric', fontsize=14)
            plt.ylabel('Correlation Coefficient', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/ambiguity_correlations.png', dpi=300)
            plt.close()
        else:
            print("Warning: Not enough correlation data to create ambiguity correlations plot.")
        
        return corr_matrix
    
    def check_stationarity(self, series):
        """Check stationarity of time series"""
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def granger_causality_test(self, max_lag=15):
        """Enhanced Granger causality testing"""
        print("\nPerforming enhanced Granger causality tests...")
        
        # Check stationarity of key series that have data
        key_series = []
        # Start with essential variables
        essential_vars = ['daily_return', 'ambiguity_metric', 'Correlation_Adjusted_Index']
        for var in essential_vars:
            if var in self.combined_data.columns and not self.combined_data[var].isna().all():
                key_series.append(var)
        
        # Check GPR variables if they exist and have data
        gpr_vars = ['GPR_China', 'GPR_Japan', 'GPR_US']
        for var in gpr_vars:
            if var in self.combined_data.columns and not self.combined_data[var].isna().all():
                key_series.append(var)
        
        print("\nStationarity tests:")
        for col in key_series:
            try:
                stationarity = self.check_stationarity(self.combined_data[col])
                print(f"{col}:")
                print(f"  ADF Statistic: {stationarity['adf_statistic']:.4f}")
                print(f"  p-value: {stationarity['p_value']:.4f}")
                print(f"  Stationary: {stationarity['is_stationary']}")
            except Exception as e:
                print(f"Error checking stationarity for {col}: {e}")
        
        # Test causality from uncertainty variables to ambiguity
        cause_vars = []
        # Check if ambiguity_metric has data first
        if 'ambiguity_metric' not in self.combined_data.columns or self.combined_data['ambiguity_metric'].isna().all():
            print("\nWarning: ambiguity_metric has no valid data for causality testing.")
            return {}
        
        # Add potential cause variables that have data
        potential_causes = ['Correlation_Adjusted_Index', 'GPR_China', 'GPR_Japan', 'GPR_US']
        for var in potential_causes:
            if var in self.combined_data.columns and not self.combined_data[var].isna().all():
                cause_vars.append(var)
        
        print(f"\nSelected cause variables for Granger causality testing: {cause_vars}")
        
        results = {}
        for cause in cause_vars:
            print(f"\nGranger causality test: {cause} -> ambiguity_metric")
            try:
                # Create a dataframe with only non-null pairs
                test_data = self.combined_data[[cause, 'ambiguity_metric']].dropna()
                
                if len(test_data) > max_lag + 1:  # Need enough data for the test
                    test_result = grangercausalitytests(
                        test_data, 
                        maxlag=max_lag, verbose=False
                    )
                    # Store p-values
                    p_values = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]
                    results[cause] = p_values
                    
                    # Find best lag
                    min_p = min(p_values)
                    best_lag = p_values.index(min_p) + 1
                    print(f"  Best lag: {best_lag}, p-value: {min_p:.4f}")
                    print(f"  Significant: {min_p < 0.05}")
                else:
                    print(f"  Not enough valid data for test (only {len(test_data)} observations)")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Visualize results
        plt.figure(figsize=(14, 8))
        lags = range(1, max_lag + 1)
        
        for cause, p_values in results.items():
            plt.plot(lags, p_values, 'o-', label=cause)
        
        plt.axhline(y=0.05, color='r', linestyle='--', label='Significance (0.05)')
        plt.axhline(y=0.10, color='orange', linestyle='--', label='Significance (0.10)')
        plt.xlabel('Lag', fontsize=12)
        plt.ylabel('p-value', fontsize=12)
        plt.title('Enhanced Granger Causality Tests: Uncertainty -> Ambiguity', fontsize=16)
        plt.legend(fontsize=10)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/granger_causality.png', dpi=300)
        plt.close()
        
        # Create heatmap of best p-values
        best_p_values = {cause: min(p_values) for cause, p_values in results.items()}
        best_lags = {cause: p_values.index(min(p_values)) + 1 for cause, p_values in results.items()}
        
        # Create DataFrame for heatmap
        heatmap_data = pd.DataFrame({
            'p_value': [best_p_values[cause] for cause in cause_vars],
            'best_lag': [best_lags[cause] for cause in cause_vars]
        }, index=cause_vars)
        
        plt.figure(figsize=(10, 6))
        mask = heatmap_data['p_value'] >= 0.05
        sns.heatmap(heatmap_data[['p_value']], annot=True, fmt='.4f', 
                   cmap='RdYlGn_r', vmin=0, vmax=0.1, mask=mask)
        plt.title('Best Granger Causality Results (p-values < 0.05 significant)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/granger_causality_heatmap.png', dpi=300)
        plt.close()
        
        return results
    
    def run_regression_analysis(self):
        """Enhanced regression analysis"""
        print("\nRunning enhanced regression analysis...")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Check if essential variables exist
        if 'daily_return' not in self.combined_data.columns or \
           'ambiguity_metric' not in self.combined_data.columns or \
           'risk' not in self.combined_data.columns:
            print("Error: Missing essential variables for regression analysis.")
            return {}
        
        # Model 1: Impact on returns (original)
        X1 = self.combined_data[['ambiguity_metric', 'risk']].dropna()
        y1 = self.combined_data.loc[X1.index, 'daily_return']
        
        # Model 2: Impact on returns with uncertainty indices
        X2_cols = ['ambiguity_metric', 'risk']
        # Add correlation index if available
        if 'Correlation_Adjusted_Index' in self.combined_data.columns and not self.combined_data['Correlation_Adjusted_Index'].isna().all():
            X2_cols.append('Correlation_Adjusted_Index')
        # Add GPR variables if available
        gpr_vars = ['GPR_China', 'GPR_Japan', 'GPR_US']
        for var in gpr_vars:
            if var in self.combined_data.columns and not self.combined_data[var].isna().all():
                X2_cols.append(var)
        
        X2 = self.combined_data[X2_cols].dropna()
        y2 = self.combined_data.loc[X2.index, 'daily_return']
        
        # Model 3: Predicting ambiguity
        X3_cols = []
        if 'Correlation_Adjusted_Index' in self.combined_data.columns and not self.combined_data['Correlation_Adjusted_Index'].isna().all():
            X3_cols.append('Correlation_Adjusted_Index')
        for var in gpr_vars:
            if var in self.combined_data.columns and not self.combined_data[var].isna().all():
                X3_cols.append(var)
        
        X3 = self.combined_data[X3_cols].dropna() if X3_cols else pd.DataFrame()
        y3 = self.combined_data.loc[X3.index, 'ambiguity_metric'] if X3_cols else pd.Series()
        
        print(f"Selected features for Model 2: {X2_cols}")
        print(f"Selected features for Model 3: {X3_cols}")
        
        # Initialize dictionaries to store results
        regression_results = {}
        
        # Model 1: Impact on returns (original)
        if len(X1) > 1:  # Need at least 2 data points for regression
            try:
                scaler = StandardScaler()
                X1_scaled = scaler.fit_transform(X1)
                
                model1 = LinearRegression()
                model1.fit(X1_scaled, y1)
                
                y1_pred = model1.predict(X1_scaled)
                r2_1 = r2_score(y1, y1_pred)
                
                regression_results['model1'] = {
                    'r2': r2_1,
                    'coef': model1.coef_,
                    'intercept': model1.intercept_
                }
                
                print("\nModel 1 (Ambiguity + Risk):")
                print(f"R² score: {r2_1:.4f}")
                print(f"Ambiguity coefficient: {model1.coef_[0]:.4f}")
                print(f"Risk coefficient: {model1.coef_[1]:.4f}")
            except Exception as e:
                print(f"Error in Model 1: {e}")
                regression_results['model1'] = {'error': str(e)}
        else:
            print("\nModel 1: Not enough data for regression")
            regression_results['model1'] = {'error': 'Not enough data'}
        
        # Model 2: Impact on returns with uncertainty indices
        if len(X2) > len(X2_cols):  # Need more data points than features
            try:
                scaler = StandardScaler()
                X2_scaled = scaler.fit_transform(X2)
                
                model2 = LinearRegression()
                model2.fit(X2_scaled, y2)
                
                y2_pred = model2.predict(X2_scaled)
                r2_2 = r2_score(y2, y2_pred)
                
                regression_results['model2'] = {
                    'r2': r2_2,
                    'coef': model2.coef_,
                    'features': X2_cols,
                    'intercept': model2.intercept_
                }
                
                print(f"\nModel 2 (With Uncertainty Indices):")
                print(f"R² score: {r2_2:.4f}")
                for i, col in enumerate(X2_cols):
                    print(f"{col} coefficient: {model2.coef_[i]:.4f}")
            except Exception as e:
                print(f"Error in Model 2: {e}")
                regression_results['model2'] = {'error': str(e)}
        else:
            print("\nModel 2: Not enough data for regression")
            regression_results['model2'] = {'error': 'Not enough data'}
        
        # Model 3: Predicting ambiguity
        if X3_cols and len(X3) > len(X3_cols):
            try:
                scaler = StandardScaler()
                X3_scaled = scaler.fit_transform(X3)
                
                model3 = LinearRegression()
                model3.fit(X3_scaled, y3)
                
                y3_pred = model3.predict(X3_scaled)
                r2_3 = r2_score(y3, y3_pred)
                
                regression_results['model3'] = {
                    'r2': r2_3,
                    'coef': model3.coef_,
                    'features': X3_cols,
                    'intercept': model3.intercept_
                }
                
                print(f"\nModel 3 (Predicting Ambiguity):")
                print(f"R² score: {r2_3:.4f}")
                for i, col in enumerate(X3_cols):
                    print(f"{col} coefficient: {model3.coef_[i]:.4f}")
            except Exception as e:
                print(f"Error in Model 3: {e}")
                regression_results['model3'] = {'error': str(e)}
        else:
            print("\nModel 3: Not enough data or features for regression")
            regression_results['model3'] = {'error': 'Not enough data or features'}
            
        # Return regression_results instead of building a new dictionary
        return regression_results
        
        # Plot coefficients
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Model 2 coefficients
        ax1.bar(range(len(model2.coef_)), model2.coef_)
        ax1.set_xticks(range(len(model2.coef_)))
        ax1.set_xticklabels([col.replace('_', ' ').title() for col in X2_cols], rotation=45, ha='right')
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax1.set_title('Model 2: Impact on Returns', fontsize=14)
        ax1.set_ylabel('Coefficient Value', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Model 3 coefficients
        ax2.bar(range(len(model3.coef_)), model3.coef_, color='orange')
        ax2.set_xticks(range(len(model3.coef_)))
        ax2.set_xticklabels([col.replace('_', ' ').title() for col in X3_cols], rotation=45, ha='right')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax2.set_title('Model 3: Predictors of Ambiguity', fontsize=14)
        ax2.set_ylabel('Coefficient Value', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Enhanced Regression Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f'{self.output_dir}/regression_analysis.png', dpi=300)
        plt.close()
        
        return {
            'model1': {'r2': r2_1, 'coef': model1.coef_},
            'model2': {'r2': r2_2, 'coef': model2.coef_, 'features': X2_cols},
            'model3': {'r2': r2_3, 'coef': model3.coef_, 'features': X3_cols}
        }
    
    def plot_time_series(self):
        """Enhanced time series plotting"""
        print("\nPlotting enhanced time series...")
        
        # Check if combined_data exists and has data
        if self.combined_data is None or self.combined_data.empty:
            print("Warning: No data available for time series plotting.")
            # Create a placeholder plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No valid data available for time series plotting\nPlease check data loading and preprocessing.', 
                     ha='center', va='center', fontsize=14)
            plt.title('Time Series Analysis', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/time_series_plots.png', dpi=300)
            plt.close()
            return
        
        # Convert date to datetime for plotting
        df_plot = self.combined_data.copy()
        df_plot['date'] = pd.to_datetime(df_plot['date'])
        
        print(f"Available columns in combined_data: {df_plot.columns.tolist()}")
        
        # Define key variables for plotting with their display names
        key_variables = {
            # Check both capitalized and lowercase versions
            'Daily_Return': 'Daily Return',
            'daily_return': 'Daily Return',
            'Ambiguity_Metric': 'Ambiguity Metric',
            'ambiguity_metric': 'Ambiguity Metric',
            'Risk_Metric': 'Risk Metric',
            'risk': 'Risk Metric',
            'Correlation_Adjusted_Index': 'Correlation Adjusted Index',
            'GPR_China': 'GPR China',
            'GPR_Japan': 'GPR Japan',
            'GPR_US': 'GPR US',
            'Climate_Risk_Component': 'Climate Risk Component',
            'Climate_Risk_Index': 'Climate Risk Index'
        }
        
        # Check which variables are available in the data
        available_vars = []
        added_vars = set()  # To avoid adding both capitalized and lowercase versions
        
        for var, display_name in key_variables.items():
            if var in df_plot.columns and not df_plot[var].isna().all():
                # Get the base name (remove potential case variations)
                base_name = display_name.lower()
                if base_name not in added_vars:
                    available_vars.append((var, display_name))
                    added_vars.add(base_name)
        
        key_vars = [var[0] for var in available_vars]  # Get variable names for processing
        display_names = {var[0]: var[1] for var in available_vars}  # Map variable to display name
        
        print(f"Selected key variables for time series plotting: {key_vars}")
        
        # Check if we have any variables to plot
        if not key_vars:
            print("Warning: No valid variables found for time series plotting.")
            # Create a placeholder plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No valid data available for time series plotting\nPlease check data loading and preprocessing.', 
                     ha='center', va='center', fontsize=14)
            plt.title('Time Series Analysis', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/time_series_plots.png', dpi=300)
            plt.close()
            return
        
        # Normalize variables with error handling
        scaler = StandardScaler()
        norm_vars = []  # Track which variables were successfully normalized
        
        for var in key_vars:
            try:
                # Only normalize if there are non-NaN values
                non_na_data = df_plot[var].dropna()
                if len(non_na_data) > 0:
                    # Create a normalized version
                    df_plot[f'norm_{var}'] = df_plot[var].copy()
                    # Scale the non-NaN values
                    mask = df_plot[var].notna()
                    df_plot.loc[mask, f'norm_{var}'] = scaler.fit_transform(df_plot.loc[mask, [var]])
                    norm_vars.append(var)
            except Exception as e:
                print(f"Error normalizing {var}: {e}")
                # Skip this variable if normalization fails
        
        # Create a grid based on available normalized data
        n_vars = len(norm_vars)
        if n_vars == 0:
            print("Warning: No variables could be normalized for plotting.")
            # Create a placeholder plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Error normalizing variables for plotting\nPlease check data quality.', 
                     ha='center', va='center', fontsize=14)
            plt.title('Time Series Analysis', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/time_series_plots.png', dpi=300)
            plt.close()
            return
        
        # Ensure at least 1 row for subplots
        n_rows = max(1, (n_vars + 1) // 2)
        
        # Create the subplots
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
        
        # Handle the case where we might have a single axis
        if n_rows == 1 and n_vars == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        # Plot each variable
        for i, var in enumerate(norm_vars):
            if i < len(axes):
                ax = axes[i]
                ax.plot(df_plot['date'], df_plot[f'norm_{var}'], linewidth=1.5)
                
                # Use display name if available, otherwise use formatted variable name
                title = display_names.get(var, var.replace("_", " ").title())
                ax.set_title(f'Time Series of {title}', fontsize=12)
                ax.set_ylabel('Normalized Value', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Calculate and annotate correlation with ambiguity if available
                if 'ambiguity_metric' in df_plot.columns and not df_plot['ambiguity_metric'].isna().all():
                    try:
                        corr = df_plot['ambiguity_metric'].corr(df_plot[var])
                        ax.annotate(f'Corr with ambiguity: {corr:.3f}', 
                                   xy=(0.05, 0.95), xycoords='axes fraction',
                                   fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
                    except:
                        pass
        
        # Remove empty subplots
        for i in range(n_vars, len(axes)):
            fig.delaxes(axes[i])
        
        plt.xlabel('Date', fontsize=12)
        plt.suptitle('Enhanced Time Series Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'{self.output_dir}/time_series_plots.png', dpi=300)
        plt.close()
        
        # Create a combined plot showing multiple series only if we have the essential variables
        if 'norm_ambiguity_metric' in df_plot.columns and 'norm_Correlation_Adjusted_Index' in df_plot.columns:
            plt.figure(figsize=(14, 7))
            plt.plot(df_plot['date'], df_plot['norm_ambiguity_metric'], 'b-', linewidth=2, label='Ambiguity (Normalized)')
            plt.plot(df_plot['date'], df_plot['norm_Correlation_Adjusted_Index'], 'r-', linewidth=2, label='Uncertainty Index (Normalized)')
            
            # Add country GPR if available
            if 'norm_GPR_China' in df_plot.columns:
                plt.plot(df_plot['date'], df_plot['norm_GPR_China'], 'g--', linewidth=1.5, label='GPR China (Normalized)')
            if 'norm_GPR_US' in df_plot.columns:
                plt.plot(df_plot['date'], df_plot['norm_GPR_US'], 'm--', linewidth=1.5, label='GPR US (Normalized)')
            
            plt.title('Combined Time Series of Key Variables', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Normalized Value', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/combined_time_series.png', dpi=300)
            plt.close()
    
    def enhance_causality_relationships(self):
        """Suggest methods to enhance weak causal relationships"""
        print("\nRecommendations to enhance causal relationships:")
        print("1. Use higher-frequency data if available (e.g., intraday instead of daily)")
        print("2. Apply time series transformations:")
        print("   - First differences to ensure stationarity")
        print("   - Log transformations for volatility stabilization")
        print("3. Consider non-linear models:")
        print("   - GARCH models for volatility relationships")
        print("   - Non-linear Granger causality tests")
        print("4. Add control variables:")
        print("   - Macroeconomic indicators")
        print("   - Market liquidity measures")
        print("   - Seasonal dummy variables")
        print("5. Apply machine learning approaches:")
        print("   - Random Forest feature importance")
        print("   - Gradient Boosting models")
        print("   - Neural networks for complex relationships")
        print("6. Use wavelet transforms to analyze relationships at different time scales")
        print("7. Consider vector autoregression (VAR) models to capture system dynamics")
    
    def run_all_analysis(self):
        """Run all analysis and save results with robust error handling"""
        print("\n=== RUNNING COMPREHENSIVE DATA ANALYSIS ===\n")
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results dictionary
        results = {}
        
        try:
            # Load and prepare data
            print("1. Loading and preparing data...")
            try:
                self.load_data()
                # Check if data was loaded successfully
                if self.combined_data is None or self.combined_data.empty:
                    print("Error: Combined data is empty after loading. Please check data sources.")
                    return results
            except Exception as e:
                print(f"Warning: Error loading data: {e}")
                return results
            
            # Save combined data
            print("\n2. Saving combined analysis data...")
            try:
                self.save_combined_data()
            except Exception as e:
                print(f"Warning: Error saving combined data: {e}")
            
            # Run correlation analysis with error handling
            print("\n3. Running correlation analysis...")
            try:
                corr_matrix = self.calculate_correlations()
                results['correlation'] = corr_matrix
            except Exception as e:
                print(f"Warning: Error in correlation analysis: {e}")
                results['correlation'] = None
            
            # Run stationarity tests with error handling
            print("\n4. Checking stationarity...")
            stationarity_results = {}
            potential_vars = ['daily_return', 'ambiguity_metric', 'Correlation_Adjusted_Index',
                             'Daily_Return', 'Ambiguity_Metric', 'GPR_China', 'GPR_Japan', 'GPR_US']
            
            for var in potential_vars:
                try:
                    if var in self.combined_data.columns and not self.combined_data[var].isna().all():
                        stationarity = self.check_stationarity(self.combined_data[var])
                        stationarity_results[var] = stationarity
                        print(f"{var}: ADF Stat={stationarity['adf_statistic']:.4f}, p-val={stationarity['p_value']:.4f}, Stationary={stationarity['is_stationary']}")
                except Exception as e:
                    print(f"Warning: Error checking stationarity for {var}: {e}")
            
            results['stationarity_results'] = stationarity_results
            
            # Run Granger causality tests with error handling
            print("\n5. Running Granger causality tests...")
            try:
                granger_results = self.granger_causality_test()
                results['granger'] = granger_results
            except Exception as e:
                print(f"Warning: Error in Granger causality tests: {e}")
                results['granger'] = None
            
            # Run regression analysis with error handling
            print("\n6. Running regression analysis...")
            try:
                regression_results = self.run_regression_analysis()
                results['regression'] = regression_results
            except Exception as e:
                print(f"Warning: Error in regression analysis: {e}")
                results['regression'] = None
            
            # Plot time series with error handling
            print("\n7. Creating time series plots...")
            try:
                self.plot_time_series()
            except Exception as e:
                print(f"Warning: Error creating time series plots: {e}")
            
            # Enhance causality relationships
            print("\n8. Suggesting strategies to enhance causality relationships...")
            try:
                self.enhance_causality_relationships()
            except Exception as e:
                print(f"Warning: Error generating enhancement suggestions: {e}")
            
            print("\n=== ANALYSIS COMPLETE ===")
            print(f"All outputs saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"\nError in overall analysis: {e}")
            import traceback
            traceback.print_exc()
        
        return results

if __name__ == "__main__":
    analysis = DataAnalysisAndCausality()
    results = analysis.run_all_analysis()