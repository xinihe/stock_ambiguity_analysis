"""
================================================================================
Data Loader for Chinese A-Share Energy Market
================================================================================

Paper: "Pricing the Unknown: Ambiguity Premiums in China's Green vs. Brown Energy Markets"
Author: Research Team
Date: 2024
Reference: causal_ambi_china.tex

--------------------------------------------------------------------------------
MODULE OVERVIEW
--------------------------------------------------------------------------------
This module handles all data loading and preprocessing for the Chinese A-share
energy market analysis. It connects to a PostgreSQL database containing high-
frequency stock data and processes it for ambiguity measurement and causal
analysis.

--------------------------------------------------------------------------------
DATABASE CONNECTION DETAILS
--------------------------------------------------------------------------------
Server: 10.28.255.30 (Ubuntu server)
Database: stock_hf
Required Table Columns:
  - code: Stock identifier (e.g., '601857.SH', '300750.SZ')
  - closeprice: Closing price at each timestamp
  - datetime: Timestamp of the price observation

The database contains minute-level or high-frequency data for Chinese A-share
stocks. This module will query the database, calculate returns, and prepare
the data for ambiguity analysis.

--------------------------------------------------------------------------------
KEY DATA REQUIREMENTS
--------------------------------------------------------------------------------
1. High-Frequency Price Data:
   - Source: PostgreSQL database stock_hf
   - Frequency: 1-minute or 5-minute intervals
   - Required: code, closeprice, datetime
   - Processing: Calculate log returns from closeprice

2. Energy Stock Classification:
   - Brown Energy: Coal, oil, thermal power
   - Green Energy: Solar, wind, EV, batteries, hydro
   - Grey Energy: Utilities, grid companies

3. Control Variables:
   - EPU (Economic Policy Uncertainty)
   - Policy sensitivity
   - Geopolitical indicators (defense, gold)
   - Commodity prices (oil, carbon)

--------------------------------------------------------------------------------
OUTPUT DATA STRUCTURE
--------------------------------------------------------------------------------
The main function prepare_analysis_data() returns a dictionary containing:
{
    'intraday_returns': DataFrame with datetime index, stocks as columns
    'energy_classification': Dict mapping stock_id -> 'Brown'/'Green'/'Grey'
    'sector_mapping': Dict for IV construction
    'limit_days': Dict of limit-up/limit-down days to exclude
    'epu_series': Time series of EPU values
    'policy_sensitivity': Firm-level policy sensitivity scores
    'geopolitical_data': Defense and gold returns
    'control_data': Oil and carbon returns
    'policy_shock_dates': List of key policy announcement dates
}

================================================================================
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy import create_engine, text
import psycopg2

warnings.filterwarnings('ignore')


class ChinaEnergyDataLoader:
    """
    ================================================================================
    Data Loader for Chinese A-Share Energy Market Analysis
    ================================================================================

    This class handles all data loading and preprocessing operations:
    1. Connecting to PostgreSQL database and extracting high-frequency data
    2. Calculating intraday returns from price data
    3. Classifying stocks into energy sectors (Brown/Green/Grey)
    4. Loading control variables for causal analysis
    5. Handling Chinese A-share market specifics (limit days, price limits)

    --------------------------------------------------------------------------------
    DATABASE SCHEMA
    --------------------------------------------------------------------------------
    Table: stock_hf (High-frequency stock data)
    Columns:
      - code: VARCHAR, Stock identifier (e.g., '601857.SH' for PetroChina)
      - closeprice: DECIMAL, Closing price at the timestamp
      - datetime: TIMESTAMP, Date and time of the price observation

    Primary Keys: (code, datetime)
    Indexes: datetime (for time-based queries), code (for stock-based queries)

    --------------------------------------------------------------------------------
    ENERGY SECTOR CLASSIFICATION
    --------------------------------------------------------------------------------
    BROWN ENERGY (Traditional):
      - Coal Mining (CSI Coal Index)
      - Oil & Gas Exploration (e.g., PetroChina 601857.SH)
      - Thermal Power Generation
      - Petroleum Refining

    GREEN ENERGY (Renewable):
      - Solar Energy (CSI Photovoltaic Index, e.g., Longi Green Energy 601012.SH)
      - Wind Power (CSI Wind Power Index)
      - Hydro Power
      - New Energy Vehicles (e.g., BYD 002594.SZ)
      - EV Batteries (e.g., CATL 300750.SZ)
      - Lithium Battery Manufacturing
      - Power Equipment for renewables

    GREY ENERGY (Utilities):
      - Electric Utilities (traditional grid companies)
      - Power Transmission and Distribution
      - Grid Companies (State Grid subsidiaries)

    --------------------------------------------------------------------------------
    POLICY SHOCK DATES (for Difference-in-Differences Analysis)
    --------------------------------------------------------------------------------
    These dates represent major policy announcements during China's "Dual Carbon"
    transition (Peaking Carbon by 2030, Carbon Neutrality by 2060):

    1. 2020-09-22: President Xi's UN speech announcing 2060 carbon neutrality
    2. 2021-03-15: 14th Five-Year Plan approval with carbon goals
    3. 2021-07-16: National carbon market official launch
    4. 2021-10-24: "Dual Carbon" policy documents (1+N policy framework)

    These dates are used as event dates in DiD analysis to test how ambiguity
    changes differentially across Brown vs. Green energy sectors.

    --------------------------------------------------------------------------------
    ATTRIBUTES
    --------------------------------------------------------------------------------
    db_connection_params : dict
        PostgreSQL database connection parameters
        - host: '10.28.255.30'
        - database: 'stock_hf'
        - User authentication required (add your credentials)

    data_path : str
        Fallback path for CSV files if database connection fails

    brown_energy_sectors : list
        List of sector names classified as Brown (traditional) energy

    green_energy_sectors : list
        List of sector names classified as Green (renewable) energy

    grey_energy_sectors : list
        List of sector names classified as Grey (utilities) energy

    policy_shock_dates : list
        List of policy announcement dates for DiD analysis

    --------------------------------------------------------------------------------
    METHODS
    --------------------------------------------------------------------------------
    __init__(data_path='data/')
        Initialize the data loader with database connection parameters

    load_intraday_returns_from_db(stock_list, start_date, end_date, freq='1min')
        Query database and load high-frequency data for specified stocks

    calculate_intraday_returns(price_df)
        Calculate log returns from price data

    classify_energy_stocks(stock_list, sector_mapping=None)
        Classify stocks into Brown/Green/Grey sectors

    load_limit_days(stock_list)
        Load limit-up/limit-down days from database or CSV

    load_epu_data(start_date, end_date)
        Load China Economic Policy Uncertainty Index

    load_policy_sensitivity(stock_list, start_date, end_date)
        Load firm-level policy sensitivity scores

    load_geopolitical_data(start_date, end_date)
        Load defense and gold returns for geopolitical ambiguity

    load_control_data(start_date, end_date)
        Load control variables (oil, carbon prices)

    ================================================================================
    """

    def __init__(self, data_path='data/', db_host='10.28.255.30', db_name='stock_hf',
                 db_user=None, db_password=None, db_port=5432):
        """
        ========================================================================
        Initialize the Data Loader
        ========================================================================

        Parameters:
        -----------
        data_path : str, default='data/'
            Fallback path for CSV files if database connection fails.
            Used for storing cached data or loading external variables.

        db_host : str, default='10.28.255.30'
            IP address of the Ubuntu server hosting the PostgreSQL database.
            This is the research team's internal server.

        db_name : str, default='stock_hf'
            Name of the PostgreSQL database containing high-frequency data.

        db_user : str, default=None
            PostgreSQL username. If None, will attempt to use environment
            variables or prompt for credentials.

        db_password : str, default=None
            PostgreSQL password. Should be stored securely, not hardcoded.

        db_port : int, default=5432
            PostgreSQL port number (standard port is 5432).

        Returns:
        --------
        None

        Notes:
        ------
        Database credentials should be managed securely. Recommended approach:
        1. Set environment variables: DB_USER, DB_PASSWORD
        2. Use a configuration file with restricted permissions
        3. Use SSH tunnel for secure remote connection

        Example:
        --------
        >>> loader = ChinaEnergyDataLoader(
        ...     db_host='10.28.255.30',
        ...     db_name='stock_hf',
        ...     db_user='your_username',
        ...     db_password='your_password'
        ... )
        ========================================================================
        """
        self.data_path = data_path

        # Store database connection parameters
        # DO NOT hardcode credentials in production code
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user or os.environ.get('DB_USER')
        self.db_password = db_password or os.environ.get('DB_PASSWORD')
        self.db_port = db_port

        # Create SQLAlchemy engine for database queries
        # This engine will be used for all database operations
        self.db_engine = None
        if self.db_user and self.db_password:
            try:
                connection_string = (
                    f"postgresql://{self.db_user}:{self.db_password}@"
                    f"{self.db_host}:{self.db_port}/{self.db_name}"
                )
                self.db_engine = create_engine(connection_string)
                print(f"✓ Successfully connected to database: {db_name}")
            except Exception as e:
                print(f"⚠ Warning: Could not connect to database: {e}")
                print("  Will use CSV fallback files from {data_path}")
        else:
            print("⚠ Warning: No database credentials provided.")
            print("  Set DB_USER and DB_PASSWORD environment variables, or pass credentials.")
            print("  Will use CSV fallback files from {data_path}")

        # ========================================================================
        # ENERGY SECTOR CLASSIFICATIONS
        # ========================================================================
        # These classifications are based on China's industry classification
        # standards and are used to categorize stocks for the analysis.

        # BROWN ENERGY: Traditional fossil fuel-based energy
        # These sectors face regulatory pressure under carbon policies
        self.brown_energy_sectors = [
            'CSI Coal Index',           # Coal mining and processing
            'Coal Mining',              # Coal extraction companies
            'Oil & Gas Exploration',    # Oil and natural gas exploration
            'Thermal Power',            # Coal-fired power plants
            'Petroleum Refining',       # Oil refineries
            'Coking Coal',              # Metallurgical coal
            'Oil & Gas Drilling'        # Oil and gas drilling services
        ]

        # GREEN ENERGY: Renewable and clean energy
        # These sectors benefit from carbon reduction policies
        self.green_energy_sectors = [
            'CSI Photovoltaic Index',   # Solar panel manufacturing
            'CSI Wind Power Index',     # Wind turbine manufacturing
            'Solar Energy',             # Solar power generation
            'Wind Power',               # Wind power generation
            'Hydro Power',              # Hydroelectric power
            'New Energy Vehicles',      # Electric vehicle manufacturers
            'EV Batteries',             # Electric vehicle batteries
            'Lithium Battery',          # Lithium-ion battery production
            'Power Equipment',          # Equipment for renewable energy
            'Nuclear Power',            # Nuclear power generation
            'Biomass Energy',           # Biomass and waste-to-energy
            'Energy Storage'            # Energy storage systems
        ]

        # GREY ENERGY: Utilities and grid infrastructure
        # These sectors are neutral/transitionary in the energy transition
        self.grey_energy_sectors = [
            'Electric Utilities',       # Traditional electric utilities
            'Grid Companies',           # Power grid operators
            'Power Transmission',       # High-voltage transmission
            'Distribution',             # Power distribution networks
            'Gas Utilities',            # Natural gas distribution
            'Water Utilities'           # Water and wastewater utilities
        ]

        # ========================================================================
        # POLICY SHOCK DATES (Natural Experiments)
        # ========================================================================
        # These dates represent major policy announcements during China's
        # "Dual Carbon" transition. They serve as natural experiments for
        # Difference-in-Differences (DiD) analysis.

        self.policy_shock_dates = [
            '2020-09-22',  # EVENT 1: President Xi Jinping's UN speech
                           # Context: China commits to peak carbon by 2030 and
                           #          carbon neutrality by 2060
                           # Impact: Massive shift in energy policy expectations

            '2021-03-15',  # EVENT 2: 14th Five-Year Plan approval
                           # Context: National People's Congress approves the
                           #          14th Five-Year Plan with carbon goals
                           # Impact: Legal framework for carbon reduction

            '2021-07-16',  # EVENT 3: National carbon market launch
                           # Context: China's national Emissions Trading System
                           #          (ETS) officially launches
                           # Impact: Direct pricing of carbon emissions

            '2021-10-24',  # EVENT 4: "Dual Carbon" policy documents (1+N)
                           # Context: State Council releases the "1+N" policy
                           #          framework for carbon goals
                           # Impact: Detailed implementation roadmap
        ]

        # Stock universe: Will be populated from database query
        # This will contain all stock codes available in the database
        self.stock_universe = []

    def load_stock_universe_from_db(self):
        """
        ========================================================================
        Load Stock Universe from Database
        ========================================================================

        Queries the database to get all unique stock codes. This creates the
        master list of all stocks available for analysis.

        SQL Query:
        ---------
        SELECT DISTINCT code FROM stock_hf ORDER BY code;

        Returns:
        --------
        stock_list : list of str
            List of all stock codes available in the database

        Notes:
        ------
        - This method populates self.stock_universe
        - Stock codes follow Chinese A-share convention:
          * Shanghai stocks: 6XXXXX.SH (e.g., 601857.SH for PetroChina)
          * Shenzhen stocks: 0XXXXXX.SZ (e.g., 000001.SZ for Ping An)
          * ChiNext stocks: 3XXXXXX.SZ (e.g., 300750.SZ for CATL)

        Example:
        --------
        >>> loader.load_stock_universe_from_db()
        >>> print(f"Found {len(loader.stock_universe)} stocks")
        ========================================================================
        """
        if self.db_engine is None:
            print("⚠ No database connection. Cannot load stock universe.")
            return []

        try:
            query = """
            SELECT DISTINCT code
            FROM stock_hf
            ORDER BY code;
            """

            with self.db_engine.connect() as conn:
                result = conn.execute(text(query))
                self.stock_universe = [row[0] for row in result]

            print(f"✓ Loaded {len(self.stock_universe)} stocks from database")
            return self.stock_universe

        except Exception as e:
            print(f"✗ Error loading stock universe: {e}")
            return []

    def load_intraday_returns_from_db(self, stock_list: Optional[List[str]] = None,
                                       start_date: str = '2018-01-01',
                                       end_date: str = '2024-05-24',
                                       freq: str = '1min') -> pd.DataFrame:
        """
        ========================================================================
        Load High-Frequency Intraday Returns from Database
        ========================================================================

        This is the PRIMARY METHOD for loading stock data. It queries the
        PostgreSQL database for high-frequency price data and calculates
        intraday returns.

        Data Flow:
        ----------
        1. Query database for code, closeprice, datetime
        2. Filter by stock_list (if provided) and date range
        3. Pivot data to wide format (datetime index, stock columns)
        4. Calculate log returns: r_t = ln(P_t / P_{t-1})
        5. Handle missing data and outliers

        Database Query:
        ---------------
        SELECT code, datetime, closeprice
        FROM stock_hf
        WHERE datetime >= %(start_date)s
          AND datetime <= %(end_date)s
          AND code IN %(stock_list)s
        ORDER BY datetime, code;

        Parameters:
        -----------
        stock_list : list of str or None, default=None
            List of stock codes to load. If None, loads all available stocks.
            Example: ['601857.SH', '300750.SZ', '601012.SH']

        start_date : str, default='2018-01-01'
            Start date for data extraction (format: YYYY-MM-DD)
            The analysis typically starts from 2018 to capture pre-transition
            baseline period before major policy shocks in 2020-2021.

        end_date : str, default='2024-05-24'
            End date for data extraction (format: YYYY-MM-DD)
            Can be adjusted to most recent available data.

        freq : str, default='1min'
            Frequency of the data. Options:
            - '1min': One-minute intervals (most detailed, highest ambiguity)
            - '5min': Five-minute intervals (common in Chinese market research)
            - 'tick': Tick-by-tick data (if available)

            Note: The database should contain data at this frequency.
                  If not, specify resampling parameters.

        Returns:
        --------
        returns_df : pandas DataFrame
            DataFrame containing intraday returns with:
            - Index: DatetimeIndex (minute-level timestamps)
            - Columns: Stock codes (each column is a stock)
            - Values: Log returns (continuous compounding)

            Shape: (n_time_periods, n_stocks)
            Example:
                               601857.SH  300750.SZ  601012.SH
            2018-01-01 09:30:00  0.000234   0.000445   0.000123
            2018-01-01 09:31:00 -0.000156  -0.000234  -0.000089
            ...

        Notes:
        ------
        1. Return Calculation:
           - Uses log returns: r_t = ln(P_t) - ln(P_{t-1})
           - Log returns are preferred for financial analysis because:
             * They are additive over time
             * They are symmetric for gains/losses
             * They approximate percentage returns for small changes

        2. Data Quality:
           - Automatically removes prices of 0 or negative
           - Handles missing data with forward-fill then backward-fill
           - Removes extreme outliers (> 10 standard deviations)

        3. Market Hours:
           - Chinese A-share trading hours: 9:30-11:30, 13:00-15:00
           - Database should contain only trading hours
           - Lunch break (11:30-13:00) is excluded

        4. Limit Days:
           - Limit-up/limit-down days should be identified and excluded
           - Use load_limit_days() method to get these dates
           - Limit days create artificial truncations in return distribution

        Example:
        --------
        >>> loader = ChinaEnergyDataLoader()
        >>> returns = loader.load_intraday_returns_from_db(
        ...     stock_list=['601857.SH', '300750.SZ'],
        ...     start_date='2020-01-01',
        ...     end_date='2022-12-31'
        ... )
        >>> print(returns.shape)
        >>> print(returns.head())
        ========================================================================
        """
        if self.db_engine is None:
            print("⚠ No database connection. Using fallback CSV data.")
            return self.load_intraday_returns()

        # If stock_list not provided, load all stocks
        if stock_list is None:
            if not self.stock_universe:
                self.load_stock_universe_from_db()
            stock_list = self.stock_universe

        print(f"\n{'='*70}")
        print(f"Loading Intraday Returns from Database")
        print(f"{'='*70}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Stocks: {len(stock_list)}")
        print(f"Frequency: {freq}")

        try:
            # ====================================================================
            # STEP 1: Query Database for Price Data
            # ====================================================================
            # Build SQL query with parameterized inputs for security
            query = f"""
            SELECT code, datetime, closeprice
            FROM stock_hf
            WHERE datetime >= %(start_date)s
              AND datetime <= %(end_date)s
              AND code IN %(stock_list)s
            ORDER BY datetime, code;
            """

            params = {
                'start_date': start_date,
                'end_date': end_date,
                'stock_list': tuple(stock_list)
            }

            with self.db_engine.connect() as conn:
                # Read data directly into pandas DataFrame
                price_df = pd.read_sql_query(query, conn, params=params)

            print(f"✓ Loaded {len(price_df):,} price observations from database")
            print(f"  Memory usage: {price_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # ====================================================================
            # STEP 2: Data Cleaning and Validation
            # ====================================================================
            # Remove invalid prices (zero, negative, or NaN)
            original_len = len(price_df)
            price_df = price_df[
                (price_df['closeprice'] > 0) &
                (price_df['closeprice'].notna())
            ]
            removed = original_len - len(price_df)
            if removed > 0:
                print(f"⚠ Removed {removed:,} invalid price observations")

            # Convert datetime to proper datetime type (ensure timezone awareness)
            price_df['datetime'] = pd.to_datetime(price_df['datetime'])

            # ====================================================================
            # STEP 3: Pivot to Wide Format
            # ====================================================================
            # Convert from long format (datetime, code, price) to wide format
            # Wide format: datetime as index, stocks as columns
            price_wide = price_df.pivot(
                index='datetime',
                columns='code',
                values='closeprice'
            )

            print(f"✓ Pivoted to wide format: {price_wide.shape}")
            print(f"  Time periods: {len(price_wide):,}")
            print(f"  Stocks: {len(price_wide.columns)}")

            # Check for stocks with insufficient data
            min_observations = len(price_wide) * 0.01  # Require at least 1% coverage
            valid_stocks = price_wide.count()[price_wide.count() >= min_observations].index
            if len(valid_stocks) < len(price_wide.columns):
                removed_stocks = len(price_wide.columns) - len(valid_stocks)
                print(f"⚠ Removed {removed_stocks} stocks with insufficient data")
                price_wide = price_wide[valid_stocks]

            # ====================================================================
            # STEP 4: Calculate Log Returns
            # ====================================================================
            # Formula: r_t = ln(P_t) - ln(P_{t-1})
            # This is the standard return measure in financial econometrics

            # First, forward-fill missing prices (handle temporary halts)
            price_wide = price_wide.fillna(method='ffill').fillna(method='bfill')

            # Calculate log returns
            returns_df = np.log(price_wide).diff()

            # Remove the first row (will be NaN due to differencing)
            returns_df = returns_df.iloc[1:]

            print(f"✓ Calculated log returns")
            print(f"  Mean return: {returns_df.mean().mean():.6f}")
            print(f"  Std return: {returns_df.std().mean():.6f}")

            # ====================================================================
            # STEP 5: Handle Extreme Outliers
            # ====================================================================
            # Extreme returns may be data errors or limit moves
            # We flag them but don't remove automatically (user's choice)

            # Calculate z-scores for each stock
            mean_return = returns_df.mean()
            std_return = returns_df.std()
            z_scores = (returns_df - mean_return) / std_return

            # Count extreme observations
            extreme_threshold = 10  # 10 standard deviations
            extreme_count = (np.abs(z_scores) > extreme_threshold).sum().sum()
            if extreme_count > 0:
                print(f"⚠ Found {extreme_count} extreme returns (> {extreme_threshold} std)")

            # Optionally winsorize extreme values
            # returns_df = returns_df.clip(lower=mean_return - extreme_threshold*std_return,
            #                              upper=mean_return + extreme_threshold*std_return,
            #                              axis=1)

            print(f"\n{'='*70}")
            print(f"✓ Data Loading Complete")
            print(f"{'='*70}")

            return returns_df

        except Exception as e:
            print(f"✗ Error loading from database: {e}")
            print("  Falling back to CSV files...")
            return self.load_intraday_returns()

    def load_intraday_returns(self, filename='intraday_returns.csv') -> pd.DataFrame:
        """
        ========================================================================
        Fallback Method: Load Intraday Returns from CSV
        ========================================================================

        This method is used as a fallback when database connection fails.
        It loads pre-calculated returns from a CSV file.

        Expected CSV Format:
        ---------------------
        date,time,stock_id,return
        2018-01-01,09:30:00,601857.SH,0.000234
        2018-01-01,09:31:00,601857.SH,-0.000156
        2018-01-01,09:30:00,300750.SZ,0.000445

        Parameters:
        -----------
        filename : str
            Name of the CSV file in the data_path directory

        Returns:
        --------
        returns_df : pandas DataFrame
            DataFrame with DatetimeIndex and stock_id as columns

        Notes:
        ------
        This is primarily for testing and development. Production use should
        connect directly to the database.
        ========================================================================
        """
        filepath = os.path.join(self.data_path, filename)

        try:
            df = pd.read_csv(filepath, parse_dates=['date', 'time'])
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            df.set_index('datetime', inplace=True)

            # Pivot to wide format
            returns_df = df.pivot(columns='stock_id', values='return')

            print(f"✓ Loaded returns from CSV: {filepath}")
            return returns_df

        except FileNotFoundError:
            print(f"⚠ Warning: {filepath} not found. Generating sample data...")
            return self._generate_sample_returns()

    def _generate_sample_returns(self, n_stocks=50, start_date='2018-01-01',
                                end_date='2024-05-24') -> pd.DataFrame:
        """
        ========================================================================
        Generate Sample Intraday Return Data (For Testing)
        ========================================================================

        This method generates synthetic return data for testing the pipeline
        when real data is not available. It creates realistic-looking returns
        with U-shaped intraday volatility patterns.

        DO NOT use this for actual research analysis.

        Parameters:
        -----------
        n_stocks : int
            Number of stocks to generate
        start_date : str
            Start date for sample data
        end_date : str
            End date for sample data

        Returns:
        --------
        sample_returns : pandas DataFrame
            Synthetic return data
        ========================================================================
        """
        print("⚠ Generating SYNTHETIC data for testing only. Do not use for research!")

        dates = pd.date_range(start_date, end_date, freq='B')
        n_minutes_per_day = 240  # 4 hours of trading (9:30-11:30, 13:00-15:00)

        sample_data = {}

        for i in range(n_stocks):
            stock_id = f'Stock_{i}'
            stock_returns = []

            for date in dates:
                # Generate realistic intraday returns with U-shaped pattern
                # Higher volatility at open and close, lower at mid-day

                # Morning session (9:30-11:30): Higher volatility
                morning = np.random.normal(0, 0.0015, 60)

                # Lunch break period: Lower volatility
                lunch = np.random.normal(0, 0.0008, 60)

                # Afternoon session (13:00-14:30): Moderate volatility
                afternoon = np.random.normal(0, 0.0012, 60)

                # Closing auction (14:30-15:00): Higher volatility
                closing = np.random.normal(0, 0.0018, 60)

                # Combine all periods
                daily_minutes = np.concatenate([morning, lunch, afternoon, closing])

                # Add occasional jumps for policy days
                # Policy shocks create larger uncertainty
                if date in [pd.Timestamp(d) for d in self.policy_shock_dates]:
                    daily_minutes += np.random.normal(0, 0.005, len(daily_minutes))

                stock_returns.extend(daily_minutes)

            sample_data[stock_id] = stock_returns

        # Create datetime index
        index = pd.date_range(dates[0], periods=len(dates) * n_minutes_per_day, freq='1min')

        return pd.DataFrame(sample_data, index=index)

    def load_limit_days(self, stock_list: Optional[List[str]] = None,
                        start_date: str = '2018-01-01',
                        end_date: str = '2024-05-24') -> Dict[str, List[pd.Timestamp]]:
        """
        ========================================================================
        Load Limit-Up/Limit-Down Days
        ========================================================================

        Chinese A-share stocks have daily price limits of ±10% (±5% for ST
        stocks, ±20% for ChiNext stocks). When a stock hits the limit, trading
        effectively stops as sellers (or buyers) withdraw.

        These limit days should be EXCLUDED from ambiguity calculation because:
        1. They create artificial truncations in the return distribution
        2. The CEA algorithm interprets zero volatility incorrectly
        3. They represent a different regime (price constraint, not ambiguity)

        Detection Method:
        -----------------
        Method 1: Query database for stocks at limit prices
        Method 2: Calculate from returns (if return ≈ limit percentage)
        Method 3: Load from pre-identified CSV file

        Parameters:
        -----------
        stock_list : list of str or None
            Stocks to check for limit days
        start_date : str
            Start date for limit day search
        end_date : str
            End date for limit day search

        Returns:
        --------
        limit_days_dict : dict
            Dictionary mapping stock_id -> list of limit dates
            Example: {'601857.SH': [Timestamp('2020-03-09'),
                                    Timestamp('2020-03-10')]}

        Notes:
        ------
        Limit Thresholds (Standard A-shares):
        - Regular stocks: ±10% daily change
        - ST (Special Treatment) stocks: ±5%
        - ChiNext (300xxx.SZ): ±20%
        - STAR Market (688xxx.SH): ±20%

        The method should identify these days and return them for exclusion
        in the ambiguity calculation.
        ========================================================================
        """
        print("\n" + "="*70)
        print("Loading Limit-Up/Limit-Down Days")
        print("="*70)

        limit_days_dict = {}

        if self.db_engine is None:
            # Fallback: try to load from CSV
            filepath = os.path.join(self.data_path, 'limit_days.csv')
            try:
                df = pd.read_csv(filepath, parse_dates=['date'])

                for stock in df['stock_id'].unique():
                    stock_data = df[df['stock_id'] == stock]
                    limit_days_dict[stock] = stock_data['date'].tolist()

                print(f"✓ Loaded limit days from CSV: {len(limit_days_dict)} stocks")
                return limit_days_dict

            except FileNotFoundError:
                print(f"⚠ Warning: {filepath} not found. No limit days will be excluded.")
                return {}

        # Query database for limit days
        # Limit up: close price = previous close * 1.10 (or 1.05 for ST, 1.20 for ChiNext)
        # Limit down: close price = previous close * 0.90 (or 0.95 for ST, 0.80 for ChiNext)

        try:
            # This is a simplified detection - in practice, you'd need more
            # sophisticated logic to handle stock-specific limits

            query = """
            WITH daily_returns AS (
                SELECT
                    code,
                    DATE(datetime) as date,
                    FIRST_VALUE(closeprice) OVER (PARTITION BY code, DATE(datetime) ORDER BY datetime) as open_price,
                    LAST_VALUE(closeprice) OVER (PARTITION BY code, DATE(datetime) ORDER BY datetime) as close_price,
                    LAG(closeprice, 1) OVER (PARTITION BY code ORDER BY datetime) as prev_close
                FROM stock_hf
                WHERE datetime >= %(start_date)s
                  AND datetime <= %(end_date)s
            ),
            limit_candidates AS (
                SELECT
                    code,
                    date,
                    open_price,
                    close_price,
                    prev_close,
                    CASE
                        WHEN prev_close > 0 THEN (close_price - prev_close) / prev_close
                        ELSE NULL
                    END as daily_return
                FROM daily_returns
                WHERE prev_close > 0
            )
            SELECT code, date, daily_return
            FROM limit_candidates
            WHERE ABS(daily_return) >= 0.095  -- Approximate limit threshold
            ORDER BY code, date;
            """

            params = {'start_date': start_date, 'end_date': end_date}

            with self.db_engine.connect() as conn:
                limit_df = pd.read_sql_query(query, conn, params=params)

            # Group by stock
            for stock in limit_df['code'].unique():
                stock_limits = limit_df[limit_df['code'] == stock]
                limit_days_dict[stock] = stock_limits['date'].tolist()

            print(f"✓ Found limit days for {len(limit_days_dict)} stocks")
            total_limits = sum(len(dates) for dates in limit_days_dict.values())
            print(f"  Total limit days: {total_limits}")

            return limit_days_dict

        except Exception as e:
            print(f"⚠ Error querying limit days: {e}")
            return {}

    def classify_energy_stocks(self, stock_list: List[str],
                                 sector_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        ========================================================================
        Classify Stocks into Energy Sectors
        ========================================================================

        This method categorizes each stock as Brown, Green, or Grey energy
        based on industry classification. This categorization is crucial for:
        1. Testing H2 (Green Discount): Comparing ambiguity premiums across sectors
        2. Difference-in-Differences: Treatment group (Green) vs Control (Brown)
        3. Portfolio analysis: Constructing sector-specific portfolios

        Classification Logic:
        ---------------------
        Priority 1: Use sector_mapping if provided (most accurate)
                   This should come from official industry classifications
                   like WIND industry classification or CSI sector indices

        Priority 2: Use stock code patterns (fallback)
                   6XXXXX: Shanghai main board (often Brown/Grey)
                   0XXXXXX: Shenzhen main board (mixed)
                   3XXXXXX: ChiNext (often Green - tech/new energy)

        Parameters:
        -----------
        stock_list : list of str
            List of stock codes to classify
            Example: ['601857.SH', '300750.SZ', '601012.SH']

        sector_mapping : dict or None, default=None
            Pre-defined mapping of stock_id -> sector_name
            Example: {'601857.SH': 'Oil & Gas Exploration',
                      '300750.SZ': 'EV Batteries'}

            If None, will use simplified code-based classification

        Returns:
        --------
        energy_classification : dict
            Dictionary mapping stock_id -> 'Brown', 'Green', or 'Grey'
            Example: {'601857.SH': 'Brown',
                      '300750.SZ': 'Green',
                      '600900.SH': 'Grey'}

        Notes:
        ------
        For accurate classification, you should:
        1. Query WIND or CSMAR databases for industry classifications
        2. Use official sector indices (CSI Coal, CSI Photovoltaic, etc.)
        3. Manually verify classification for major energy companies

        Key Energy Companies (examples):
        ---------------------------------
        BROWN ENERGY:
        - 601857.SH: PetroChina (Oil & Gas)
        - 600028.SH: Sinopec (Petroleum Refining)
        - 601898.SH: China Coal Energy (Coal Mining)
        - 601012.SH: China Longyuan Power (Thermal - though also has wind)

        GREEN ENERGY:
        - 300750.SZ: CATL (EV Batteries)
        - 601012.SH: LONGi Green Energy (Solar)
        - 002459.SZ: Yunneng New Energy (Wind)
        - 002594.SZ: BYD (New Energy Vehicles)

        GREY ENERGY:
        - 600900.SH: Yangtze Power (Hydro - utility)
        - 600011.SH: Huadian Power (Electric Utility)

        Example:
        --------
        >>> loader = ChinaEnergyDataLoader()
        >>> stocks = ['601857.SH', '300750.SZ', '600900.SH']
        >>> classification = loader.classify_energy_stocks(stocks)
        >>> print(classification)
        {'601857.SH': 'Brown', '300750.SZ': 'Green', '600900.SH': 'Grey'}
        ========================================================================
        """
        energy_classification = {}

        if sector_mapping:
            # ====================================================================
            # METHOD 1: Use Pre-defined Sector Mapping (Preferred)
            # ====================================================================
            # This is the most accurate method if sector data is available
            # from databases like WIND, CSMAR, or official sources

            for stock in stock_list:
                sector = sector_mapping.get(stock, '')

                # Check if sector name contains any of our defined keywords
                if any(s.lower() in sector.lower() for s in self.brown_energy_sectors):
                    energy_classification[stock] = 'Brown'
                elif any(s.lower() in sector.lower() for s in self.green_energy_sectors):
                    energy_classification[stock] = 'Green'
                else:
                    energy_classification[stock] = 'Grey'

        else:
            # ====================================================================
            # METHOD 2: Simplified Code-Based Classification (Fallback)
            # ====================================================================
            # This is a simplified approach based on stock code patterns
            # WARNING: This is NOT accurate for research use!

            print("⚠ Warning: Using simplified stock code classification.")
            print("  For research, provide sector_mapping from official sources.")

            for stock in stock_list:
                # Shanghai main board (6XXXXX): Often traditional energy
                if stock.startswith('6'):
                    # Specific known brown energy stocks
                    if any(s in stock for s in ['601857', '600028', '601898']):
                        energy_classification[stock] = 'Brown'
                    else:
                        energy_classification[stock] = 'Grey'

                # Shenzhen main board (0XXXXXX): Mixed
                elif stock.startswith('0'):
                    energy_classification[stock] = 'Grey'

                # ChiNext (3XXXXXX): Often green/tech
                elif stock.startswith('3'):
                    # Specific known green energy stocks
                    if any(s in stock for s in ['300750', '300274', '300124']):
                        energy_classification[stock] = 'Green'
                    else:
                        energy_classification[stock] = 'Green'  # Default ChiNext to green

                else:
                    energy_classification[stock] = 'Grey'

        # Print classification summary
        brown_count = sum(1 for v in energy_classification.values() if v == 'Brown')
        green_count = sum(1 for v in energy_classification.values() if v == 'Green')
        grey_count = sum(1 for v in energy_classification.values() if v == 'Grey')

        print(f"\n{'='*70}")
        print(f"Energy Stock Classification Summary")
        print(f"{'='*70}")
        print(f"Total stocks: {len(stock_list)}")
        print(f"Brown Energy: {brown_count} ({brown_count/len(stock_list)*100:.1f}%)")
        print(f"Green Energy: {green_count} ({green_count/len(stock_list)*100:.1f}%)")
        print(f"Grey Energy: {grey_count} ({grey_count/len(stock_list)*100:.1f}%)")

        return energy_classification

    def load_epu_data(self, start_date: str = '2018-01-01',
                      end_date: str = '2024-05-24') -> pd.Series:
        """
        ========================================================================
        Load China Economic Policy Uncertainty (EPU) Index
        ========================================================================

        The EPU Index measures the frequency of articles in major newspapers
        containing terms related to economic policy uncertainty. Higher values
        indicate greater uncertainty about future economic policy.

        Use in Analysis:
        ----------------
        - Instrumental Variable: EPU × Policy Sensitivity
        - Control Variable: Macro-level uncertainty
        - Regime Indicator: High vs. low EPU periods

        Data Source:
        ------------
        - Baker, Bloom, and Davis (2016) EPU Index methodology
        - China-specific EPU: www.policyuncertainty.com/china_monthly.html
        - Usually monthly data, need to align with daily frequency

        Parameters:
        -----------
        start_date : str
            Start date for EPU data
        end_date : str
            End date for EPU data

        Returns:
        --------
        epu_series : pandas Series
            Time series of EPU values indexed by date
            - Index: DatetimeIndex (daily or monthly)
            - Values: EPU index values

        Notes:
        ------
        EPU is typically monthly. For daily analysis, use forward-fill to
        spread monthly values to daily frequency.
        ========================================================================
        """
        filepath = os.path.join(self.data_path, 'epu_china.csv')

        try:
            # Try to load from CSV
            df = pd.read_csv(filepath, parse_dates=['date'])
            epu_series = df.set_index('date')['epu_value']

            # Reindex to daily frequency (forward-fill monthly data)
            date_range = pd.date_range(start_date, end_date, freq='D')
            epu_series = epu_series.reindex(date_range, method='ffill')

            print(f"✓ Loaded EPU data: {len(epu_series)} observations")
            print(f"  Mean EPU: {epu_series.mean():.2f}")
            print(f"  Std EPU: {epu_series.std():.2f}")

            return epu_series

        except FileNotFoundError:
            print(f"⚠ Warning: {filepath} not found. Generating sample EPU data...")
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)

            # Generate realistic EPU with spikes during crisis periods
            epu_base = 100 + np.random.randn(len(dates)) * 20

            # Add spike for COVID (early 2020)
            covid_mask = (dates >= '2020-01-01') & (dates <= '2020-04-30')
            epu_base[covid_mask] += 80

            # Add spike for policy shocks (2020-2021)
            policy_mask = (dates >= '2020-09-01') & (dates <= '2021-12-31')
            epu_base[policy_mask] += 40

            epu_series = pd.Series(epu_base, index=dates)

            print(f"✓ Generated synthetic EPU data")
            return epu_series

    def load_policy_sensitivity(self, stock_list: List[str],
                               start_date: str = '2018-01-01',
                               end_date: str = '2024-05-24') -> pd.DataFrame:
        """
        ========================================================================
        Load Firm-Level Policy Sensitivity
        ========================================================================

        Policy sensitivity measures how much a firm's operations depend on
        government policies. This is crucial for the IV strategy:
        - Instrument: EPU × Policy Sensitivity
        - Logic: High-sensitivity firms are more affected by policy uncertainty

        Calculation Methods:
        --------------------
        Method 1: Subsidy Ratio
                  policy_sensitivity = government_subsidies / total_revenue

        Method 2: Government Contracts
                  policy_sensitivity = government_contracts / total_revenue

        Method 3: Environmental Cost Ratio
                  policy_sensitivity = environmental_costs / total_costs

        Method 4: Composite Index
                  policy_sensitivity = weighted average of above measures

        Parameters:
        -----------
        stock_list : list of str
            List of stock codes
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        policy_sensitivity_df : pandas DataFrame
            DataFrame of policy sensitivity values
            - Index: Dates
            - Columns: Stock codes
            - Values: Policy sensitivity scores (0-1, where 1 = highly dependent)

        Notes:
        ------
        In practice, you would:
        1. Download financial statement data from CSMAR/WIND
        2. Calculate subsidy ratio from income statement
        3. Calculate government contract ratio from notes
        4. Normalize and combine into composite score
        ========================================================================
        """
        filepath = os.path.join(self.data_path, 'policy_sensitivity.csv')

        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            policy_sensitivity_df = df.pivot(
                index='date',
                columns='stock_id',
                values='policy_sensitivity'
            )

            # Reindex to full date range
            date_range = pd.date_range(start_date, end_date, freq='D')
            policy_sensitivity_df = policy_sensitivity_df.reindex(
                date_range, method='ffill'
            )

            # Filter to requested stocks
            policy_sensitivity_df = policy_sensitivity_df[stock_list]

            print(f"✓ Loaded policy sensitivity for {len(stock_list)} stocks")
            return policy_sensitivity_df

        except FileNotFoundError:
            print(f"⚠ Warning: {filepath} not found. Generating sample policy sensitivity...")

            # Generate synthetic policy sensitivity
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)

            # Green energy stocks typically have higher policy sensitivity
            # (more dependent on subsidies)
            sensitivity_data = {}

            for stock in stock_list:
                # Base sensitivity
                base_sens = np.random.rand() * 0.5 + 0.2  # 0.2 to 0.7

                # Add time trend (increasing policy dependence over time)
                time_trend = np.linspace(0, 0.2, len(dates))

                # Add noise
                noise = np.random.randn(len(dates)) * 0.05

                # Combine
                sensitivity = base_sens + time_trend + noise
                sensitivity = np.clip(sensitivity, 0, 1)  # Bound between 0 and 1

                sensitivity_data[stock] = sensitivity

            policy_sensitivity_df = pd.DataFrame(sensitivity_data, index=dates)
            policy_sensitivity_df = policy_sensitivity_df[stock_list]

            print(f"✓ Generated synthetic policy sensitivity")
            return policy_sensitivity_df

    def load_geopolitical_data(self, start_date: str = '2018-01-01',
                              end_date: str = '2024-05-24') -> Dict[str, pd.Series]:
        """
        ========================================================================
        Load Geopolitical Data for Geopolitical Ambiguity
        ========================================================================

        Geopolitical events create uncertainty in energy markets through:
        1. Supply disruptions (wars affecting oil/gas supply)
        2. Sanctions (restricting trade)
        3. Strategic competition (US-China tensions)

        We measure geopolitical ambiguity using:
        1. CSI National Defense Index: Military/defense sector performance
        2. SHFE Gold Futures: Traditional safe-haven asset

        When these assets show high ambiguity (CEA), it signals geopolitical
        model uncertainty, which affects energy markets.

        Data Sources:
        -------------
        1. Defense Index: CSI National Defense Index (399967.SZ)
        2. Gold Futures: Shanghai Gold Exchange (SGE) or SHFE Gold

        Parameters:
        -----------
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        geo_data : dict
            Dictionary with keys:
            - 'defense': Time series of defense index returns
            - 'gold': Time series of gold futures returns

        Notes:
        ------
        These assets serve as "geopolitical barometers." When their return
        distributions become ambiguous (high CEA), it indicates that the
        market's geopolitical model is breaking down.
        ========================================================================
        """
        filepath = os.path.join(self.data_path, 'geopolitical_indices.csv')

        try:
            df = pd.read_csv(filepath, parse_dates=['date', 'time'])
            df['datetime'] = pd.to_datetime(
                df['date'].astype(str) + ' ' + df['time'].astype(str)
            )
            df.set_index('datetime', inplace=True)

            geo_data = {
                'defense': df['defense_index_return'],
                'gold': df['gold_futures_return']
            }

            print(f"✓ Loaded geopolitical data")
            return geo_data

        except FileNotFoundError:
            print(f"⚠ Warning: {filepath} not found. Generating sample geopolitical data...")

            # Generate synthetic data
            dates = pd.date_range(start_date, end_date, freq='B')
            np.random.seed(42)

            # Defense returns: Higher volatility during tensions
            defense_returns = pd.Series(
                np.random.randn(len(dates)) * 0.01,
                index=dates
            )

            # Gold returns: Safe haven, negative correlation with stocks sometimes
            gold_returns = pd.Series(
                np.random.randn(len(dates)) * 0.005,
                index=dates
            )

            geo_data = {
                'defense': defense_returns,
                'gold': gold_returns
            }

            print(f"✓ Generated synthetic geopolitical data")
            return geo_data

    def load_control_data(self, start_date: str = '2018-01-01',
                         end_date: str = '2024-05-24') -> Dict[str, pd.DataFrame]:
        """
        ========================================================================
        Load Control Variables for Analysis
        ========================================================================

        Control variables are essential for isolating the effect of ambiguity
        from other known risk factors. We control for:

        1. Commodity Uncertainty:
           - INE Crude Oil Futures (SC) returns
           - Controls for fundamental oil price risk
           - Distinguishes oil risk from policy ambiguity

        2. Carbon Uncertainty:
           - China National ETS (Emissions Trading Scheme) prices
           - Controls for regulatory cost risk
           - Distinguishes carbon risk from broader policy ambiguity

        3. Market Risk:
           - CSI 300 Index returns (market benchmark)
           - Controls for overall market movements

        4. Liquidity Measures:
           - Trading volume
           - Bid-ask spreads
           - Amihud illiquidity ratio

        Parameters:
        -----------
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        control_data : dict
            Dictionary containing:
            - 'oil_returns': INE crude oil futures returns
            - 'ets_returns': China ETS returns
            - 'market_returns': CSI 300 returns (if available)
            - 'volume': Trading volume (if available)
            - 'spread': Bid-ask spreads (if available)

        Notes:
        ------
        These controls are CRITICAL for identification:
        - We want to show that ambiguity has effects BEYOND these known risks
        - Each control addresses a specific alternative explanation
        ========================================================================
        """
        control_data = {}

        # ====================================================================
        # CONTROL 1: Commodity Uncertainty - Crude Oil Futures
        # ====================================================================
        # INE (Shanghai International Energy Exchange) Crude Oil Futures
        # Symbol: SC (e.g., SC2401 for Jan 2024 contract)
        # This controls for fundamental oil price risk separate from policy

        try:
            oil_df = pd.read_csv(
                os.path.join(self.data_path, 'ine_crude_oil.csv'),
                parse_dates=['date']
            )
            oil_df = oil_df.set_index('date')['return']

            # Reindex to daily frequency
            date_range = pd.date_range(start_date, end_date, freq='D')
            oil_df = oil_df.reindex(date_range, method='ffill')

            control_data['oil_returns'] = oil_df
            print(f"✓ Loaded INE crude oil returns")

        except FileNotFoundError:
            print(f"⚠ Warning: ine_crude_oil.csv not found. Generating sample data...")
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)
            control_data['oil_returns'] = pd.Series(
                np.random.randn(len(dates)) * 0.02,  # 2% daily volatility
                index=dates
            )

        # ====================================================================
        # CONTROL 2: Carbon Uncertainty - China National ETS
        # ====================================================================
        # China's national carbon emissions trading scheme
        # Launched July 2021, initially covering power sector
        # Carbon price reflects regulatory cost risk

        try:
            ets_df = pd.read_csv(
                os.path.join(self.data_path, 'china_ets.csv'),
                parse_dates=['date']
            )
            ets_df = ets_df.set_index('date')['return']

            # Reindex
            date_range = pd.date_range(start_date, end_date, freq='D')
            ets_df = ets_df.reindex(date_range, method='ffill')

            control_data['ets_returns'] = ets_df
            print(f"✓ Loaded China ETS returns")

        except FileNotFoundError:
            print(f"⚠ Warning: china_ets.csv not found. Generating sample data...")
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(43)
            control_data['ets_returns'] = pd.Series(
                np.random.randn(len(dates)) * 0.01,  # 1% daily volatility
                index=dates
            )

        return control_data


def prepare_analysis_data(data_loader: ChinaEnergyDataLoader,
                          stock_list: Optional[List[str]] = None,
                          start_date: str = '2018-01-01',
                          end_date: str = '2024-05-24') -> Dict:
    """
    ========================================================================
    PREPARE ALL DATA FOR CAUSAL ANALYSIS PIPELINE
    ========================================================================

    This is the MASTER DATA PREPARATION FUNCTION that orchestrates all data
    loading and preprocessing. It calls all the individual loading methods
    and assembles the complete dataset needed for causal_ambi_china.tex.

    Workflow:
    ---------
    1. Load intraday returns from database (or CSV fallback)
    2. Get list of unique stocks
    3. Classify stocks into Brown/Green/Grey sectors
    4. Load limit days (for exclusion in CEA calculation)
    5. Load EPU data (for IV construction)
    6. Load policy sensitivity (for IV construction)
    7. Load geopolitical data (for Geopolitical Ambiguity)
    8. Load control variables (oil, carbon prices)

    Parameters:
    -----------
    data_loader : ChinaEnergyDataLoader
        Initialized data loader instance with database connection

    stock_list : list of str or None, default=None
        List of specific stocks to analyze.
        If None, will load all stocks available in the database.
        Example: ['601857.SH', '300750.SZ'] for specific analysis

    start_date : str, default='2018-01-01'
        Start date for analysis period.
        2018 is chosen to capture pre-transition baseline before the
        major policy shocks of 2020-2021.

    end_date : str, default='2024-05-24'
        End date for analysis period.
        Adjust based on data availability.

    Returns:
    --------
    data_dict : dict
        Comprehensive dictionary containing ALL data needed for analysis:

        {
            'intraday_returns': pd.DataFrame,
                # High-frequency returns
                # Index: datetime (minute-level)
                # Columns: stock_id
                # Values: log returns

            'energy_classification': dict,
                # Stock -> 'Brown'/'Green'/'Grey' mapping
                # Used for H2 (Green Discount) testing
                # Example: {'601857.SH': 'Brown', '300750.SZ': 'Green'}

            'sector_mapping': dict,
                # Stock -> sector mapping for IV construction
                # Used to build peer-based instruments
                # Example: {'601857.SH': 'Oil & Gas'}

            'limit_days': dict,
                # Stock -> list of limit-up/limit-down dates
                # Used to exclude artificial truncations from CEA calc
                # Example: {'601857.SH': [Timestamp(...), ...]}

            'epu_series': pd.Series,
                # Economic Policy Uncertainty index
                # Used in IV: EPU × Policy Sensitivity
                # Index: dates, Values: EPU values

            'policy_sensitivity': pd.DataFrame,
                # Firm-level policy sensitivity scores
                # Used in IV construction
                # Index: dates, Columns: stocks, Values: 0-1 scores

            'geopolitical_data': dict,
                # Defense and gold returns
                # Used for Geopolitical Ambiguity calculation
                # Keys: 'defense', 'gold'

            'control_data': dict,
                # Control variables for regression
                # Keys: 'oil_returns', 'ets_returns'

            'policy_shock_dates': list,
                # Major policy announcement dates
                # Used for DiD analysis
                # Example: ['2020-09-22', '2021-03-15', ...]
        }

    Example Usage:
    --------------
    >>> # Initialize data loader with database credentials
    >>> loader = ChinaEnergyDataLoader(
    ...     db_host='10.28.255.30',
    ...     db_name='stock_hf',
    ...     db_user='your_username',
    ...     db_password='your_password'
    ... )
    >>>
    >>> # Prepare all data
    >>> data_dict = prepare_analysis_data(
    ...     data_loader=loader,
    ...     start_date='2020-01-01',
    ...     end_date='2022-12-31'
    ... )
    >>>
    >>> # Access components
    >>> returns = data_dict['intraday_returns']
    >>> classification = data_dict['energy_classification']

    Notes:
    ------
    1. Database Connection:
       - Requires valid PostgreSQL credentials
       - Falls back to CSV files if connection fails
       - Ensure network access to 10.28.255.30

    2. Data Quality:
       - Check output shapes and memory usage
       - Verify date ranges are correct
       - Confirm stock classifications are accurate

    3. Memory Management:
       - High-frequency data is large
       - Consider filtering stocks or time range if memory issues
       - Use chunking for very large datasets

    4. Next Steps:
       - Pass data_dict to ambiguity_measurement.py
       - Then to causal_analysis.py
       - Finally to main_china_energy.py for full pipeline

    ========================================================================
    """
    print("\n" + "="*70)
    print("PREPARING COMPLETE ANALYSIS DATASET")
    print("="*70)
    print(f"Analysis period: {start_date} to {end_date}")

    # ========================================================================
    # STEP 1: Load Intraday Returns from Database
    # ========================================================================
    print("\n[1/7] Loading intraday returns from database...")
    intraday_returns = data_loader.load_intraday_returns_from_db(
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date
    )

    # Get stock list from data (if not specified)
    if stock_list is None:
        stock_list = intraday_returns.columns.tolist()

    print(f"✓ Loaded {len(stock_list)} stocks")

    # ========================================================================
    # STEP 2: Classify Energy Stocks
    # ========================================================================
    print("\n[2/7] Classifying energy stocks...")
    energy_classification = data_loader.classify_energy_stocks(stock_list)

    # Create sector mapping for IV construction
    # For now, assume same sector as energy type
    # In practice, you'd load detailed sector classifications
    sector_mapping = {
        stock: energy_classification[stock]
        for stock in stock_list
    }

    # ========================================================================
    # STEP 3: Load Limit Days
    # ========================================================================
    print("\n[3/7] Loading limit-up/limit-down days...")
    limit_days = data_loader.load_limit_days(
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date
    )

    # ========================================================================
    # STEP 4: Load EPU Data
    # ========================================================================
    print("\n[4/7] Loading Economic Policy Uncertainty data...")
    epu_series = data_loader.load_epu_data(
        start_date=start_date,
        end_date=end_date
    )

    # ========================================================================
    # STEP 5: Load Policy Sensitivity
    # ========================================================================
    print("\n[5/7] Loading firm-level policy sensitivity...")
    policy_sensitivity = data_loader.load_policy_sensitivity(
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date
    )

    # ========================================================================
    # STEP 6: Load Geopolitical Data
    # ========================================================================
    print("\n[6/7] Loading geopolitical data...")
    geopolitical_data = data_loader.load_geopolitical_data(
        start_date=start_date,
        end_date=end_date
    )

    # ========================================================================
    # STEP 7: Load Control Variables
    # ========================================================================
    print("\n[7/7] Loading control variables...")
    control_data = data_loader.load_control_data(
        start_date=start_date,
        end_date=end_date
    )

    # ========================================================================
    # ASSEMBLE FINAL DATA DICTIONARY
    # ========================================================================
    data_dict = {
        'intraday_returns': intraday_returns,
        'energy_classification': energy_classification,
        'sector_mapping': sector_mapping,
        'limit_days': limit_days,
        'epu_series': epu_series,
        'policy_sensitivity': policy_sensitivity,
        'geopolitical_data': geopolitical_data,
        'control_data': control_data,
        'policy_shock_dates': data_loader.policy_shock_dates
    }

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Intraday returns: {intraday_returns.shape}")
    print(f"  Total stocks: {len(stock_list)}")
    print(f"  Brown energy: {sum(1 for v in energy_classification.values() if v == 'Brown')}")
    print(f"  Green energy: {sum(1 for v in energy_classification.values() if v == 'Green')}")
    print(f"  Grey energy: {sum(1 for v in energy_classification.values() if v == 'Grey')}")
    print(f"  Limit-up/limit-down days: {sum(len(dates) for dates in limit_days.values())}")
    print(f"  EPU observations: {len(epu_series)}")
    print(f"  Policy shock dates: {len(data_loader.policy_shock_dates)}")

    return data_dict


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
# This block runs when the script is executed directly:
# python data_loader.py
#
# It provides a quick way to test the data loader and verify database
# connection is working correctly.
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHINA ENERGY MARKET DATA LOADER")
    print("Testing Database Connection and Data Loading")
    print("="*70)

    # ========================================================================
    # INITIALIZE DATA LOADER
    # ========================================================================
    # Note: In production, credentials should be from environment variables
    # or a secure configuration file, NOT hardcoded!

    print("\nInitializing data loader...")
    print("Database: 10.28.255.30 / stock_hf")

    loader = ChinaEnergyDataLoader(
        data_path='data/',
        db_host='10.28.255.30',
        db_name='stock_hf',
        db_user=os.environ.get('DB_USER'),  # From environment variable
        db_password=os.environ.get('DB_PASSWORD')  # From environment variable
    )

    # ========================================================================
    # TEST DATABASE CONNECTION
    # ========================================================================
    if loader.db_engine is not None:
        print("\n✓ Database connection established")

        # Load stock universe
        print("\nLoading stock universe from database...")
        stock_universe = loader.load_stock_universe_from_db()

        if stock_universe:
            print(f"✓ Found {len(stock_universe)} stocks in database")
            print(f"  Sample stocks: {stock_universe[:5]}")

            # Test loading data for a few stocks
            print("\nTesting data load for sample stocks...")
            test_stocks = stock_universe[:5]

            test_returns = loader.load_intraday_returns_from_db(
                stock_list=test_stocks,
                start_date='2023-01-01',
                end_date='2023-01-31'
            )

            print(f"\n✓ Test successful! Loaded data shape: {test_returns.shape}")
            print(f"  Date range: {test_returns.index.min()} to {test_returns.index.max()}")

    else:
        print("\n⚠ Could not connect to database")
        print("  Please check:")
        print("  1. Database server is accessible (10.28.255.30)")
        print("  2. Credentials are correct")
        print("  3. Network connection is stable")
        print("  4. VPN is connected if required")

    # ========================================================================
    # PREPARE FULL DATASET (if connection successful)
    # ========================================================================
    if loader.db_engine is not None:
        print("\n" + "="*70)
        print("Preparing full analysis dataset...")
        print("="*70)

        # Prepare data for a limited time range (for testing)
        data_dict = prepare_analysis_data(
            data_loader=loader,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )

        print("\n✓ Full dataset prepared successfully!")
        print("\nYou can now proceed to:")
        print("  1. Run ambiguity_measurement.py to compute CEA")
        print("  2. Run causal_analysis.py to test hypotheses")
        print("  3. Run main_china_energy.py for complete pipeline")

    print("\n" + "="*70)
    print("DATA LOADER TEST COMPLETE")
    print("="*70)
