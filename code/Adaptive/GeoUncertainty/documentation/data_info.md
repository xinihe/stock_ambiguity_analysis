# Data Information Document

This document describes two main datasets extracted and processed from Excel files: Climate Risk Data and Geopolitical Risk Country Data.

## 1. Climate Risk Data

### Data Source
The original data was sourced from the Climate_Risk_Index.xlsx file, specifically from the 'Climate Risk Data' worksheet.

### Data Processing Steps
1. Extract climate risk data from the Excel file
2. Skip the first 6 descriptive text lines and directly read the actual data
3. Rename columns for better understanding
4. Ensure correct date formatting
5. Filter out data before January 1, 2005
6. Delete all columns with empty values
7. Convert non-numeric values to standard missing value format
8. Save as a tabular format file

### Data Overview
- **Number of Rows**: 5343
- **Number of Columns**: 7
- **Date Range**: January 3, 2005 to June 30, 2025

### Data Column Description
- **Date**: Date (YYYY-MM-DD format)
- **Physical_Risk_Index**: Physical Risk Index
- **Physical_Risk_Change**: Physical Risk Change Rate
- **Transition_Risk_Index**: Transition Risk Index
- **Transition_Risk_Change**: Transition Risk Change Rate
- **Climate_Policy_Risk**: Climate Policy Risk Indicator
- **Market_Sentiment_Risk**: Market Sentiment Risk Indicator

## 2. Geopolitical Risk Country Data

### 2.1 Original Geopolitical Risk Data

#### Data Source
The original data was sourced from the geopolitical risk data in the data_geopolitical_risk_export.xls file.

#### Data Processing Steps
1. Extract geopolitical risk data from the Excel file
2. Filter data for four countries/regions: China, Hong Kong Special Administrative Region, Japan, and the United States
3. Rename columns for clarity
4. Exclude explanatory lines, keeping only actual data
5. Ensure correct date formatting
6. Save as a tabular format file

#### Data Overview
- **Number of Rows**: 1509
- **Number of Columns**: 5
- **Date Range**: January 1, 1900 to September 1, 2025

### 2.2 Filtered Geopolitical Risk Data

#### Data Source
Derived from filtering the original geopolitical risk data.

#### Data Processing Steps
1. Read the original geopolitical risk country data
2. Ensure correct date formatting
3. Filter out data before January 1, 2005
4. Save as a new tabular format file

#### Data Overview
- **Number of Rows**: 249
- **Number of Columns**: 5
- **Date Range**: January 1, 2005 to September 1, 2025
- **Filtering Information**: A total of 1260 rows of data (from 1900 to 2004) were filtered out

### Data Column Description
- **Date**: Date (YYYY-MM-DD format)
- **China**: China Geopolitical Risk Index
- **Hong Kong**: Hong Kong Special Administrative Region Geopolitical Risk Index
- **Japan**: Japan Geopolitical Risk Index
- **US**: United States Geopolitical Risk Index
