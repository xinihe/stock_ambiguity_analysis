# Data Requirements & Source Guide: Chinese Energy Market Ambiguity

This guide details the specific data requirements for the "Ambiguity in Chinese Energy Transition" research project. Since you already possess the high-frequency A-share stock data, this document focuses on the **Explanatory Variables** and **Sector-Specific Data**.

## 1. Primary Data (You Already Have)
*   **A-Share High-Frequency Data**:
    *   *Content*: 1-minute or 5-minute tick data (Price, Volume).
    *   *Coverage*: All stocks in CSI 300 Energy, CSI New Energy, and Utilities sectors.
    *   *Usage*: Calculating the core **CEA Ambiguity Index** ($CEA_{i,t}$).

## 2. Required Explanatory Variables (China Energy Specific)

To disentangle "Ambiguity" from "Risk" and "Fundamental News," you need the following external variables.

### A. Commodity & Fundamental Uncertainty
These variables control for the "Risk" (Variance) component, allowing your CEA index to capture the "Ambiguity" (Model Uncertainty) component.

| Variable Name | Description | Ticker / Code | Recommended Source | Frequency |
| :--- | :--- | :--- | :--- | :--- |
| **China Crude Oil Futures** | INE Shanghai Crude Oil Futures (Dominant Contract) | **SC.INE** (or `SC00.INE`) | **Wind Terminal**, Choice, or CSMAR | Daily / Intraday |
| **Thermal Coal Futures** | Zhengzhou Commodity Exchange Thermal Coal | **ZC.CZC** | Wind / Choice | Daily |
| **Spot Oil Price** | Daqing or Shengli Spot Price (Local benchmark) | N/A | Wind (Commodity DB) | Daily |

### B. Policy & Transition Ambiguity (Crucial)
These are your key "Ambiguity Shocks" for the identification strategy.

| Variable Name | Description | Source / Access | Notes |
| :--- | :--- | :--- | :--- |
| **China National ETS** | Carbon Emission Allowance prices (CEA) from Shanghai Environment & Energy Exchange. | **Wind**: `EXT_SHEEX_CEA` | The national market launched in July 2021. For pre-2021, use pilot markets (Hubei/Guangdong). |
| **China EPU Index** | Economic Policy Uncertainty Index for China (Baker et al. methodology). | [PolicyUncertainty.com](https://www.policyuncertainty.com/china_monthly.html) | **Free**. Use the "Mainland China" monthly index. |
| **Climate Policy Index** | CPU (Climate Policy Uncertainty) Index. | [ClimatePolicyUncertainty.org](https://www.policyuncertainty.com/climate_uncertainty.html) | **Free**. Check if a China-specific sub-index is available; if not, use Global. |

### C. Market Microstructure & Risk Controls
Standard controls to ensure your results aren't driven by illiquidity.

| Variable | Description | Source |
| :--- | :--- | :--- |
| **Fama-French 3 Factors (China)** | SMB, HML, Mkt-RF specific to A-shares. | **CSMAR** (Factor Database) or RESSET |
| **Margin Trading** | Margin Buy / Short Sell balances for energy stocks. | **CSMAR** or Wind (`Rzrq` table) |
| **Analyst Coverage** | Number of analyst reports / dispersion of forecasts. | **CSMAR** (Analyst Forecasts) |

## 3. Where to Get This Data (Access Strategy)

### Option 1: Institutional Terminals (Wind / Choice)
If you are at a Chinese university or financial institution, you likely have access to **Wind Financial Terminal (WFT)** or **EastMoney Choice**.
*   **Action**: Use the Excel Add-in or Python API (`WindPy`) to download:
    *   `SC.INE` (Close, High, Low, Vol)
    *   Carbon Prices (Search "National ETS")
    *   Commodity Futures

### Option 2: Database Downloads (CSMAR / RESSET)
*   **CSMAR (GTA)**:
    *   Go to "Economy" -> "Macroeconomy" for EPU proxies.
    *   Go to "Factor Research" for Fama-French factors.
    *   Go to "Futures" for INE Crude Oil data.
*   **RESSET**:
    *   Good alternative for high-frequency data validation if needed.

### Option 3: Public/Free Sources (If Terminal Access is Limited)
1.  **PolicyUncertainty.com**: Download the China EPU Excel file directly.
2.  **Sina Finance (新浪财经)**: Historical data for Futures (SC0) is often scrapable or available via `Tushare` (Python package).
3.  **Tushare Pro**:
    *   A popular Python data interface for Chinese markets.
    *   *Cost*: Free (basic) to Low Cost (pro).
    *   *Code*:
        ```python
        import tushare as ts
        pro = ts.pro_api('your_token')
        # Get Daily Carbon Prices
        df = pro.opt_daily(ts_code='...') # Check specific Tushare docs for Carbon
        ```

## 4. Data Construction Checklist

1.  **[ ] Match Frequencies**: Ensure all your control variables (Daily) can be mapped to your Ambiguity Index (Daily, aggregated from Minutely).
2.  **[ ] Handle Trading Hours**: INE Crude Oil (SC) has night trading sessions. Decide whether to include night session volatility in your "Daily" control. *Recommendation: Use Close-to-Close to capture overnight ambiguity.*
3.  **[ ] Currency**: Ensure Oil prices (if using Brent/WTI as global control) are converted to RMB or used as log-returns to be unit-invariant.

## 5. Summary of Recommended Action
1.  **Use Wind/Choice** to get **INE Crude Oil** and **Carbon (ETS)** data.
2.  **Use PolicyUncertainty.com** for the **EPU Index**.
3.  **Use CSMAR** for **Fama-French Factors** and **Analyst Coverage**.
