# Research Proposal: Ambiguity in the Chinese Energy Transition
## "Pricing the Unknown: Ambiguity Premiums in China's Green vs. Brown Energy Markets"

### 1. Strategic Rationale (China Context)
China is the ideal laboratory for this research due to the "Dual Carbon" goals (Peaking Carbon by 2030, Carbon Neutrality by 2060). This massive structural shift creates intense **Model Uncertainty (Ambiguity)** for investors trying to value energy assets.

**Core Argument**: In the Chinese A-share market, the divergence in returns between Traditional Energy (Coal/Oil) and New Energy (Solar/Wind/EV) is driven by **Ambiguity Aversion** measured by your CEA index, not just fundamental risk.

### 2. Ambiguity Measurement Framework (CEA-Centric)

All ambiguity metrics in this study are derived using your **Cross-Entropy Ambiguity (CEA)** algorithm based on high-frequency intraday returns. We do not rely on "black box" external volatility indices.

#### A. The Hierarchy of CEA Measures
1.  **Firm-Level Ambiguity ($CEA_{i,t}$)**:
    *   *Input*: 1-minute high-frequency returns for stock $i$ (e.g., PetroChina 601857.SH, Longi Green Energy 601012.SH).
    *   *Algorithm*: Your entropy-based KL divergence measure (Variance of Variance).
    *   *Meaning*: How much the intraday return distribution of this specific firm deviates from its "expected" distribution.

2.  **Sector Ambiguity ($CEA_{Sector,t}$)**:
    *   *Construction*: Value-weighted average of Firm-Level CEA for all stocks in the CSI Energy / CSI New Energy sectors.
    *   *Meaning*: The aggregate level of "Model Uncertainty" for the entire Chinese energy industry.

3.  **Composite Energy Ambiguity (PCA-based)**:
    *   *Measurement*: The first principal component (PCA) extracted from the CEA time series of all energy firms.
    *   *Meaning*: Captures the "common component" of model uncertainty across the industry, derived entirely from high-frequency return data without relying on external indices. This distinguishes systematic ambiguity from firm-specific idiosyncratic ambiguity.
    *   *Formula*: $CEA_{Composite,t} = \sum_{i=1}^{N} \lambda_i \cdot CEA_{i,t}$, where $\lambda_i$ are the principal component loadings.

4.  **Policy Ambiguity (Proxy)**:
    *   *Measurement*: The CEA calculated on the **CSI 300 Energy Index ETF** or **Shanghai Crude Oil Futures (INE SC)** high-frequency data.
    *   *Logic*: When policy is unclear, the *index itself* exhibits the high-frequency distributional shifts that your algorithm detects.

5.  **Geopolitical Ambiguity (New)**:
    *   *Rationale*: Energy markets are uniquely sensitive to war and diplomatic tension (e.g., Russia-Ukraine, Middle East).
    *   *Measurement*: The CEA calculated on **defense/military stocks** (e.g., CSI National Defense Index) or **gold futures** (SHFE Gold).
    *   *Logic*: Defense stocks and Gold are pure "geometers." When their intraday return distributions become ambiguous (high CEA), it signals that the market's "geopolitical model" is breaking down.
    *   *Combined Measure*: $GeoAmbiguity_t = PC1(\text{Defense CEA}_t, \text{Gold CEA}_t)$ - first principal component of Defense and Gold CEA.

### 3. Data Requirements (Chinese Market)

#### A. High-Frequency Data Sources
*   **Universe**: Components of **CSI 300 Energy** (Traditional) and **CSI New Energy** (Green).
*   **Frequency**: 1-minute or 5-minute tick data.
*   **Exchange**: Shanghai (SSE) and Shenzhen (SZSE).

#### B. Explanatory Variables (China Specific)

| Variable | Proxy in China | Classification | Role in Model |
| :--- | :--- | :--- | :--- |
| **Commodity Uncertainty** | **INE Crude Oil Futures (SC)** | **Control Variable** | Controls for fundamental oil price risk (volatility) to isolate pure ambiguity. |
| **Carbon Uncertainty** | **China National ETS Prices** | **Control Variable** | Controls for regulatory cost risk so it is not confused with policy ambiguity. |
| **Policy Shocks** | **EPU China Index** | **Instrumental Variable (IV)** | Used in 1st-stage regression to isolate exogenous variation in Ambiguity. |
| **Geopolitical Ambiguity** | **CEA of CSI National Defense** | **Independent Variable** | A systematic ambiguity factor (beta) that affects all energy stocks. |

### 4. Identification Strategy (Adapted for China)

#### A. Instrumental Variable: "Peer-Based CEA"
*   **Logic**: Instrument the CEA of *China Shenhua Energy* using the average CEA of other *Coal* companies in the A-share market.
*   **Why it works**: Coal companies share regulatory ambiguity (e.g., "supply side reform" policies) but have idiosyncratic operational risks.

#### B. Natural Experiments (Policy Shocks)
Use "Difference-in-Differences" around key "Dual Carbon" announcements:
*   *Event*: President Xi's 2020 UN speech (2060 Neutrality Pledge).
*   *Test*: Did the **CEA** of Green firms drop relative to Brown firms? Did the **Ambiguity Premium** change sign?

### 5. Execution Roadmap

#### Phase 1: Data Preparation
1.  **Select Stocks**: Filter A-share list for GICS Energy + Utilities + Electrical Equipment (Solar/Wind).
2.  **Classify**: Tag as "Green" (Renewables), "Brown" (Coal/Oil), or "Grey" (Grid/Utilities).
3.  **Clean HF Data**: Remove limit-up/limit-down days (common in China) to avoid artificial zero-volatility periods affecting entropy calc.

#### Phase 2: Computing CEA
1.  Run `ambiguity_measurement.py` on the Chinese HF data.
2.  **Validation**: Check if CEA spikes during the 2015 market crash and the 2021 power crunch.

#### Phase 3: Empirical Analysis
1.  **Panel Regression**:
    $$r_{i,t+1} = \alpha + \beta_1 CEA_{i,t} + \beta_2 (CEA_{i,t} \times GreenDummy_i) + \beta_3 GeoAmbiguity_t + Controls$$
    *   *Hypothesis*: $\beta_1 > 0$ (Ambiguity is priced), $\beta_2 < 0$ (Green firms have lower ambiguity premiums due to policy support).

2.  **Portfolio Sorts**:
    *   Form "High Ambiguity" vs. "Low Ambiguity" portfolios within the Energy sector.
    *   Check if the "High - Low" spread generates significant alpha in the A-share market.

### 6. Expected Contribution
*   First paper to apply **High-Frequency Entropy Measures** to the **Chinese Energy Transition**.
*   Demonstrates that "Green Finance" in China is partly about reducing *Ambiguity* (making the future model clearer), not just subsidizing *Risk*.
