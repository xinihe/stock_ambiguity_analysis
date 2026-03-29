# Research Proposal: Ambiguity in the Energy Transition
## "Pricing the Unknown: Ambiguity Premiums in Green vs. Brown Energy Assets"

### 1. Strategic Rationale
Focusing on the Energy industry maximizes the impact of the Ambiguity framework because this sector faces the highest level of **"Structural Uncertainty"** (the definition of Ambiguity):
*   **Technological Ambiguity**: Will Hydrogen, Nuclear, or Solar win?
*   **Policy Ambiguity**: Carbon taxes, subsidies, and net-zero timelines are constantly shifting.
*   **Geopolitical Ambiguity**: Dependence on OPEC+, wars, and supply chain weaponization.

**Core Argument**: The "Greenium" (lower returns for green stocks) and the "Carbon Premium" (higher returns for brown stocks) are not just about *Risk* (volatility), but about *Ambiguity* (uncertainty about the future state of the world).

### 2. Specific Features & Data Requirements

If we concentrate on Energy, we must introduce sector-specific variables that capture these unique ambiguity sources.

#### A. New Data Variables
| Variable Category | Specific Metrics | Source |
| :--- | :--- | :--- |
| **Sector Ambiguity** | **OVX (CBOE Crude Oil Volatility Index)**: The "VIX" of the oil market. | CBOE / Bloomberg |
| **Policy Ambiguity** | **Carbon Futures (EUA) Volatility**: Uncertainty in the price of emissions in Europe. | ICE / ECX |
| **Climate Uncertainty** | **Climate Policy Uncertainty (CPU) Index**: Text-based index of climate regulation news. | *Existing in project* |
| **Asset Classification** | **Green vs. Brown Revenue Share**: Classify firms not just by industry, but by "Greenness". | MSCI / Sustainalytics |

#### B. Methodological Adaptations
The general model ($r = \alpha + \beta Ambiguity$) needs to be expanded to test for **Asymmetry**:

$$r_{i,t+1} = \alpha + \beta_1 Ambiguity_{i,t} + \beta_2 (Ambiguity_{i,t} \times BrownDummy_i) + \dots$$

*   **Hypothesis**: $\beta_2 > 0$. Brown firms (Oil & Gas) face *negative* ambiguity (fear of obsolescence), leading to a higher premium than Green firms, which might face *speculative* ambiguity.

### 3. Detailed Execution Outline

#### Phase 1: Universe Construction & Classification
1.  **Select Universe**: S&P 500 Energy Sector + S&P Global Clean Energy Index components.
2.  **Classification**: Split firms into three buckets:
    *   **Brown**: Pure-play Oil & Gas (e.g., Exxon, Chevron).
    *   **Grey**: Transitioning firms / Utilities (e.g., NextEra, BP).
    *   **Green**: Pure Renewables (e.g., SolarEdge, Vestas).

#### Phase 2: Constructing "Energy Ambiguity"
Instead of just using intraday returns, construct a **Composite Energy Ambiguity Index**:
1.  Compute standard `CEA_Ambiguity` (your current measure) for all energy stocks.
2.  Correlate it with **OVX** (Oil VIX) to validate it captures sector stress.
3.  **Event Study**: Plot the Ambiguity Index around key dates:
    *   Paris Agreement (2015).
    *   COVID-19 Oil Crash (April 2020).
    *   Russia-Ukraine War (2022).

#### Phase 3: Identification Strategy (Energy Specific)
Use **Exogenous Supply Shocks** as Instruments (IVs):
1.  **OPEC+ Announcements**: Unexpected supply cuts/hikes are pure ambiguity shocks for the sector.
2.  **Climate Summits (COP)**: News from COP meetings creates "Policy Ambiguity" shocks that are exogenous to firm fundamentals.

#### Phase 4: The "Stranded Asset" Test
Test if Ambiguity explains the valuation gap between Green and Brown companies.
*   **Regression**: Does High Ambiguity cause a widening of the spread between Green and Brown P/E ratios?

### 4. Expected Contributions
1.  **Disentangling Risk vs. Uncertainty in Energy**: Show that the "Carbon Premium" is actually an "Ambiguity Premium."
2.  **Policy Implication**: Policymakers can reduce the cost of capital for green energy by reducing *policy ambiguity* (clearer rules), not just by subsidies.

### 5. Next Steps for Implementation
1.  **Filter Data**: Subset your current dataset to Energy (GICS Code 10) and Utilities (GICS Code 55).
2.  **Ingest External Data**: Download OVX and CPU (Climate Policy Uncertainty) data.
3.  **Run Pilot**: Re-run the `ambiguity_measurement.py` script *only* on the Energy subset to see if the signal-to-noise ratio improves.
