# List of Changes - geoAmb03_article_rev01.tex

## Changes Sorted by Location

### 1. Abstract Section (Lines 27-29)
**Addition:**
- Added paragraph highlighting distinction between risk and ambiguity channels with quantitative metrics
- Added key findings: (i) GPR increases ambiguity more than volatility (0.42 vs 0.18 std deviations); (ii) ambiguity explains 38.5% of GPR's effect; (iii) models ignoring ambiguity misattribute effects to volatility (47% inflation)

### 2. Introduction Section (Lines 39-43)
**Additions:**
- Added comparison of KL-divergence measure vs traditional proxies (variance of variance, forecast dispersion)
- Added three advantages of KL-based approach: (i) direct quantification of model uncertainty; (ii) daily construction for all liquid stocks; (iii) minimum KL divergence identifies relevant historical analogues

**Additions after Line 45:**
- Added identification assumptions for causal interpretation
- Added temporal ordering details (opening auction data, GPR shock timing)
- Added Granger causality test results

### 3. Literature Review Section (Lines 52-67)
**New Subsection Added (Lines 64-67):**
- "Alternative Ambiguity Measures and Our Contribution" subsection
- Added references to emerging market ambiguity literature: Bekaert2014, Bali2020, Chen2021, Luo2022

### 4. Hypotheses Development Section (Line 82)
**Expansion of H4:**
- Added expectations for heterogeneity across firm characteristics
- Added predictions for high-tech/export-oriented firms, small/less liquid stocks, analyst coverage effects

### 5. Data and Variables Section (Lines 90-95, 102-111)
**Precise Date Specifications:**
- Updated sample period: January 3, 2018 to December 29, 2023
- Added Pre-COVID and COVID period dates

**GPR Data Details:**
- Added source: 12 major international newspapers
- Added publication timing: 00:00 UTC, Beijing time conversion
- Added 3:00 PM cutoff for same-day effects
- Added acknowledgment of timing limitations

**Volatility Construction Details:**
- Added 5-minute return calculation between 9:30 AM-3:00 PM
- Added lunch break exclusion
- Added microstructure bias correction reference
- Added outlier removal criteria

**Data Quality Section Added:**
- Added exclusion criteria for financial firms, ST stocks
- Added minimum trading days requirement (200 days)
- Added winsorization details
- Added survivorship bias statistics

### 6. Ambiguity Measures Section (Line 106)
**Benchmark Selection Details:**
- Added rolling window specification (252 trading days)
- Added quarterly anchoring
- Added extreme volatility exclusion
- Added stationarity discussion

### 7. SOE Classification Section (Lines 116-117)
**SOE Definition and Mechanism:**
- Added CSRC definition details
- Added ownership thresholds (>50% direct or >30% through pyramids)
- Added three transmission channels for ambiguity cushion
- Added COVID-19 support evidence

### 8. Table Updates
**Table 1 - Baseline Time-Series Regression:**
- Added detailed variable definitions in notes
- Added units and frequency specifications

**Table 2 - GPR Shocks and Financial Ambiguity:**
- Added exact date ranges for Pre-COVID and COVID periods

**Table 3 - Fama-MacBeth Cross-Sectional Results:**
- Updated notation: $\beta_{i,\text{AMB}}$ and $\beta_{i,\text{VOL}}$
- Added ambiguity beta statistics (mean, median, std, etc.)

**New Table Added - Heterogeneity Analysis:**
- Added Table X showing ambiguity effects across firm characteristics

**Table 5 - SOE Moderation:**
- Added SOE sample statistics (34.2% of sample)
- Added market cap, industry, leverage, ownership details

**Table 6 - Robustness Checks:**
- Added IV statistics (F-statistic: 47.3, J-statistic: 2.84)

### 9. Methodology Section (Lines 279-290)
**Mediation Analysis:**
- Added causal narrative mechanism
- Added temporal sequence explanation
- Added investor behavior description

### 10. New Section Added - Data Timing and Alignment
**Three Subsections Added:**
- Timestamp Conventions
- Aggregation Rules
- Lead-Lag Justification

### 11. Conclusion Section (Lines 426-432)
**Practical Implications:**
- Added portfolio manager implications (67 bps underperformance)
- Added policy intervention limitations
- Added economic significance (11.3 bps daily, 28.3% annualized)
- Added comparison with volatility effects

**Limitations Added:**
- After-hours trading limitations
- English-language source bias
- Calendar-time alignment issues

**Stationarity Discussion:**
- Added ambiguity persistence (15-day half-life)
- Added alternative benchmark comparisons
- Added expanding windows and regime-switching alternatives

### 12. Appendix A Section (Lines 442-458)
**Algorithmic Summary Added:**
- Added Algorithm 1: Daily Ambiguity Calculation
- Added inputs, outputs, procedure steps, complexity

**Theoretical Connection Added:**
- Added smooth ambiguity formulation
- Added intuition for KL minimization
- Added case-based reasoning analogy
- Added Equation linking to multiple priors foundation
- Added reference to Equation (1)

### 13. Citation Updates
**New References Added:**
- Anderson2018
- Bekaert2014
- Bali2020
- Chen2021
- Davis2023
- Hansen2012
- Liu2024
- Luo2022
- Wang2023
- Zhang2024

### 14. Bibliography Fix
- Corrected "\bibliliography" to "\bibliography" at line 433

## Summary of Changes by Category

### Major Content Additions:
1. **Novelty Clarification** - Detailed comparison with existing ambiguity proxies
2. **Temporal Ordering** - Comprehensive timing documentation and causal identification
3. **Control Variable Rationale** - Theoretical justification and limitation discussion
4. **Heterogeneity Analysis** - New table and expanded hypotheses
5. **Timing Documentation** - New dedicated section with timestamp conventions

### Methodological Enhancements:
1. **Algorithm Specification** - Step-by-step ambiguity calculation procedure
2. **Benchmark Selection** - Detailed window specifications and alternatives
3. **IV Statistics** - First-stage and overidentification test results
4. **Granger Causality** - Asymmetric relationship test results

### Empirical Additions:
1. **Precise Dates** - All sample periods now have exact start/end dates
2. **Data Quality** - Comprehensive data cleaning and survivorship discussion
3. **SOE Details** - Classification mechanism and descriptive statistics
4. **Heterogeneity** - Cross-sectional analysis across firm characteristics

### Theoretical Connections:
1. **Smooth Ambiguity** - Formal connection to Klibanoff framework
2. **Multiple Priors** - Mathematical formulation linking to Gilboa-Schmeidler
3. **Case-Based Reasoning** - Intuitive explanation for KL minimization
4. **Information Asymmetry** - Discussion of emerging market features