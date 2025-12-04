**BEGIN REVIEW**

**Paper Title:** Global Geopolitical Risk, Ambiguity, and Emerging Market Returns: Evidence from China
**Paper ID:** geopoliticalAmb03_article

### Part 1: Summary of the Paper

This paper examines how global geopolitical risk (GPR) influences Chinese equity returns through two distinct uncertainty channels: volatility (risk) and ambiguity (Knightian uncertainty). The authors construct daily ambiguity measures using a cross-entropy/KL-divergence framework and assess their pricing in both time-series and cross-sectional settings, including mediation analysis to quantify the GPR→ambiguity→returns pathway and moderation by SOE status. The main findings are that GPR significantly raises financial ambiguity, ambiguity is negatively priced, and including ambiguity alongside volatility improves explanatory power. Evidence is assembled using time-series regressions for the CSI 300, Fama–MacBeth cross-sectional tests for constituents, and robustness checks, with heterogeneity explored through SOE moderation.

### Part 2: Major Comments

1. The causal interpretation in the mediation analysis relies on strong assumptions and may be underidentified given potential confounders and timing frictions; tighten identification by aligning event-time windows, using plausibly exogenous instruments with validity checks, and reporting sensitivity to mediator endogeneity.
2. The KL-based ambiguity measure depends on the choice of benchmark distributions and implicit stationarity across regimes; detail how benchmarks are selected, justify stationarity assumptions, and add concise robustness using alternative benchmark sets, rolling windows, and allowance for heavy tails.
3. Frequency and timing alignment between daily GPR, ambiguity, realized volatility, and market/stock returns could introduce measurement mismatch; clarify timestamp conventions (local vs UTC), intraday aggregation rules, and lead–lag structures, and verify stability with close-to-close aligned variants.
4. The SOE moderation may reflect sample composition or correlated institutional features rather than a distinct cushion from ambiguity; consider alternative SOE definitions, matched-sample or simple balance checks, and add governance/liquidity controls to isolate the mechanism.
5. The contribution would benefit from clearer positioning of the ambiguity channel relative to volatility-focused frameworks; strengthen the narrative in the introduction and related work to articulate incremental insight and practical relevance without expanding the empirical scope.

### Part 3: Minor Comments

- Clarify the sample period and exact trading days for all subsamples (pre-COVID vs COVID) with start/end dates.
- Define all variables in table captions (e.g., “ΔAmbiguity”, “RV”) and specify units/frequencies.
- Increase figure/table font sizes and ensure captions state standard errors and lag choices (e.g., Newey–West 5-day) consistently.
- Ensure consistent notation for ambiguity and volatility betas across text, equations, and tables.
- Add a brief algorithmic summary of the ambiguity measure in Appendix A with inputs/outputs.
- Report instrument strength and overidentification tests where IV is used.
- Citations: include more up-to-date references and situate comparisons to contemporary uncertainty proxies (e.g., EPU and recent geopolitical uncertainty measures).

### Part 4: Recommendation (Confidential to the Editor)

- Major Revisions Required.
- The paper addresses an important question with promising methodology, but causal identification, measurement robustness, and generalizability require additional analyses; resolving these will materially strengthen the contribution.

**END REVIEW**