
## **Paper Title:** Global Geopolitical Risk, Ambiguity, and Emerging Market Returns: Evidence from China


### Summary 

- The paper proposes a daily ambiguity measure built from a cross-entropy/KL framework to distinguish ambiguity from volatility in asset pricing.
- Using Chinese market data (CSI 300 and constituents), the study tests whether geopolitical risk increases ambiguity and whether ambiguity is negatively priced in returns.
- The empirical strategy includes time-series regressions, Fama–MacBeth cross-sectional models, mediation analysis to quantify the GPR→ambiguity→returns pathway, and SOE moderation.
- Results indicate significant effects of GPR on ambiguity, a negative ambiguity premium, and improved explanatory power when ambiguity is included alongside volatility.

### Major Comments

1. Problem: Novelty relative to existing ambiguity proxies and smooth-ambiguity implementations is not fully articulated. Suggestion: Sharpen the methodological contribution by contrasting with variance-of-variance, forecast dispersion, and model-uncertainty approaches; add theoretical links and empirical head-to-head comparisons.
2. Problem: Potential reverse causality from returns/volatility to ambiguity could bias estimates. Suggestion: Employ stronger designs (e.g., lag structures, external instruments tied to exogenous geopolitical events, placebo windows) and document robustness to alternative timing assumptions.
3. Problem: The mediator–outcome treatment may be sensitive to omitted macro or micro controls. Suggestion: Expand controls (liquidity, turnover, illiquidity, governance) and include fixed effects where relevant; report variance inflation and partial R² to show incremental explanatory power.
4. Problem: Cross-sectional Fama–MacBeth results may mask heterogeneity in industry or size/liquidity segments. Suggestion: Provide stratified analyses by sector, size/liquidity deciles, and test interaction terms to ensure stability of the ambiguity premium.
5. Problem: The link between daily GPR and local-market ambiguity may suffer from news timing and coverage biases. Suggestion: Validate using alternative GPR series, Chinese-language sources, and event-based measures; show results with rolling-news windows and sentiment controls.

### Minor Comments

- Add exact construction details for realized volatility (sampling interval, trading hours, microstructure filters).
- Provide a short data-quality section on survivorship, ST stocks, and winsorization thresholds.
- Standardize confidence-level markers and HAC lag choices across all tables.
- Include a brief intuition for the KL-minimization step with a schematic figure.
- Verify consistency of local-time conversions between U.S.-based GPR and China trading calendars.
- Report average cross-section size and dispersion for ambiguity betas over time.
- Add references situating ambiguity in emerging-market contexts.

### Recommendation (Confidential to the Editor)

- Major Revisions.
- The empirical patterns are compelling and the measure is useful; targeted clarifications and robustness checks will improve clarity and credibility without requiring substantial redesign.
