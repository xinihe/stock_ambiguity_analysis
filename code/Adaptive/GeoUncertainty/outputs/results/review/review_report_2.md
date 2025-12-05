## **Paper Title:** Global Geopolitical Risk, Ambiguity, and Emerging Market Returns: Evidence from China

### Summary

- The paper proposes a daily ambiguity measure built from a cross-entropy/KL framework to distinguish ambiguity from volatility in asset pricing.
- Using Chinese market data (CSI 300 and constituents), the study tests whether geopolitical risk increases ambiguity and whether ambiguity is negatively priced in returns.
- The empirical strategy includes time-series regressions, Fama–MacBeth cross-sectional models, mediation analysis to quantify the GPR→ambiguity→returns pathway, and SOE moderation.
- Results indicate significant effects of GPR on ambiguity, a negative ambiguity premium, and improved explanatory power when ambiguity is included alongside volatility.

### Major Comments

1. Novelty relative to existing ambiguity proxies and smooth-ambiguity implementations is not fully articulated; clarify the methodological contribution by explicitly positioning the KL-based measure against variance-of-variance and forecast-dispersion proxies, and strengthen theoretical links in the text.
2. Potential reverse causality from returns/volatility to ambiguity could bias interpretation; explain temporal ordering and design choices, and add a brief discussion of why reverse causality is unlikely under the adopted setup.
3. The mediator–outcome treatment may be sensitive to omitted macro or micro controls; state the rationale for the selected control set, acknowledge limitations, and indicate how residual confounding is mitigated.
4. Cross-sectional results may mask heterogeneity across industry or size/liquidity segments; provide interpretive context on where ambiguity effects are expected to be stronger or weaker, and note any observable patterns from descriptive statistics.
5. The link between daily GPR and local-market ambiguity may face timing and coverage considerations; document source coverage, local-time conversions, and the handling of news windows, and include a short limitations paragraph.

### Minor Comments

- Add exact construction details for realized volatility (sampling interval, trading hours, microstructure filters).
- Provide a short data-quality section on survivorship, ST stocks, and winsorization thresholds.
- Standardize confidence-level markers across all tables.
- Include a brief intuition for the KL-minimization step.
- Verify consistency of local-time conversions between U.S.-based GPR and China trading calendars.
- Report average cross-section size and dispersion for ambiguity betas over time.
- Add references situating ambiguity in emerging-market contexts.

### Recommendation (Confidential to the Editor)

- Major Revisions.
- The empirical patterns are compelling and the measure is useful; targeted clarifications and robustness checks will improve clarity and credibility without requiring substantial redesign.
