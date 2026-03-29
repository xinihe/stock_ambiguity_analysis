def enrich_manuscript():
    with open('code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb04_article.tex', 'r') as f:
        content = f.read()

    # 1. Abstract
    old_abstract = r"\textcolor{red}{When ambiguity is included alongside traditional risk, models explain more variation in returns, illustrating that ambiguity captures a distinct dimension of Knightian uncertainty without invalidating the role of traditional risk.}"
    new_abstract = r"\textcolor{red}{When ambiguity is included alongside traditional risk, models explain more variation in returns, illustrating that ambiguity captures a distinct dimension of Knightian uncertainty without invalidating the role of traditional risk. Rather than competing for explanatory supremacy, volatility and ambiguity function as dual, coexisting channels that jointly capture investor distress during geopolitical shocks.}"
    content = content.replace(old_abstract, new_abstract)

    # 1.1 Intro
    old_intro1 = r"\textcolor{red}{(iii) models ignoring ambiguity misattribute a portion of ambiguity's impact to volatility, illustrating the complementary nature of these two uncertainty channels.}"
    new_intro1 = r"\textcolor{red}{(iii) models ignoring ambiguity misattribute a portion of ambiguity's impact to volatility, illustrating the complementary nature of these two uncertainty channels. By treating ambiguity and volatility as dual, coexisting factors, we align our narrative with foundational theories that posit ambiguity aversion as a distinct preference from risk aversion.}"
    content = content.replace(old_intro1, new_intro1)

    # 2. Lit Review - proxies
    old_proxies = r"\textcolor{red}{Unlike forecast dispersion, which relies on analyst coverage, or variance-of-variance, which captures second-moment uncertainty, our cross-entropy measure directly quantifies the divergence from historical reference models, thereby capturing the essence of Knightian uncertainty---where the probability distributions themselves are unknown.}"
    new_proxies = r"\textcolor{red}{Unlike forecast dispersion, which heavily relies on subjective analyst coverage and biases samples toward large firms, or variance-of-variance, which fundamentally captures second-moment uncertainty and is sensitive to high-frequency measurement errors, our cross-entropy measure directly quantifies the divergence from historical reference models. By doing so, it captures the essence of Knightian uncertainty---where the probability distributions themselves are unknown---and can be constructed daily for the entire universe of liquid stocks.}"
    content = content.replace(old_proxies, new_proxies)

    # 3. Intro - magnitudes
    old_intro2 = r"\textcolor{red}{Our findings have practical implications. The daily associations suggest that portfolios ignoring ambiguity may face uncompensated risks during high-GPR periods, highlighting the importance of ambiguity-aware risk management. Policy interventions focusing solely on volatility stabilization may miss a significant portion of the ambiguity transmission channel.}"
    new_intro2 = r"\textcolor{red}{Our findings have practical implications. The daily associations suggest that portfolios optimized solely for minimum variance may leave investors exposed to uncompensated ambiguity risks during periods of heightened geopolitical tension, highlighting the importance of ambiguity-aware risk management. However, given the episodic and mean-reverting nature of geopolitical shocks, these associations should be interpreted cautiously as daily impacts rather than deterministic long-term outcomes.}"
    content = content.replace(old_intro2, new_intro2)

    # 3.1 Results - magnitudes
    old_mag = r"\textcolor{red}{In terms of economic significance, a one-standard-deviation increase in ambiguity is associated with a daily return reduction of 11.3 bps.}"
    new_mag = r"\textcolor{red}{In terms of economic significance, a one-standard-deviation increase in ambiguity is associated with a daily return reduction of 11.3 bps. We focus on this daily impact, which is more appropriate than annualized figures given the high-frequency nature of our data and the transient nature of the shocks we study.}"
    content = content.replace(old_mag, new_mag)

    # 4. Mediation limitations
    old_med = r"\textcolor{red}{However, while the Baron-Kenny framework provides a useful statistical decomposition, we caution against strict causal interpretations. In high-frequency financial data, unobserved contemporaneous shocks can affect both the mediator and the outcome simultaneously, complicating the identification of isolated transmission paths.}"
    new_med = r"\textcolor{red}{However, while the Baron-Kenny framework provides a useful statistical decomposition, we caution against strict causal interpretations. In high-frequency financial data, unobserved contemporaneous shocks (such as sudden shifts in global liquidity or simultaneous macroeconomic data releases) can affect both the mediator and the outcome simultaneously. This potential for unobserved confounding complicates the identification of isolated transmission paths, meaning our mediation results should be viewed as an associational decomposition rather than a definitive causal proof.}"
    content = content.replace(old_med, new_med)

    # 5. IV exclusion restriction
    old_iv = r"Econometric tests confirm the validity of our instrument:"
    new_iv = r"\textcolor{red}{While the 'Non-Asian GPR' instrument is temporally predetermined relative to the Chinese trading session, we acknowledge that global geopolitical shocks might influence Chinese markets through alternative contemporaneous channels. For instance, a major geopolitical escalation overnight might immediately trigger a spike in global commodity prices or induce massive safe-haven currency flows. To the extent these parallel channels are correlated with our instrument, the exclusion restriction may be partially violated, suggesting our IV estimates should be interpreted with appropriate caution as they might capture broader global shock effects alongside the local ambiguity channel.} Econometric tests confirm the validity of our instrument:"
    content = content.replace(old_iv, new_iv)

    # 6. Timing assumptions
    old_time = r"\textcolor{red}{We use a contemporaneous alignment $(t, t)$ for our main analysis. This is justified because the GPR news (arriving pre-open, between 5:00 AM and 9:30 AM Beijing time) has ample time to be incorporated into investor beliefs and trading behavior throughout the trading day. We also introduce a lagged specification $(t-1, t)$ in our robustness checks to verify that the effects are not purely driven by our contemporaneous timing assumptions.}"
    new_time = r"\textcolor{red}{We use a contemporaneous alignment $(t, t)$ for our main analysis. This is justified because the GPR news (compiled at 00:00 UTC, arriving between 5:00 AM and 8:00 AM Beijing time) has ample time to be incorporated into investor beliefs and trading behavior throughout the trading day before the 15:00 close. However, recognizing that information frictions or behavioral delays might cause market reactions to spill over, we also introduce a lagged specification $(t-1, t)$ in our robustness checks. This verifies that the negative pricing of ambiguity is robust and not merely an artifact of our specific contemporaneous timing assumptions.}"
    content = content.replace(old_time, new_time)

    # 7. Broader Literature
    old_lit2 = r"\textcolor{red}{Our work extends the emerging literature on uncertainty transmission in several critical ways, bridging the gap between macroeconomic uncertainty and ambiguity pricing. First, while studies like \citet{Baker2016} and \citet{Caldara2022} document that uncertainty matters for markets, they often treat uncertainty as monolithic. By decomposing uncertainty into complementary risk (volatility) and ambiguity channels, we build on the insights of \citet{Brenner2018} and reveal that ambiguity is a distinct factor in GPR's pricing effect.}"
    new_lit2 = r"\textcolor{red}{Our work extends the emerging literature on uncertainty transmission in several critical ways, bridging the gap between macroeconomic uncertainty and ambiguity pricing. First, while studies like \citet{Baker2016} and \citet{Caldara2022} document that broad uncertainty shocks depress asset prices, they often treat uncertainty as a monolithic construct. Conversely, the ambiguity literature (e.g., \citet{Brenner2018}) provides theoretical frameworks but often lacks applications to specific, exogenous macroeconomic shocks. By decomposing uncertainty into complementary risk (volatility) and ambiguity channels using our novel measure, we build on the insights of \citet{Brenner2018} and reveal that ambiguity is a distinct, critical factor in GPR's pricing effect.}"
    content = content.replace(old_lit2, new_lit2)

    # 8. Regression models
    old_reg = r"\textcolor{red}{In our time-series specifications, we incorporate a streamlined set of control variables to isolate the impact of geopolitical ambiguity. We control for the \textbf{Market Return ($MKT$)} to capture systematic market movements, the \textbf{VIX Index} for global risk sentiment, the \textbf{Term Spread ($TERM$)} for domestic monetary policy, and the \textbf{Risk-Free Rate ($RF$)}.}"
    new_reg = r"\textcolor{red}{In our time-series specifications, we incorporate a streamlined set of control variables to cleanly isolate the impact of geopolitical ambiguity. We control for the \textbf{Market Return ($MKT$)} to capture systematic market movements, the \textbf{VIX Index} for global risk sentiment, the \textbf{Term Spread ($TERM$)} for domestic monetary policy conditions, and the \textbf{Risk-Free Rate ($RF$)}. This concise set of controls ensures that the core focus remains on the interplay between geopolitical risk and ambiguity.}"
    content = content.replace(old_reg, new_reg)

    with open('code/Adaptive/GeoUncertainty/outputs/results/geopoliticalAmb04_article.tex', 'w') as f:
        f.write(content)

def enrich_response():
    with open('code/Adaptive/GeoUncertainty/outputs/results/review/respond_3.tex', 'r') as f:
        content = f.read()

    # 1. Abstract
    content = content.replace(
        r'\textit{\textcolor{red}{"When ambiguity is included alongside traditional risk, models explain more variation in returns, illustrating that ambiguity captures a distinct dimension of Knightian uncertainty without invalidating the role of traditional risk."}}',
        r'\textit{\textcolor{red}{"When ambiguity is included alongside traditional risk, models explain more variation in returns, illustrating that ambiguity captures a distinct dimension of Knightian uncertainty without invalidating the role of traditional risk. Rather than competing for explanatory supremacy, volatility and ambiguity function as dual, coexisting channels that jointly capture investor distress during geopolitical shocks."}}'
    )

    # 1.1 Intro
    content = content.replace(
        r'\textit{\textcolor{red}{"(iii) models ignoring ambiguity misattribute a portion of ambiguity\'s impact to volatility, illustrating the complementary nature of these two uncertainty channels."}}',
        r'\textit{\textcolor{red}{"(iii) models ignoring ambiguity misattribute a portion of ambiguity\'s impact to volatility, illustrating the complementary nature of these two uncertainty channels. By treating ambiguity and volatility as dual, coexisting factors, we align our narrative with foundational theories that posit ambiguity aversion as a distinct preference from risk aversion."}}'
    )

    # 2. Lit Review - proxies
    content = content.replace(
        r'\textit{\textcolor{red}{"Unlike forecast dispersion, which relies on analyst coverage, or variance-of-variance, which captures second-moment uncertainty, our cross-entropy measure directly quantifies the divergence from historical reference models, thereby capturing the essence of Knightian uncertainty---where the probability distributions themselves are unknown."}}',
        r'\textit{\textcolor{red}{"Unlike forecast dispersion, which heavily relies on subjective analyst coverage and biases samples toward large firms, or variance-of-variance, which fundamentally captures second-moment uncertainty and is sensitive to high-frequency measurement errors, our cross-entropy measure directly quantifies the divergence from historical reference models. By doing so, it captures the essence of Knightian uncertainty---where the probability distributions themselves are unknown---and can be constructed daily for the entire universe of liquid stocks."}}'
    )

    # 3. Intro - magnitudes
    content = content.replace(
        r'\textit{\textcolor{red}{"Our findings have practical implications. The daily associations suggest that portfolios ignoring ambiguity may face uncompensated risks during high-GPR periods, highlighting the importance of ambiguity-aware risk management. Policy interventions focusing solely on volatility stabilization may miss a significant portion of the ambiguity transmission channel."}}',
        r'\textit{\textcolor{red}{"Our findings have practical implications. The daily associations suggest that portfolios optimized solely for minimum variance may leave investors exposed to uncompensated ambiguity risks during periods of heightened geopolitical tension, highlighting the importance of ambiguity-aware risk management. However, given the episodic and mean-reverting nature of geopolitical shocks, these associations should be interpreted cautiously as daily impacts rather than deterministic long-term outcomes."}}'
    )

    # 3.1 Results - magnitudes
    content = content.replace(
        r'\textit{\textcolor{red}{"In terms of economic significance, a one-standard-deviation increase in ambiguity is associated with a daily return reduction of 11.3 bps."}}',
        r'\textit{\textcolor{red}{"In terms of economic significance, a one-standard-deviation increase in ambiguity is associated with a daily return reduction of 11.3 bps. We focus on this daily impact, which is more appropriate than annualized figures given the high-frequency nature of our data and the transient nature of the shocks we study."}}'
    )

    # 4. Mediation limitations
    content = content.replace(
        r'\textit{\textcolor{red}{"However, while the Baron-Kenny framework provides a useful statistical decomposition, we caution against strict causal interpretations. In high-frequency financial data, unobserved contemporaneous shocks can affect both the mediator and the outcome simultaneously, complicating the identification of isolated transmission paths."}}',
        r'\textit{\textcolor{red}{"However, while the Baron-Kenny framework provides a useful statistical decomposition, we caution against strict causal interpretations. In high-frequency financial data, unobserved contemporaneous shocks (such as sudden shifts in global liquidity or simultaneous macroeconomic data releases) can affect both the mediator and the outcome simultaneously. This potential for unobserved confounding complicates the identification of isolated transmission paths, meaning our mediation results should be viewed as an associational decomposition rather than a definitive causal proof."}}'
    )

    # 5. IV exclusion restriction
    content = content.replace(
        r'\textit{\textcolor{red}{"While the \'Non-Asian GPR\' instrument is temporally predetermined relative to the Chinese trading session, we acknowledge that global geopolitical shocks might influence Chinese markets through alternative contemporaneous channels, such as overnight shifts in global commodity prices or safe-haven currency flows. To the extent these channels are correlated with our instrument, the exclusion restriction may be partially violated, suggesting our IV estimates should be interpreted with appropriate caution."}}',
        r'\textit{\textcolor{red}{"While the \'Non-Asian GPR\' instrument is temporally predetermined relative to the Chinese trading session, we acknowledge that global geopolitical shocks might influence Chinese markets through alternative contemporaneous channels. For instance, a major geopolitical escalation overnight might immediately trigger a spike in global commodity prices or induce massive safe-haven currency flows. To the extent these parallel channels are correlated with our instrument, the exclusion restriction may be partially violated, suggesting our IV estimates should be interpreted with appropriate caution as they might capture broader global shock effects alongside the local ambiguity channel."}}'
    )

    # 6. Timing assumptions
    content = content.replace(
        r'\textit{\textcolor{red}{"We use a contemporaneous alignment $(t, t)$ for our main analysis. This is justified because the GPR news (arriving pre-open, between 5:00 AM and 9:30 AM Beijing time) has ample time to be incorporated into investor beliefs and trading behavior throughout the trading day. We also introduce a lagged specification $(t-1, t)$ in our robustness checks to verify that the effects are not purely driven by our contemporaneous timing assumptions."}}',
        r'\textit{\textcolor{red}{"We use a contemporaneous alignment $(t, t)$ for our main analysis. This is justified because the GPR news (compiled at 00:00 UTC, arriving between 5:00 AM and 8:00 AM Beijing time) has ample time to be incorporated into investor beliefs and trading behavior throughout the trading day before the 15:00 close. However, recognizing that information frictions or behavioral delays might cause market reactions to spill over, we also introduce a lagged specification $(t-1, t)$ in our robustness checks. This verifies that the negative pricing of ambiguity is robust and not merely an artifact of our specific contemporaneous timing assumptions."}}'
    )

    # 7. Broader Literature
    content = content.replace(
        r'\textit{\textcolor{red}{"Our work extends the emerging literature on uncertainty transmission in several critical ways, bridging the gap between macroeconomic uncertainty and ambiguity pricing. First, while studies like Baker et al. (2016) and Caldara and Iacoviello (2022) document that uncertainty matters for markets, they often treat uncertainty as monolithic. By decomposing uncertainty into complementary risk (volatility) and ambiguity channels, we build on the insights of Brenner and Izhakian (2018) and reveal that ambiguity is a distinct factor in GPR\'s pricing effect."}}',
        r'\textit{\textcolor{red}{"Our work extends the emerging literature on uncertainty transmission in several critical ways, bridging the gap between macroeconomic uncertainty and ambiguity pricing. First, while studies like Baker et al. (2016) and Caldara and Iacoviello (2022) document that broad uncertainty shocks depress asset prices, they often treat uncertainty as a monolithic construct. Conversely, the ambiguity literature (e.g., Brenner and Izhakian (2018)) provides theoretical frameworks but often lacks applications to specific, exogenous macroeconomic shocks. By decomposing uncertainty into complementary risk (volatility) and ambiguity channels using our novel measure, we build on the insights of Brenner and Izhakian (2018) and reveal that ambiguity is a distinct, critical factor in GPR\'s pricing effect."}}'
    )

    # 8. Regression models
    content = content.replace(
        r'\textit{\textcolor{red}{"In our time-series specifications, we incorporate a streamlined set of control variables to isolate the impact of geopolitical ambiguity. We control for the \textbf{Market Return ($MKT$)} to capture systematic market movements, the \textbf{VIX Index} for global risk sentiment, the \textbf{Term Spread ($TERM$)} for domestic monetary policy, and the \textbf{Risk-Free Rate ($RF$)}."}}',
        r'\textit{\textcolor{red}{"In our time-series specifications, we incorporate a streamlined set of control variables to cleanly isolate the impact of geopolitical ambiguity. We control for the \textbf{Market Return ($MKT$)} to capture systematic market movements, the \textbf{VIX Index} for global risk sentiment, the \textbf{Term Spread ($TERM$)} for domestic monetary policy conditions, and the \textbf{Risk-Free Rate ($RF$)}. This concise set of controls ensures that the core focus remains on the interplay between geopolitical risk and ambiguity."}}'
    )

    with open('code/Adaptive/GeoUncertainty/outputs/results/review/respond_3.tex', 'w') as f:
        f.write(content)

enrich_manuscript()
enrich_response()
