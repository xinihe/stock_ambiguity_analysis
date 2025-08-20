## A Comprehensive Methodology to Estimate the Adaptive Ambiguity Aversion Parameter, $\rho$

To capture the complex dynamics of ambiguity aversion, we propose a **Markov-Switching Time-Varying Parameter (MS-TVP) Model**. This framework is uniquely suited to model a process that evolves gradually over time but is also subject to abrupt structural breaks corresponding to shifts in the underlying market regime.

#### 1. Market Regimes

We model the market as operating in one of $k=2$ latent states, $s_t$, at any given time $t$:

* **$s_t = 1$**: A **Calm Regime**, characterized by lower volatility, stable policy, and generally positive investor sentiment.
* **$s_t = 2$**: A **Turbulent Regime**, characterized by high volatility, heightened economic uncertainty, and widespread investor fear.

The evolution of these states is governed by a first-order Markov process with a constant transition probability matrix, $P$:

$$
P = \begin{pmatrix} p_{11} & p_{12} \\ p_{21} & p_{22} \end{pmatrix} 
$$

$$


$$

where $p_{ij} = P(s_t = j | s_{t-1} = i)$ is the probability of transitioning from regime $i$ to regime $j$, and $\sum_{j=1}^{2} p_{ij} = 1$ for $i \in \{1,2\}$.

#### 2. The State-Dependent Model for $\rho_t$

The ambiguity aversion parameter, $\rho_t$, is modeled as a time-varying linear function of a set of explanatory variables. The coefficients of this linear relationship are dependent on the market regime, $s_t$, allowing the sensitivity of $\rho_t$ to each factor to change depending on the market state.

The formal equation is:

$$
\rho_t = \beta_{0, s_t} + \beta_{1, s_t} \cdot \text{iVIX}_{t-1} + \beta_{2, s_t} \cdot \text{EPU}_{t-1} + \beta_{3, s_t} \cdot \text{Turnover}_{t-1} + \beta_{4, s_t} \cdot \text{PPP}_{t-1} + \beta_{5, s_t} \cdot \text{PTP}_{t-1} + \varepsilon_t 
$$

$$


$$

The components of the model are defined as:

* **$\rho_t$**: The latent ambiguity aversion parameter at time $t$.
* **$s_t$**: The unobserved market regime at time $t$, which determines which set of coefficients ($\beta_{k,1}$ or $\beta_{k,2}$) applies.
* **$\beta_{k, s_t}$**: The regime-dependent coefficients. For instance, we hypothesize that the sensitivity of $\rho_t$ to volatility will be much higher in the turbulent regime (i.e., $\beta_{1,2} > \beta_{1,1}$).
* **$\text{iVIX}_{t-1}$**: The lagged China ETF Volatility Index, serving as a proxy for market-wide risk expectations.
* **$\text{EPU}_{t-1}$**: The lagged Economic Policy Uncertainty Index for China, capturing ambiguity stemming from the political and regulatory environment.
* **$\text{Turnover}_{t-1}$**: The lagged stock market turnover rate, used as a measure of retail investor sentiment and speculative activity in the Chinese market.
* **$\text{PPP}_{t-1}$**: The lagged Price Peak Proximity indicator for a specific asset or an aggregate index.
* **$\text{PTP}_{t-1}$**: The lagged Price Trough Proximity indicator.
* **$\varepsilon_t$**: A heteroskedastic error term, where $\varepsilon_t \sim N(0, \sigma^2_{\varepsilon, s_t})$.

#### 3. Estimation

The parameters of this complex, non-linear state-space model are best estimated using Bayesian inference. A **Gibbs sampling algorithm** is a standard and effective approach for this class of models. This iterative Monte Carlo method allows for the joint estimation of the time-varying coefficients ($\beta_{k, s_t}$), the regime-dependent variances ($\sigma^2_{\varepsilon, s_t}$), the transition probabilities ($p_{ij}$), and the unobserved sequence of market regimes ($s_t$) by drawing from their respective conditional posterior distributions.

### Rationale for the Proposed Estimation Methodology

The assumption of a static, time-invariant ambiguity aversion parameter, common in traditional asset pricing and portfolio choice models, is a significant limitation. It fails to capture the empirically observable fact that investor attitudes toward uncertainty are not fixed but are instead highly dependent on the prevailing economic and financial environment. To address this deficiency, we propose a dynamic estimation framework that characterizes the ambiguity aversion parameter, $\rho$, as a time-varying process driven by observable market conditions.

Our methodology is specifically designed to capture the complex nature of investor sentiment, which exhibits both gradual evolution and abrupt structural shifts. A simple time-varying parameter (TVP) model, while allowing for smooth, continuous adaptation, is ill-equipped to handle the sudden, discrete changes in behavior associated with financial crises or major policy shocks. Conversely, a standard Markov-switching model, which allows for shifts between a finite set of regimes, captures structural breaks effectively but fails to account for the gradual evolution of sentiment *within* a given market state. Neither approach, in isolation, provides a complete picture.

To overcome these individual limitations, we propose a **Markov-Switching Time-Varying Parameter (MS-TVP) model**. This hybrid framework provides a much richer and more realistic representation of the dynamics of ambiguity aversion. The Markov-switching component explicitly captures the large, discrete shifts in the underlying data-generating process corresponding to distinct market regimes—such as a "calm" state and a "turbulent" state. Within each of these regimes, the model allows the coefficients that link ambiguity aversion to our explanatory variables to vary over time. This dual structure allows us to model both the sharp re-pricing of ambiguity during a regime shift and the more nuanced, continuous adjustments investors make in response to new information within an established market context.

Furthermore, we enrich the model by incorporating not only macro-level sentiment indicators (iVIX, EPU) but also novel, asset-specific behavioral proxies: the **Price Peak Proximity (PPP)** and **Price Trough Proximity (PTP)** indicators. These are defined as:

* **Price Peak Proximity**: $\text{PPP}_t = \frac{\text{Current Price}_t}{\text{All-Time High Price}_t}$
* **Price Trough Proximity**: $\text{PTP}_t = \frac{\text{All-Time Low Price}_t}{\text{Current Price}_t}$

The inclusion of these variables is grounded in behavioral finance. They serve to capture how the perception of ambiguity is shaped by an asset's position within its historical price range. When an asset's price is near its historical peak ($\text{PPP}_t \to 1$), a sense of imminent mean reversion can reduce the perceived ambiguity of future positive returns, potentially dampening the influence of ambiguity aversion. Conversely, when an asset is near its historical trough ($\text{PTP}_t \to 1$), the fundamental uncertainty about its future viability is arguably at its maximum, a condition that likely amplifies an investor's sensitivity to ambiguity. By integrating these micro-level behavioral indicators, our model gains the ability to connect broad market sentiment to the tangible decision-making context of individual assets. This comprehensive approach yields a more accurate and granular measure of $\rho_t$, providing a superior foundation for dynamic risk management and asset allocation.
