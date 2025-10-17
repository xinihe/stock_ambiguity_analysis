
## Estimation Procedure for the Adaptive Ambiguity Parameter $\rho_t$

The model for the adaptive ambiguity parameter, $\rho_t$, is characterized by multiple latent variables: the time-varying coefficients ($\beta_t$), the unobserved market regimes ($s_t$), and the parameter $\rho_t$ itself. The complex, path-dependent nature of these variables renders the joint posterior distribution analytically intractable. Therefore, we employ Bayesian inference through a **Gibbs sampling algorithm**, a Markov Chain Monte Carlo (MCMC) method designed for such problems. The Gibbs sampler constructs the joint posterior distribution by iteratively sampling from the full conditional posterior distribution (FCPD) of each block of parameters.

#### 1. Data Preparation

Before estimation, the raw data must be processed:

1. **Construct Behavioral Indicators**: From the historical price series of an aggregate market index (e.g., the CSI 300), we compute the **Price Peak Proximity (PPP)** and **Price Trough Proximity (PTP)** indicators. This requires calculating an expanding window for the all-time high and all-time low prices up to time $t$.
2. **Lag Explanatory Variables**: All independent variables—$\text{iVIX}$, $\text{EPU}$, $\text{Turnover}$, $\text{PPP}$, and $\text{PTP}$—are lagged by one period to ensure they are in the information set at time $t-1$ when modeling $\rho_t$.
3. **Standardization**: To ensure numerical stability and improve the efficiency of the MCMC sampler, all time-series variables are standardized to have a mean of zero and a standard deviation of one.

#### 2. State-Space Representation of the MS-TVP Model

To facilitate estimation, we first cast the model into a state-space form.

**Measurement Equation:**
The measurement equation links the observed data (or in this case, the first-level latent variable $\rho_t$) to the state variables.

$$
\rho_t = X_t' \beta_{t, s_t} + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2_{\varepsilon, s_t})
$$

where:

* $X_t = [1, \text{iVIX}_{t-1}, \text{EPU}_{t-1}, \text{Turnover}_{t-1}, \text{PPP}_{t-1}, \text{PTP}_{t-1}]'$ is the $6 \times 1$ vector of regressors.
* $\beta_{t, s_t} = [\beta_{0,t,s_t}, \dots, \beta_{5,t,s_t}]'$ is the $6 \times 1$ vector of time-varying, regime-dependent coefficients.

**State Equation:**
The state equation governs the evolution of the time-varying coefficients. We assume they follow a random walk, which is a common and flexible specification.

$$
\beta_{t, s_t} = \beta_{t-1, s_t} + \eta_t, \quad \eta_t \sim N(0, Q_{s_t})
$$

where $Q_{s_t}$ is the regime-dependent variance-covariance matrix of the innovations to the state variables.

**Regime Switching:**
The latent state variable $s_t \in \{1, 2\}$ evolves according to the transition probability matrix $P$. The key feature of the model is that the parameters $\{\beta_{t, s_t}, \sigma^2_{\varepsilon, s_t}, Q_{s_t}\}$ are all dependent on the state $s_t$.

#### 3. The Gibbs Sampling Algorithm

The algorithm proceeds by initializing all parameters and then iteratively cycling through the following steps for a large number of iterations. After a sufficient "burn-in" period, the draws converge to their stationary posterior distributions.

Let $\Psi$ denote the full set of parameters to be estimated. One iteration of the Gibbs sampler involves the sequential drawing of each parameter block from its FCPD, conditional on the most recent draws of all other parameters and the data.

---

**Step 0: Initialization**
Initialize the parameters $\beta_{t,s_t}^{(0)}$, $Q_{s_t}^{(0)}$, $\sigma^{2(0)}_{\varepsilon,s_t}$, $P^{(0)}$, and the state vector $S_T^{(0)} = \{s_1^{(0)}, \dots, s_T^{(0)}\}$.

**Step 1: Sample the Time-Varying Coefficients $\beta_{T, s_t}$**
Conditional on the history of states $S_T = \{s_1, \dots, s_T\}$ and the other parameters, the model for $\rho_t$ simplifies to a standard linear Gaussian state-space model for each regime. We can therefore use the forward-filtering, backward-sampling algorithm of **Carter and Kohn (1994)**. This is performed separately for each regime's parameter set.

1. **For each regime $j \in \{1, 2\}$:**
2. Run the **Kalman filter** forward through time for $t=1, \dots, T$, using only the observations where $s_t=j$, to obtain the filtered estimates of the states $\beta_{t|t, s_t=j}$ and their covariances $P_{t|t, s_t=j}$.
3. Run the **backward-smoother** (or simulation smoother) for $t=T, \dots, 1$ to draw the full path of coefficients $\{\beta_{1,j}, \dots, \beta_{T,j}\}$ from their joint posterior distribution.

**Step 2: Sample the Latent States $S_T$**
Conditional on the path of time-varying coefficients $\beta_T$ and the other parameters, we can determine the likelihood of the data for each regime at each time point. We sample the entire vector of states $S_T$ at once using a **multi-move Gibbs sampling** approach based on the Hamilton filter.

1. **Filtering**: For $t=1, \dots, T$, run the **Hamilton (1989) filter** to compute the filtered probabilities $P(s_t=j | Y_t; \Psi)$, where $Y_t$ is the history of data up to time $t$. This step involves a prediction and an updating step based on the likelihood of the data in each regime.
2. **Backward Sampling**:
   * Draw the final state $s_T$ from its filtered probability mass function $P(s_T | Y_T; \Psi)$.
   * For $t=T-1, \dots, 1$, draw the state $s_t$ from its conditional probability $P(s_t | s_{t+1}, Y_t; \Psi)$, which is calculated using the filtered probabilities and the transition matrix $P$.

**Step 3: Sample the Hyperparameters ($Q_{s_t}, \sigma^2_{\varepsilon, s_t}$)**
The variance parameters are sampled from their respective FCPDs, which are typically from the Inverse-Gamma or Inverse-Wishart family given conjugate prior specifications.

1. **Sample $\sigma^2_{\varepsilon, s_t}$**: For each regime $j \in \{1, 2\}$, the conditional posterior for $\sigma^2_{\varepsilon, j}$ is an Inverse-Gamma distribution. The parameters of this distribution are updated using the sum of squared residuals $\sum_{t:s_t=j} ( \rho_t - X_t' \beta_{t,j} )^2$.
2. **Sample $Q_{s_t}$**: Similarly, for each regime $j$, the conditional posterior for the covariance matrix $Q_j$ is an Inverse-Wishart distribution. Its parameters are updated using the sum of squared innovations from the state equation, $\sum_{t:s_t=j} (\beta_{t,j} - \beta_{t-1,j})'(\beta_{t,j} - \beta_{t-1,j})$.

**Step 4: Sample the Transition Probabilities $P$**
The transition probabilities have a Dirichlet prior, which is conjugate to the multinomial likelihood of the observed transitions in the sampled state path $S_T$.

1. Count the number of transitions from regime $i$ to regime $j$ ($n_{ij}$) in the currently sampled path $S_T$.
2. For each row $i$ of the transition matrix, the posterior distribution of the vector $p_i = (p_{i1}, p_{i2})$ is a Dirichlet distribution. The parameters are updated by adding the transition counts $n_{ij}$ to the parameters of the Dirichlet prior.

---

This cycle is repeated for a large number of iterations (e.g., 20,000). The initial portion (e.g., the first 10,000) is discarded as the burn-in period. The remaining draws form an empirical approximation of the joint posterior distribution. The final estimate for the time path of ambiguity aversion, $\rho_t$, is typically taken as the mean or median of the posterior draws for $\rho_t$ at each time point, along with credible intervals (e.g., the 5th and 95th percentiles) to quantify its uncertainty.
