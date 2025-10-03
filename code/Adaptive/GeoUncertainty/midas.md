### Rationale for the Advanced Mixture Index

The core objective of this research is to robustly link global systemic uncertainty to the adaptive ambiguity aversion index (![]()) in the Chinese capital market. Achieving this requires creating a composite index that accurately reflects the full complexity of external shocks, addressing two significant methodological challenges: the **Data Frequency Mismatch** and the risk of  **Multicollinearity** .

#### 1. Addressing Data Frequency Mismatch and Informational Loss

The Geopolitical Risk (GPR) Index by Caldara and Iacoviello is available daily, capturing sudden, high-frequency shocks—like immediate market anticipation of a conflict or sanctions threat. In contrast, the Climate Risk Index, like most policy-related uncertainty measures, is typically available only monthly.

A simple arithmetic average of the monthly Climate Risk ($CR_t$) and the monthly average of GPR would flatten the GPR series, destroying the intraday or daily volatility spikes that are crucial drivers of sudden investor fear and capital flow shifts in emerging markets. The high-frequency GPR data holds valuable informational content about the immediacy of global threats. To preserve this information while aligning the indices to the necessary monthly frequency, an advanced weighting scheme is essential.

#### 2. Mitigating Multicollinearity in the State Transition Equation

The research model uses these global measures as coefficient drivers in the state transition equation for the unobservable ambiguity aversion index ($\theta_{i,t}}$). If the two indices, GPR and CR, are included separately, their documented tendency to co-move, especially during global crises, will lead to multicollinearity. This inflates the standard errors, making it statistically difficult to determine the unique and significant contribution of each risk factor to the evolution of . Creating a single, well-weighted Global Systemic Uncertainty Index  ensures that the model robustly estimates the overall impact of aggregated global non-economic uncertainty.

---

### Methodology for Constructing the Combined Global Uncertainty Index (![]())

The Global Systemic Uncertainty Index (![]()) is constructed at a monthly frequency using the **Mixed Data Sampling (MIDAS)** model, which optimally utilizes the high-frequency GPR data to form the monthly composite.

#### Step 1: Temporal Alignment via Mixed Data Sampling (MIDAS)

The MIDAS framework is employed to aggregate the daily Geopolitical Risk data (![]()) into a monthly component (![]()) using a time-decaying weighting scheme. This ensures that daily shocks closer to the end of the month are weighted more heavily, reflecting the concept that recent information has a greater impact on market expectations.^4^

Let ![]() be the daily GPR observation on day ![]() of month ![](), and ![]() be the number of trading days in that month. The aggregated monthly GPR contribution, ![](), is defined as:

![]()The function ![]() is a parsimonious, typically smooth, function (such as a Beta polynomial) that governs the weights ![](), where ![](). The parameter vector ![]() (which includes parameters controlling the shape and decay of the weights) is estimated empirically within the model estimation process, optimizing the informational integration of the daily shocks.

#### Step 2: Final Composite Index (![]())

The final monthly index ![]() is constructed as a weighted average of the monthly Climate Risk Index (![]()) and the constructed aggregated monthly GPR component (![]()). All input indices must first be normalized (e.g., standardized or scaled to a mean of 100) to ensure comparability.

![]()* ![]() and ![]() are the structural weights assigned to the Climate Risk and Geopolitical Risk components, where ![]().

* These weights can be set equally (e.g., ![]()) for neutrality, or they can be statistically determined using methods like Principal Component Analysis (PCA) to allocate weight based on each index’s contribution to the total variance in the overall global risk environment.

The resulting ![]() is a statistically rigorous monthly composite index that avoids simple averaging while optimally integrating high-frequency shock information. This index serves as the robust driving factor (![]()) in the state transition equation for the ambiguity aversion index ![](), allowing for precise measurement of how global systemic uncertainty dynamically shifts Chinese market sentiment.
