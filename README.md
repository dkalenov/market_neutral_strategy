The goal of this project is to design and implement a **market-neutral long/short hedging strategy** with a **beta coefficient close to 0**.  
This minimizes exposure to overall market volatility (especially Bitcoin’s dominance in crypto markets) while allowing us to profit from **relative value differences** between correlated assets.

This work builds upon my previous GitHub project — [**Cointegrated Pairs Trading Bot**](https://github.com/dkalenov/Cointegrated-Pairs-Trading-bot) — where I developed a Python-based trading bot that identifies and trades cointegrated cryptocurrency pairs using **statistical arbitrage** and **mean-reversion** logic.  
The current project extends that concept toward **beta-neutral portfolio management**.

---

**Why Cointegration**

**Cointegration-based approach** allows to identify assets that move together in the long term but diverge temporarily in the short term.  
These temporary deviations offer **statistically measurable mean-reversion opportunities**.

Unlike correlation, cointegration ensures a **stationary linear relationship** between assets — allowing robust entry/exit decisions based on **z-scores** and **half-life estimates**.  
This method naturally provides **hedging** since long and short positions offset each other, reducing exposure to overall market movements.


## ****Implementation Plan****



**1. Data Preparation**

- Load and clean historical **kline (candlestick)** data for top cryptocurrencies.  
- Ensure all symbols share a unified time index, removing missing or duplicate candles.

---

**2. Beta Analysis**

- Estimate how strongly each cryptocurrency depends on **Bitcoin (BTC)** using log returns.


---

**3. Cointegration Scanning**

- Apply the **Engle–Granger two-step test** to detect statistically significant, mean-reverting long-term relationships.  
- Evaluate all symbol pairs for cointegration.  
- Retain only pairs that meet strict selection criteria:

  - p-value < 0.05  
  - Half-life < 200 bars  
  - Pair beta vs BTC ≈ 0  

---

**4. Z-Score and Signal Generation**

Compute the rolling z-score of each pair’s spread:


**Define trading signals:**

- **Go Long:**  z ≤ −2  
- **Go Short:** z ≥ +2  
- **Exit:** |z| ≤ 0.5  

Entries occur at statistical extremes; exits near equilibrium.

---

**5. Pair-Level Beta Calculation**

Identify combinations of assets which joint spread shows minimal dependency on BTC (**target β ≈ 0**).

---

**6. Position Sizing and Risk Management**

Apply **volatility parity** to balance exposure between both legs:



**Limit exposure:**

- Max notional per pair = **5% of total capital**  
- Max risk per pair = **1% of total capital**  
- Maintain **portfolio beta near zero** relative to BTC.

---

**7. Backtesting and Evaluation**

Run backtests on **≥ 1.5 years of hourly data**.

Track performance metrics:

- **CAGR** (Compound Annual Growth Rate)  
- **MDD** (Maximum Drawdown)  
- **CAGR/MDD > 1.5**  
- **Sharpe ratio** and **beta vs BTC**

Save trade logs, equity curves, and performance reports.

---

**8. Optimization and Diversification**

- Evaluate sensitivity for **z-entry/z-exit thresholds**, **lookback windows**, and **volatility weighting**.  
- Combine multiple low-correlated cointegration models into a diversified **market-neutral portfolio**.  
- Periodically re-train and re-evaluate cointegration pairs to adapt to market changes.  
- Monitor overall **portfolio beta** to maintain neutrality.

---

**9. Conclusion and Next Steps**

- **Parameter optimization:** Tune z-score thresholds, lookback windows, and half-life using grid search or Hyperopt.  
- **Advanced backtesting:** Extend the tester to stream data window-by-window to simulate live conditions.  
- **Dynamic adaptation:** Detect changes in cointegration; exit trades if relationships decay.  
- **Multi-timeframe analysis:** Explore cross-timeframe cointegration opportunities.  
- **Trade management:** Test partial take-profits, trailing stops, and dynamic stop adjustments.  
- **Integration:** Connect to Binance Futures WebSocket for real-time paper trading, then transition to live execution.





 #**Beta Analysis** | **Concept Overview**

In classical finance, the **Beta (β)** coefficient measures how strongly an asset’s returns move relative to the overall market.  
It quantifies the *systematic risk* — the portion of total risk that cannot be diversified away.

Mathematically, it is defined as:

$$
\beta = \frac{\mathrm{Cov}(R_i, R_m)}{\mathrm{Var}(R_m)}
$$

Where:

- **Rᵢ** — returns of the individual asset  
- **Rₘ** — returns of the market portfolio (e.g., S&P 500)  
- **Cov(Rᵢ, Rₘ)** — covariance between the asset and the market  
- **Var(Rₘ)** — variance of the market returns

A high β (>1) indicates that the asset amplifies market movements,  
while a low β (<1) means it moves less than the market.  
A β close to 0 implies that the asset behaves independently of market swings — **market-neutral**.


---

**Application to Cryptocurrencies**

In the cryptocurrency market, **Bitcoin (BTC)** plays the role of the *market benchmark*.  
Hence, we can rewrite the same formula as:

$$
\beta_i = \frac{\mathrm{Cov}(r_i, r_{BTC})}{\mathrm{Var}(r_{BTC})}
$$

Where:

- **rᵢ** — log returns of cryptocurrency *i*  
- **r₍BTC₎** — log returns of Bitcoin  
- **Cov(rᵢ, r₍BTC₎)** — covariance between the coin and Bitcoin  
- **Var(r₍BTC₎)** — variance of Bitcoin’s returns

This approach is inspired by the beta analysis framework presented in
**[“Cryptocurrency market structure: beta, correlations and risk” (arXiv:1808.02505)](https://arxiv.org/pdf/1808.02505)**.


---

**Hypotheses**


- **H₀:** There is **no significant dependence** between an altcoin’s returns and Bitcoin’s returns (β = 0).  
- **H₁:** The altcoin’s returns are **significantly correlated** with Bitcoin’s returns (β > 0.5).  

In practice, we are particularly interested in assets (or pairs) for which **β ≈ 0**,   indicating weak market dependence and suitability for **market-neutral hedging**.



 <img width="1320" height="836" alt="image" src="https://github.com/user-attachments/assets/00d05e7b-6072-4496-93da-0f5aed3a6e72" />

After calculating β for 247 cryptocurrencies relative to BTC:

- Average β ≈ 1.25

- Only **PAXGUSDT** has near-zero β (0.06)

- A few (like **TRXUSDT**, **SUNUSDT**) have moderate β (0.4–0.6)

- Most coins show β > 0.5, many even β > 1

- Some (like **ADAUSDT**, **XRPUSDT**, **XLMUSDT**) reach β ≈ 2



Conclusion

We reject H₀ (no correlation).
Altcoins show a **strong dependence** on BTC — they mostly move in the same direction and often with even higher volatility.

Therefore, analyzing **beta** neutrality should be done on **pairs**, not on single coins.






### 2 Position Sizing

Based on the **identified cointegrated pairs** and the computed statistical parameters  
(β — hedge ratio, z-score — deviation from equilibrium),  
we can generate **market-neutral long/short trading signals**.

The pipeline proceeds as follows:
- **Volatility-based Position Sizing**
- **Quantity Conversion**

Calculates USD allocations for both legs of a pair using inverse realized volatility
over a rolling window. Ensures that both sides contribute equally to total risk
(“volatility parity” principle). 

Converts dollar allocations into trade quantities while enforcing
a maximum notional exposure per pair (e.g., 5% of capital).
Prevents over-leverage and keeps exposure consistent.



### 3 Data Preparation and Log Transformation

Since this strategy operates over **hundreds of trading pairs** and **long historical periods**, it is computationally expensive to repeatedly load and preprocess raw price data.  

To make large-scale tests efficient, the data is **cached once** in a compact binary format. During research or parameter tuning, this cached version allows much faster access.  

In live or production environments, however, data can be **fetched and transformed on the fly** directly from the exchange.


The data preparation workflow includes:

1. **Loading and Pivoting**  
   Read the raw CSV (with `Date`, `Symbol`, `Close`) and reshape it into a matrix  
   where each column corresponds to a symbol and each row to a timestamp.

2. **Logarithmic Transformation**  
   Apply `log(price)` to stabilize variance and make returns additive.  
   Small shifts are added to handle zero or negative prices safely.

3. **Caching for Efficiency**  
   Store the processed arrays (`symbols`, `dates`, `logmat`, and `shifts`)  
   into a compressed `.npz` file.  
   This one-time operation significantly reduces preprocessing time  
   when scanning thousands of pairs across rolling windows.

This cached log matrix serves as the foundation for all subsequent  
cointegration testing, rolling window evaluation, and multi-process analysis.


### 4 Sequential search and signal generation

Rolling Window Scanning

After defining the core analytical functions for pair selection, signal generation, and position sizing, the next stage is to **run a large-scale search** across all possible symbol combinations. This process identifies which pairs are **statistically valid and tradable** under different market regimes.


Instead of testing the entire history at once, it's better to use **rolling time windows** — each representing a local market period.  

For every window, we check cointegration strength, estimate the hedge ratio, compute z-scores, and evaluate market neutrality (β to BTC).  

This approach helps capture pairs that are **temporarily cointegrated** and detect when relationships **break down or reappear**.




<img width="2400" height="1800" alt="image" src="https://github.com/user-attachments/assets/28f94620-dab9-4721-a3e8-eef8f703ecd5" />





**Observations**

Both assets move together early, then diverge sharply around June 13 — the basis for a trade entry.
BTC (orange) drops slightly during this time, but its trend does not align with the spread movement.
The normalized prices show clear mean reversion independent of BTC.
Z-score falls below −2 (entry), then reverts to 0 — a typical stationary spread.


Conclusion
- Beta to BTC is close to 0 → almost market-neutral.
- Spread behavior driven by internal price divergence, not by BTC’s trend.
- Z-score evolution confirms cointegration and mean reversion.

The pair acts independently from BTC — a good example of a market-neutral, stationary relationship.



The backtest represents a **simplified historical simulation** designed to evaluate signal performance under fixed entry and exit conditions.  
The framework does **not** employ rolling windows or dynamic recalculation of cointegration relationships — all tested pairs are assumed to remain stable throughout the backtest period.  

Trade logic is intentionally basic:  
- **Entries and exits** are based solely on static Z-score thresholds.  
- **No partial position closures**, stop or target adjustments, or dynamic position resizing are implemented.  
- **No adaptive cointegration updates** or rolling parameter recalibrations are performed.

Therefore, the results should be interpreted as a **baseline statistical validation** of the trading logic, rather than a fully dynamic or execution-accurate trading system.


**Backtest summary**

- CAGR: 3.53%
- MDD: −0.90%
- CAGR/MDD: 3.92
- Sharpe: 1.92
- Total trades: 50
- Win rate: 54.2%
- Final equity: 1,050,520  
- Beta: -0.000754



The strategy exhibits stable, low-volatility performance with minimal drawdowns and moderate annualized returns. Risk-adjusted metrics (Sharpe ≈ 1.9, CAGR/MDD ≈ 3.9) indicate strong efficiency and controlled exposure. The system behaves as a conservative market-neutral (Beta close to 0) model with consistent trade frequency and balanced win/loss distribution.

However, the headline figures are likely slightly inflated due to optimistic assumptions on transaction costs, execution latency, and potential minor accounting biases. Further validation with corrected cost modeling, out-of-sample data, and robustness checks is recommended.

Conclusion:
Overall performance is solid and consistent with a mean-reversion pairs trading framework, offering stable returns and strong downside protection. The strategy can serve as a low-risk portfolio component but requires verification of execution realism and parameter robustness before live deployment.


<img width="1588" height="790" alt="image" src="https://github.com/user-attachments/assets/5653d243-2f53-4b47-b8a3-75683954de93" />



<img width="1589" height="790" alt="image" src="https://github.com/user-attachments/assets/f3df470a-a397-4b49-9d5d-3c32b25b5354" />



<img width="1585" height="789" alt="image" src="https://github.com/user-attachments/assets/12128531-86e6-44eb-98de-700ec148bed1" />

