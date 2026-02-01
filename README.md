
# Project Name: Safe Entry

## What it does
This project is a web based tool that allows users to preview a cryptocurrency trade and assess its risk before executing it elsewhere. The user enters the asset they are trading from, the asset they are trading to, and the trade size. The system then fetches live market data and produces both a trade preview and a risk score out of 100.

The project was originally prototyped using Discord webhooks to validate the idea quickly. It has since been rebuilt with a web UI to make it easier to use, extend, and demonstrate.

The goal is not to predict price movements or provide financial advice, but to highlight structural risks such as low liquidity, small market capitalisation, or very new tokens.
## How it works
For each proposed swap, the system combines execution risk, ownership structure and market behaviour into a single score. It simulates price impact using an AMM liquidity model, penalises concentrated or thinly distributed tokens, analyses sybil-style wallet coordination, and runs Monte Carlo price simulations to estimate downside risk, VaR and profit probability.
## Risk model
The risk model blends execution, structural and market risk using established quantitative techniques. Liquidity risk is estimated via a constant-product AMM model to simulate price impact at the proposed trade size. Ownership risk is log-scaled from holder counts to capture concentration effects. Sybil risk is modelled using on-chain behavioural signals such as wallet dormancy, funding concentration, transaction synchrony and clustered ownership, aggregated non-linearly to reflect coordination stress. Market risk is assessed through Monte Carlo simulations using a jump-diffusion framework with ARMA-modelled residuals and beta-adjusted returns, from which expected return, Value at Risk and Expected Shortfall are derived and combined into a final 0–100 risk score.
## Challenges we ran into
One of the main challenges was dealing with uncertainty in the data and models. Monte Carlo simulations are useful, but small changes in assumptions, time windows or market regimes can materially change the output, especially over short horizons. On-chain data is also messy and incomplete, so ownership and sybil signals are, by nature, proxies rather than ground truth. We handled this by keeping assumptions conservative, using log scaling and stress-style metrics, and framing results as directional risk signals, not precise forecasts.
## What we learned
We learned how to bridge Python-based quantitative research with a Go backend, turning research code into production-style tools. We also learned how messy real financial data is in practice, why risk needs multiple signals rather than a single model, and how important clear assumptions and explainability are when presenting quantitative outputs to users.
## What's next?
Next, we plan to improve data quality by sampling larger holder sets via Etherscan Pro for more accurate sybil and ownership signals, refine Monte Carlo models with regime awareness, calibrate risk weights on historical outcomes, and expand multi-chain support with a clearer, more visual front end.

## Command(s)
*Prototype command '/trade' , would activate a trade triggering the discord webhook.*

