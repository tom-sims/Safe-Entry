import math
import aiohttp
from datetime import datetime, timezone

# Import your Sybil detection functions
from sybil_detection import compute_sybil_metrics, detect_blockchain

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def _cg_headers(coingecko_key: str | None) -> dict:
    headers = {}
    if coingecko_key:
        headers["x-cg-demo-api-key"] = coingecko_key
    return headers

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _score_from_log_range(value: float, low: float, high: float, invert: bool) -> int:
    if value <= 0:
        return 80
    v = _clamp(value, low, high)
    a = math.log10(low)
    b = math.log10(high)
    c = math.log10(v)
    t = (c - a) / (b - a)
    score_0_1 = (1 - t) if invert else t
    return int(round(_clamp(score_0_1, 0.0, 1.0) * 100))

def _age_risk_from_days(days_old: int | None) -> int:
    if days_old is None or days_old <= 0:
        return 70
    if days_old < 30:
        return 95
    if days_old < 180:
        return 80
    if days_old < 365:
        return 60
    if days_old < 3 * 365:
        return 35
    return 15

async def fetch_market_rows(ids: list[str], coingecko_key: str | None) -> dict:
    if not ids:
        return {}
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(ids),
        "order": "market_cap_desc",
        "per_page": min(len(ids), 250),
        "page": 1,
        "sparkline": "false",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=_cg_headers(coingecko_key)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"/coins/markets failed {resp.status}: {text}")
            data = await resp.json()
    out = {}
    for row in data:
        cid = row.get("id")
        if cid:
            out[cid] = row
    return out

async def fetch_genesis_date(coin_id: str, coingecko_key: str | None) -> str | None:
    url = f"{COINGECKO_BASE}/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=_cg_headers(coingecko_key)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    return data.get("genesis_date")

from arch import arch_model


async def estimate_drift_mean_reversion(token_ticker, btc_ticker="BTC-USD", days=365, theta=0.3):
    token_data = yf.Ticker(token_ticker).history(period=f"{days}d", interval="1d")
    btc_data = yf.Ticker(btc_ticker).history(period=f"{days}d", interval="1d")
    df = token_data["Close"].to_frame("token").join(btc_data["Close"].to_frame("btc"), how="inner").dropna()
    if len(df) < 100:
        raise ValueError(f"Insufficient data for {token_ticker}")
    r_token = np.log(df["token"]).diff().dropna().values
    r_btc = np.log(df["btc"]).diff().dropna().values
    n = min(len(r_token), len(r_btc))
    r_token, r_btc = r_token[-n:], r_btc[-n:]
    X = sm.add_constant(r_btc)
    beta_hat = sm.OLS(r_token, X).fit().params[1]
    long_term_mean = np.mean(r_token)
    recent_ret = np.mean(r_token[-30:])
    mean_reversion_component = theta * (long_term_mean - recent_ret)
    btc_momentum = np.mean(r_btc[-7:])
    mu_daily = mean_reversion_component + beta_hat * btc_momentum
    return mu_daily, beta_hat


async def estimate_volatility_garch(token_ticker, days=365):
    data = yf.Ticker(token_ticker).history(period=f"{days}d", interval="1d")
    if len(data) < 100:
        raise ValueError(f"Insufficient data for {token_ticker}")
    returns = np.diff(np.log(data["Close"].values)) * 100
    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
    fitted = model.fit(disp='off')
    forecast = fitted.forecast(horizon=1)
    sigma_daily = np.sqrt(forecast.variance.values[-1, 0]) / 100
    return sigma_daily


def calculate_lambda_from_sybil(sybil_score, attack_regime):
    regime_multipliers = {
        0: 1.0,
        1: 4.0
    }
    multiplier = regime_multipliers.get(attack_regime, 1.0)
    lambda_j = 0.02 * multiplier * (1 + sybil_score * 0.5)
    return lambda_j


def monte_carlo_sde(S0, mu, sigma, lambda_j, mu_j, sigma_j, T=7, dt=1, n_sims=1000):
    n_steps = int(T / dt)
    S = np.zeros((n_sims, n_steps + 1))
    S[:, 0] = S0
    for i in range(1, n_steps + 1):
        dW = np.random.randn(n_sims) * np.sqrt(dt)
        dS = mu * S[:, i-1] * dt + sigma * S[:, i-1] * dW
        n_jumps = np.random.poisson(lambda_j * dt, n_sims)
        jump_component = np.zeros(n_sims)
        for sim in range(n_sims):
            if n_jumps[sim] > 0:
                for _ in range(n_jumps[sim]):
                    Y = mu_j + sigma_j * np.random.randn()
                    jump_component[sim] += (np.exp(Y) - 1)
        S[:, i] = S[:, i-1] + dS + S[:, i-1] * jump_component
        S[:, i] = np.maximum(S[:, i], 0.01)
    return S


def calculate_risk_metrics(S, S0, confidence=0.95):
    terminal_prices = S[:, -1]
    returns = (terminal_prices - S0) / S0
    sorted_returns = np.sort(returns)
    var_index = int((1 - confidence) * len(sorted_returns))
    return {
        'var_95': sorted_returns[var_index],
        'es_95': np.mean(sorted_returns[:var_index]),
        'mean_return': np.mean(returns),
        'median_return': np.median(returns),
        'prob_profit': np.mean(terminal_prices > S0),
        'mean_final_price': np.mean(terminal_prices),
        'terminal_prices': terminal_prices
    }


async def run_sde_simulation(
    token_ticker: str,
    S0: float,
    sybil_score: float,
    attack_regime: int,
    T: int = 7,
    n_sims: int = 1000
) -> dict:
    mu, beta = await estimate_drift_mean_reversion(token_ticker, days=365, theta=0.3)
    sigma = await estimate_volatility_garch(token_ticker, days=365)
    lambda_j = calculate_lambda_from_sybil(sybil_score, attack_regime)
    mu_j = np.log(1 - sybil_score * 0.5)
    sigma_j = 0.2 + (0.5 * (1 - sybil_score))
    S = monte_carlo_sde(S0, mu, sigma, lambda_j, mu_j, sigma_j, T=T, n_sims=n_sims)
    risk_metrics = calculate_risk_metrics(S, S0)
    return {
        'mu': mu,
        'sigma': sigma,
        'lambda': lambda_j,
        'risk_metrics': risk_metrics,
        'paths': S.tolist()
    }

async def fetch_token_contract(coin_id: str, coingecko_key: str | None) -> dict | None:
    """Fetch contract address and blockchain info for a token"""
    url = f"{COINGECKO_BASE}/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=_cg_headers(coingecko_key)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

    platforms = data.get("platforms", {})

    # Priority: Ethereum first, then Solana
    if "ethereum" in platforms and platforms["ethereum"]:
        return {"chain": "ethereum", "address": platforms["ethereum"]}
    if "solana" in platforms and platforms["solana"]:
        return {"chain": "solana", "address": platforms["solana"]}

    return None

def _days_since(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(1, (now - dt).days)
    except Exception:
        return None

def _weighted_total(metrics: list[dict]) -> int:
    total_w = 0.0
    total_s = 0.0
    for m in metrics:
        w = float(m.get("weight", 0.0) or 0.0)
        s = float(m.get("score", 0.0) or 0.0)
        if w <= 0:
            continue
        total_w += w
        total_s += w * s
    if total_w <= 0:
        return 0
    return int(round(_clamp(total_s / total_w, 0.0, 100.0)))

async def calculate_risk(
    from_id: str,
    to_id: str,
    amount: float,
    from_price: float,
    to_price: float,
    from_cap: float | None,
    to_cap: float | None,
    coingecko_key: str | None,
) -> dict:
    trade_usd = float(amount) * float(from_price)
    market_rows = await fetch_market_rows([from_id, to_id], coingecko_key)
    to_row = market_rows.get(to_id, {})

    to_volume = to_row.get("total_volume")
    to_volume = float(to_volume) if isinstance(to_volume, (int, float)) else None
    to_cap_val = to_cap if isinstance(to_cap, (int, float)) else None

    # === EXISTING METRICS ===

    # Market Cap
    cap_score = 70
    cap_details = "Market cap unavailable."
    if to_cap_val is not None:
        cap_score = _score_from_log_range(to_cap_val, low=10_000_000, high=1_000_000_000_000, invert=True)
        cap_details = f"To asset market cap ${to_cap_val:,.0f}."

    # Age
    genesis = await fetch_genesis_date(to_id, coingecko_key)
    days_old = _days_since(genesis)
    age_score = _age_risk_from_days(days_old)
    if days_old is None:
        age_details = "Genesis date unavailable."
    else:
        age_details = f"Age ~{days_old:,} days since genesis."

    # Liquidity
    liq_score = 65
    liq_details = "Volume unavailable."
    if to_volume is not None and to_volume > 0:
        ratio = trade_usd / to_volume
        if ratio <= 0.0001:
            liq_score = 10
        elif ratio <= 0.001:
            liq_score = 25
        elif ratio <= 0.01:
            liq_score = 45
        elif ratio <= 0.05:
            liq_score = 70
        else:
            liq_score = 90
        liq_details = f"Trade ${trade_usd:,.0f} vs 24h volume ${to_volume:,.0f} (ratio {ratio:.4%})."

    # === NEW: SYBIL DETECTION ===

    sybil_score_raw = 0.35  # Default moderate risk
    sybil_score = 35  # Default 35/100
    attack_regime = "MODERATE_RISK"
    sybil_details = "Sybil detection unavailable (native coin or unsupported chain)."

    try:
        # Fetch contract address
        contract_info = await fetch_token_contract(to_id, coingecko_key)

        if contract_info and contract_info.get("address"):
            contract_address = contract_info["address"]
            chain = contract_info["chain"]

            # Run your Sybil detection
            sybil_score_raw, attack_regime, lambda_t = compute_sybil_metrics(contract_address, limit=30)

            # Convert 0-1 scale to 0-100 scale (higher = more risk)
            sybil_score = int(sybil_score_raw * 100)

            sybil_details = f"Sybil score {sybil_score_raw:.3f}/1.00 ({attack_regime}) on {chain}."

    except Exception as e:
        # If Sybil detection fails, use defaults and log the error
        sybil_details = f"Sybil detection failed: {str(e)[:50]}. Using default estimate."

    # === COMBINE METRICS ===

    stable_bonus = 0
    if to_id in {"tether", "usd-coin", "dai"}:
        stable_bonus = -15

    # === NEW: RUN SDE SIMULATION ===

    try:
        # Run your full SDE model (7-day horizon)
        sde_results = await run_sde_simulation(
            token_ticker=f"{to_id.upper()}-USD",  # Convert CG ID to ticker
            S0=to_price,
            sybil_score=sybil_score_raw,
            attack_regime=attack_regime,
            T=7,  # 7-day forecast
            n_sims=1000  # Reduce for speed
        )

        # Extract risk metrics
        var_95 = sde_results['risk_metrics']['var_95']
        expected_return = sde_results['risk_metrics']['mean_return']
        prob_profit = sde_results['risk_metrics']['prob_profit']

        # Convert VaR to 0-100 risk score (higher VaR = higher risk)
        sde_score = int(min(100, abs(var_95) * 100 * 2))  # Scale appropriately

        sde_details = f"7d VaR(95%): {var_95:+.2%}, E[return]: {expected_return:+.2%}, P(profit): {prob_profit:.1%}"

    except Exception as e:
        sde_score = 50
        sde_details = f"SDE simulation unavailable: {str(e)[:50]}"

    # Add to metrics
    metrics = [
        {"label": "market cap", "score": cap_score, "weight": 0.25, "details": cap_details},
        {"label": "age", "score": age_score, "weight": 0.15, "details": age_details},
        {"label": "liquidity", "score": liq_score, "weight": 0.20, "details": liq_details},
        {"label": "sybil risk", "score": sybil_score, "weight": 0.20, "details": sybil_details},
        {"label": "SDE forecast", "score": sde_score, "weight": 0.20, "details": sde_details},  # NEW
    ]

    total = _weighted_total(metrics)
    total = int(_clamp(total + stable_bonus, 0, 100))

    lines = [f"Total risk: `{total}/100`"]
    for m in metrics:
        lines.append(f"- {m['label']}: `{int(m['score'])}/100` ({m['details']})")

    if stable_bonus != 0:
        lines.append(f"- adjustment: `{stable_bonus}` (stablecoin target)")

    return {
        "total": total,
        "metrics": metrics,
        "text": "\n".join(lines),
        "sybil_score": sybil_score_raw,  # Return raw score for additional processing
        "attack_regime": attack_regime,

    }
