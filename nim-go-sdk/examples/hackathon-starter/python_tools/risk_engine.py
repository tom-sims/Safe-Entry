import math
from datetime import datetime, timezone

import aiohttp
import numpy as np

# Import your Sybil detection functions
from sybil_detection import compute_sybil_metrics

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Map CoinGecko IDs (e.g. 'ethereum') to valid yfinance tickers (e.g. 'ETH-USD').
YF_TICKER_MAP = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "solana": "SOL-USD",
    "tether": "USDT-USD",
    "usd-coin": "USDC-USD",
    "dai": "DAI-USD",
}


def coingecko_id_to_yf_ticker(coin_id: str, symbol: str | None = None) -> str:
    if coin_id in YF_TICKER_MAP:
        return YF_TICKER_MAP[coin_id]

    # Fallback: use CoinGecko symbol if present (often works: eth -> ETH-USD)
    if symbol:
        sym = "".join(ch for ch in symbol.strip().upper() if ch.isalnum())
        if sym:
            return f"{sym}-USD"

    raise ValueError(f"No yfinance ticker mapping for CoinGecko id: {coin_id}")


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


async def fetch_token_contract(coin_id: str, coingecko_key: str | None) -> dict | None:
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


def _regime_vol_multiplier(attack_regime: str) -> float:
    r = (attack_regime or "").upper()
    if "HIGH" in r:
        return 1.60
    if "LOW" in r:
        return 1.10
    return 1.25


async def run_sde_simulation(
    token_ticker: str,
    S0: float,
    sybil_score: float,
    attack_regime: str,
    T: int = 7,
    n_sims: int = 1000,
    lookback_days: int = 180,
) -> dict:
    """
    Option B: Use yfinance historical data for a mapped ticker (e.g. ETH-USD),
    then run a GBM SDE Monte Carlo simulation.
    Returns dict with 'risk_metrics': var_95, mean_return, prob_profit.
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not installed in the active Python environment") from e

    # Pull daily history
    hist = yf.download(
        token_ticker,
        period=f"{lookback_days}d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True,
    )

    if hist is None or hist.empty or "Close" not in hist.columns:
        raise RuntimeError(f"insufficient data for {token_ticker}")

    closes = hist["Close"].dropna().astype(float).values
    if closes.size < 30:
        raise RuntimeError(f"insufficient data for {token_ticker}")

    # Daily log returns
    rets = np.diff(np.log(closes))
    if rets.size < 10:
        raise RuntimeError(f"insufficient returns for {token_ticker}")

    mu = float(np.mean(rets))
    sigma = float(np.std(rets))

    # Adjust risk using sybil + regime
    s = float(max(0.0, min(1.0, sybil_score)))
    vol_mult = _regime_vol_multiplier(attack_regime) * (1.0 + 0.50 * s)
    sigma_adj = max(1e-8, sigma * vol_mult)
    mu_adj = mu - (0.10 * abs(mu) * s)

    # Simulate GBM for T days
    Z = np.random.normal(0.0, 1.0, size=(n_sims, T))
    increments = (mu_adj - 0.5 * sigma_adj**2) + sigma_adj * Z
    log_paths = np.cumsum(increments, axis=1)
    ST = float(S0) * np.exp(log_paths[:, -1])

    returns = (ST / float(S0)) - 1.0
    var_95 = float(np.percentile(returns, 5))
    mean_return = float(np.mean(returns))
    prob_profit = float(np.mean(returns > 0.0))

    return {
        "risk_metrics": {
            "var_95": var_95,
            "mean_return": mean_return,
            "prob_profit": prob_profit,
        }
    }


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

    cap_score = 70
    cap_details = "Market cap unavailable."
    if to_cap_val is not None:
        cap_score = _score_from_log_range(to_cap_val, low=10_000_000, high=1_000_000_000_000, invert=True)
        cap_details = f"To asset market cap ${to_cap_val:,.0f}."

    genesis = await fetch_genesis_date(to_id, coingecko_key)
    days_old = _days_since(genesis)
    age_score = _age_risk_from_days(days_old)
    if days_old is None:
        age_details = "Genesis date unavailable."
    else:
        age_details = f"Age ~{days_old:,} days since genesis."

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

    sybil_score_raw = 0.35
    sybil_score = 35
    attack_regime = "MODERATE_RISK"
    sybil_details = "Sybil detection unavailable (native coin or unsupported chain)."

    try:
        contract_info = await fetch_token_contract(to_id, coingecko_key)
        if contract_info and contract_info.get("address"):
            contract_address = contract_info["address"]
            chain = contract_info["chain"]

            sybil_score_raw, attack_regime, _lambda_t = compute_sybil_metrics(contract_address, limit=30)
            sybil_score = int(sybil_score_raw * 100)
            sybil_details = f"Sybil score {sybil_score_raw:.3f}/1.00 ({attack_regime}) on {chain}."
    except Exception as e:
        sybil_details = f"Sybil detection failed: {type(e).__name__}: {str(e)[:120]}. Using default estimate."

    stable_bonus = 0
    if to_id in {"tether", "usd-coin", "dai"}:
        stable_bonus = -15

    try:
        ticker = coingecko_id_to_yf_ticker(
            to_id,
            to_row.get("symbol") if isinstance(to_row, dict) else None,
        )
        sde_results = await run_sde_simulation(
            token_ticker=ticker,
            S0=to_price,
            sybil_score=sybil_score_raw,
            attack_regime=attack_regime,
            T=7,
            n_sims=1000,
        )

        var_95 = sde_results["risk_metrics"]["var_95"]
        expected_return = sde_results["risk_metrics"]["mean_return"]
        prob_profit = sde_results["risk_metrics"]["prob_profit"]

        sde_score = int(min(100, abs(var_95) * 100 * 2))
        sde_details = f"7d VaR(95%): {var_95:+.2%}, E[return]: {expected_return:+.2%}, P(profit): {prob_profit:.1%}"
    except Exception as e:
        sde_score = 50
        sde_details = f"SDE simulation unavailable: {type(e).__name__}: {str(e)[:120]}"

    metrics = [
        {"label": "market cap", "score": cap_score, "weight": 0.25, "details": cap_details},
        {"label": "age", "score": age_score, "weight": 0.15, "details": age_details},
        {"label": "liquidity", "score": liq_score, "weight": 0.20, "details": liq_details},
        {"label": "sybil risk", "score": sybil_score, "weight": 0.20, "details": sybil_details},
        {"label": "SDE forecast", "score": sde_score, "weight": 0.20, "details": sde_details},
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
        "sybil_score": sybil_score_raw,
        "attack_regime": attack_regime,
    }
