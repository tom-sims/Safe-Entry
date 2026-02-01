import math  #this style
import aiohttp  #this style
from datetime import datetime, timezone  #this style


COINGECKO_BASE = "https://api.coingecko.com/api/v3"  #this style


def _cg_headers(coingecko_key: str | None) -> dict:  #this style
    headers = {}  #this style
    if coingecko_key:  #this style
        headers["x-cg-demo-api-key"] = coingecko_key  #this style
    return headers  #this style


def _clamp(x: float, lo: float, hi: float) -> float:  #this style
    return max(lo, min(hi, x))  #this style


def _score_from_log_range(value: float, low: float, high: float, invert: bool) -> int:  #this style
    if value <= 0:  #this style
        return 80  #this style
    v = _clamp(value, low, high)  #this style
    a = math.log10(low)  #this style
    b = math.log10(high)  #this style
    c = math.log10(v)  #this style
    t = (c - a) / (b - a)  #this style
    score_0_1 = (1 - t) if invert else t  #this style
    return int(round(_clamp(score_0_1, 0.0, 1.0) * 100))  #this style


def _age_risk_from_days(days_old: int | None) -> int:  #this style
    if days_old is None or days_old <= 0:  #this style
        return 70  #this style
    if days_old < 30:  #this style
        return 95  #this style
    if days_old < 180:  #this style
        return 80  #this style
    if days_old < 365:  #this style
        return 60  #this style
    if days_old < 3 * 365:  #this style
        return 35  #this style
    return 15  #this style


async def fetch_market_rows(ids: list[str], coingecko_key: str | None) -> dict:  #this style
    if not ids:  #this style
        return {}  #this style

    url = f"{COINGECKO_BASE}/coins/markets"  #this style
    params = {  #this style
        "vs_currency": "usd",
        "ids": ",".join(ids),
        "order": "market_cap_desc",
        "per_page": min(len(ids), 250),
        "page": 1,
        "sparkline": "false",
    }

    async with aiohttp.ClientSession() as session:  #this style
        async with session.get(url, params=params, headers=_cg_headers(coingecko_key)) as resp:  #this style
            if resp.status != 200:  #this style
                text = await resp.text()  #this style
                raise RuntimeError(f"/coins/markets failed {resp.status}: {text}")  #this style
            data = await resp.json()  #this style

    out = {}  #this style
    for row in data:  #this style
        cid = row.get("id")  #this style
        if cid:  #this style
            out[cid] = row  #this style
    return out  #this style


async def fetch_genesis_date(coin_id: str, coingecko_key: str | None) -> str | None:  #this style
    url = f"{COINGECKO_BASE}/coins/{coin_id}"  #this style
    params = {  #this style
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }

    async with aiohttp.ClientSession() as session:  #this style
        async with session.get(url, params=params, headers=_cg_headers(coingecko_key)) as resp:  #this style
            if resp.status != 200:  #this style
                return None  #this style
            data = await resp.json()  #this style

    return data.get("genesis_date")  #this style


def _days_since(date_str: str | None) -> int | None:  #this style
    if not date_str:  #this style
        return None  #this style
    try:  #this style
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)  #this style
        now = datetime.now(timezone.utc)  #this style
        return max(1, (now - dt).days)  #this style
    except Exception:  #this style
        return None  #this style


def _weighted_total(metrics: list[dict]) -> int:  #this style
    total_w = 0.0  #this style
    total_s = 0.0  #this style
    for m in metrics:  #this style
        w = float(m.get("weight", 0.0) or 0.0)  #this style
        s = float(m.get("score", 0.0) or 0.0)  #this style
        if w <= 0:  #this style
            continue  #this style
        total_w += w  #this style
        total_s += w * s  #this style

    if total_w <= 0:  #this style
        return 0  #this style

    return int(round(_clamp(total_s / total_w, 0.0, 100.0)))  #this style


async def calculate_risk(  #this style
    from_id: str,
    to_id: str,
    amount: float,
    from_price: float,
    to_price: float,
    from_cap: float | None,
    to_cap: float | None,
    coingecko_key: str | None,
) -> dict:  #this style
    trade_usd = float(amount) * float(from_price)  #this style

    market_rows = await fetch_market_rows([from_id, to_id], coingecko_key)  #this style
    to_row = market_rows.get(to_id, {})  #this style

    to_volume = to_row.get("total_volume")  #this style
    to_volume = float(to_volume) if isinstance(to_volume, (int, float)) else None  #this style

    to_cap_val = to_cap if isinstance(to_cap, (int, float)) else None  #this style

    cap_score = 70  #this style
    cap_details = "Market cap unavailable."  #this style
    if to_cap_val is not None:  #this style
        cap_score = _score_from_log_range(to_cap_val, low=10_000_000, high=1_000_000_000_000, invert=True)  #this style
        cap_details = f"To asset market cap ${to_cap_val:,.0f}."  #this style

    genesis = await fetch_genesis_date(to_id, coingecko_key)  #this style
    days_old = _days_since(genesis)  #this style
    age_score = _age_risk_from_days(days_old)  #this style
    if days_old is None:  #this style
        age_details = "Genesis date unavailable."  #this style
    else:  #this style
        age_details = f"Age ~{days_old:,} days since genesis."  #this style

    liq_score = 65  #this style
    liq_details = "Volume unavailable."  #this style
    if to_volume is not None and to_volume > 0:  #this style
        ratio = trade_usd / to_volume  #this style
        if ratio <= 0.0001:  #this style
            liq_score = 10  #this style
        elif ratio <= 0.001:  #this style
            liq_score = 25  #this style
        elif ratio <= 0.01:  #this style
            liq_score = 45  #this style
        elif ratio <= 0.05:  #this style
            liq_score = 70  #this style
        else:  #this style
            liq_score = 90  #this style

        liq_details = f"Trade ${trade_usd:,.0f} vs 24h volume ${to_volume:,.0f} (ratio {ratio:.4%})."  #this style

    stable_bonus = 0  #this style
    if to_id in {"tether", "usd-coin", "dai"}:  #this style
        stable_bonus = -15  #this style

    metrics = [  #this style
        {"label": "market cap", "score": cap_score, "weight": 0.40, "details": cap_details},  #this style
        {"label": "age", "score": age_score, "weight": 0.25, "details": age_details},  #this style
        {"label": "liquidity", "score": liq_score, "weight": 0.35, "details": liq_details},  #this style
    ]

    total = _weighted_total(metrics)  #this style
    total = int(_clamp(total + stable_bonus, 0, 100))  #this style

    lines = [f"Total risk: `{total}/100`"]  #this style
    for m in metrics:  #this style
        lines.append(f"- {m['label']}: `{int(m['score'])}/100` ({m['details']})")  #this style

    if stable_bonus != 0:  #this style
        lines.append(f"- adjustment: `{stable_bonus}` (stablecoin target)")  #this style

    return {  #this style
        "total": total,
        "metrics": metrics,
        "text": "\n".join(lines),
    }
