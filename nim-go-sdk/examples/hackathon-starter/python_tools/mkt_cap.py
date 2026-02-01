import aiohttp

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def _cg_headers(coingecko_key: str | None) -> dict:
    headers = {}
    if coingecko_key:
        headers["x-cg-demo-api-key"] = coingecko_key
    return headers


def format_usd(val: float) -> str:
    if val is None:
        return "N/A"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"

    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    if val >= 1e3:
        return f"${val:,.0f}"
    return f"${val:.2f}"


async def get_market_caps_usd(coingecko_ids: list[str], coingecko_key: str | None) -> dict:
    """
    Returns: { "<coin_id>": <market_cap_usd or None>, ... }
    """
    if not coingecko_ids:
        return {}

    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(coingecko_ids),
        "order": "market_cap_desc",
        "per_page": min(len(coingecko_ids), 250),
        "page": 1,
        "sparkline": "false",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=_cg_headers(coingecko_key)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"/coins/markets failed {resp.status}: {text}")
            data = await resp.json()

    caps = {cid: None for cid in coingecko_ids}
    for row in data:
        cid = row.get("id")
        if cid in caps:
            caps[cid] = row.get("market_cap")

    return caps
