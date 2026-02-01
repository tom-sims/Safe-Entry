# python_tools/risk_cli.py
import sys, json, aiohttp, asyncio
from mkt_cap import get_market_caps_usd
from risk_engine import calculate_risk

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def cg_headers(key: str | None) -> dict:
    return {"x-cg-demo-api-key": key} if key else {}

async def resolve_to_coingecko_id(user_input: str, key: str | None) -> str:
    # Minimal: use /search and pick the first result
    q = (user_input or "").strip()
    if not q:
        return ""
    url = f"{COINGECKO_BASE}/search"
    params = {"query": q}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=cg_headers(key)) as resp:
            if resp.status != 200:
                return q.lower()
            data = await resp.json()
            coins = data.get("coins") or []
            if coins:
                cid = coins[0].get("id")
                if cid:
                    return cid
    return q.lower()

async def get_usd_prices(ids: list[str], key: str | None) -> dict:
    url = f"{COINGECKO_BASE}/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": "usd"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=cg_headers(key)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"/simple/price failed {resp.status}: {text}")
            return await resp.json()

async def run(payload: dict) -> dict:
    # Expected input:
    # { "from_token": "USDC", "to_token": "BTC", "amount": 1000, "coingecko_key": "..." }
    from_token = payload.get("from_token", "")
    to_token = payload.get("to_token", "")
    amount = float(payload.get("amount", 0))
    key = payload.get("coingecko_key")

    from_id = await resolve_to_coingecko_id(from_token, key)
    to_id = await resolve_to_coingecko_id(to_token, key)
    if not from_id or not to_id:
        return {"error": "Could not resolve token(s) to CoinGecko IDs."}

    prices = await get_usd_prices([from_id, to_id], key)
    if from_id not in prices or to_id not in prices:
        return {"error": "CoinGecko returned no price for one token."}

    from_price = float(prices[from_id]["usd"])
    to_price = float(prices[to_id]["usd"])

    caps = await get_market_caps_usd([from_id, to_id], key)
    from_cap = caps.get(from_id)
    to_cap = caps.get(to_id)

    # This calls your real risk function and returns its dict
    # It returns { "total", "metrics", "text" } :contentReference[oaicite:5]{index=5}
    result = await calculate_risk(
        from_id=from_id,
        to_id=to_id,
        amount=amount,
        from_price=from_price,
        to_price=to_price,
        from_cap=from_cap,
        to_cap=to_cap,
        coingecko_key=key,
    )
    return result

def main():
    payload = json.load(sys.stdin)
    out = asyncio.run(run(payload))
    json.dump(out, sys.stdout)

if __name__ == "__main__":
    main()
