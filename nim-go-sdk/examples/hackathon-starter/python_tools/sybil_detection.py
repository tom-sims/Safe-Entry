import numpy as np
import pandas as pd
import requests
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
import random

ETHERSCAN_API_KEY = "C9SI7APBI8TC9PE8C6P7UHI7Z9K6BYWU46"
ETHERSCAN_BASE = "https://api.etherscan.io/api"
SOLANA_RPC = "https://api.mainnet-beta.solana.com"
HEADERS = {"Content-Type": "application/json"}


def detect_blockchain(address):
    """
    Detects blockchain type and returns standardized metadata.
    """
    address = address.strip()
    
    # Bitcoin Detection (Legacy support - returns mock data if APIs unavailable)
    if address.startswith(("1", "3", "bc1")) and 26 <= len(address) <= 62:
        return {
            "chain": "bitcoin",
            "status": "unsupported",
            "mock": True,
            "reason": "Bitcoin analysis requires external indexers not in current scope"
        }
    
    # Ethereum Detection
    if len(address) == 42 and address.startswith("0x"):
        try:
            int(address, 16)
            return {
                "chain": "ethereum",
                "status": "supported",
                "mock": False
            }
        except ValueError:
            pass
    
    # Solana Detection (Base58)
    if 32 <= len(address) <= 44:
        base58_chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
        if all(c in base58_chars for c in address):
            return {
                "chain": "solana", 
                "status": "supported",
                "mock": False
            }
    
    return {
        "chain": "unknown",
        "status": "error",
        "mock": False,
        "reason": "Address format not recognized"
    }


def compute_sybil_metrics(token_address, limit=30):
    """
    Main entry point for Sybil analysis.
    Routes to appropriate chain implementation.
    """
    detection = detect_blockchain(token_address)
    
    if detection["status"] == "error":
        return {"error": detection.get("reason", "Unknown address format")}
    
    if detection["chain"] == "bitcoin":
        # Return mock data for Bitcoin with warning flag
        return {
            "chain": "bitcoin",
            "sybil_score": float(np.clip(np.random.normal(0.08, 0.02), 0, 1)),
            "attack_regime": 0,
            "jump_intensity": 0.04,
            "warning": "Using simulated data - Bitcoin requires additional indexing infrastructure",
            "mock": True
        }
    
    elif detection["chain"] == "ethereum":
        return compute_sybil_ethereum(token_address, limit)
    
    elif detection["chain"] == "solana":
        return compute_sybil_solana(token_address, limit)
    
    else:
        return {"error": "Unsupported blockchain"}


def get_top_holders_eth(contract_address, limit=50):
    params = {
        "module": "token",
        "action": "tokenholderlist",
        "contractaddress": contract_address,
        "page": 1,
        "offset": limit,
        "apikey": ETHERSCAN_API_KEY
    }
    try:
        r = requests.get(ETHERSCAN_BASE, params=params, timeout=10).json()
        if r.get("status") == "0":
            print(f"Etherscan API warning: {r.get('result', 'Unknown error')}")
        return r.get("result", [])
    except Exception as e:
        print(f"Error fetching ETH holders: {e}")
        return []


def dormant_wallet_ratio_eth(addresses, dormant_days=30):
    if not addresses:
        return 0.0
    
    now = datetime.now(timezone.utc)
    dormant = 0
    
    for addr in addresses:
        try:
            params = {
                "module": "account",
                "action": "txlist",
                "address": addr,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "desc",
                "apikey": ETHERSCAN_API_KEY
            }
            r = requests.get(ETHERSCAN_BASE, params=params, timeout=10).json()
            txs = r.get("result", [])
            
            if not txs or (isinstance(txs, str) and txs == "Max rate limit reached"):
                dormant += 1
            else:
                try:
                    last_ts = datetime.fromtimestamp(int(txs[0]["timeStamp"]), tz=timezone.utc)
                    if (now - last_ts).days >= dormant_days:
                        dormant += 1
                except (KeyError, IndexError, ValueError):
                    dormant += 1
        except Exception as e:
            print(f"Error checking dormant status for {addr}: {e}")
            dormant += 1
        
        time.sleep(0.2)  # Rate limiting
    
    return dormant / len(addresses)


def funding_source_concentration_eth(addresses):
    if not addresses:
        return 0.0
        
    sources = []
    
    for addr in addresses:
        try:
            params = {
                "module": "account",
                "action": "txlist",
                "address": addr,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "asc",
                "apikey": ETHERSCAN_API_KEY
            }
            r = requests.get(ETHERSCAN_BASE, params=params, timeout=10).json()
            txs = r.get("result", [])
            
            if isinstance(txs, list) and len(txs) > 0:
                sources.append(txs[0]["from"].lower())
        except Exception as e:
            print(f"Error checking funding for {addr}: {e}")
        
        time.sleep(0.2)
    
    if not sources:
        return 0.0
        
    counts = Counter(sources)
    total = sum(counts.values())
    hhi = sum((c / total) ** 2 for c in counts.values())
    return hhi


def transaction_sync_score_eth(addresses, window_seconds=300):
    if not addresses:
        return 0.0
        
    timestamps = []
    
    for addr in addresses:
        try:
            params = {
                "module": "account",
                "action": "txlist",
                "address": addr,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "desc",
                "apikey": ETHERSCAN_API_KEY
            }
            r = requests.get(ETHERSCAN_BASE, params=params, timeout=10).json()
            txs = r.get("result", [])
            
            if isinstance(txs, list) and len(txs) > 0:
                timestamps.append(int(txs[0]["timeStamp"]))
        except Exception as e:
            print(f"Error checking sync for {addr}: {e}")
        
        time.sleep(0.2)
    
    timestamps.sort()
    if len(timestamps) < 2:
        return 0.0
        
    clustered = sum(1 for i in range(1, len(timestamps)) 
                   if timestamps[i] - timestamps[i-1] <= window_seconds)
    return clustered / (len(timestamps) - 1)


def funding_clusters_eth(addresses):
    clusters = defaultdict(list)
    
    for addr in addresses:
        try:
            params = {
                "module": "account",
                "action": "txlist",
                "address": addr,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "asc",
                "apikey": ETHERSCAN_API_KEY
            }
            r = requests.get(ETHERSCAN_BASE, params=params, timeout=10).json()
            txs = r.get("result", [])
            
            if isinstance(txs, list) and len(txs) > 0:
                funder = txs[0]["from"].lower()
                clusters[funder].append(addr)
        except Exception as e:
            print(f"Error clustering {addr}: {e}")
        
        time.sleep(0.2)
    
    return clusters


def cluster_ownership_share_eth(contract_address, clusters):
    if not clusters:
        return 0.0
        
    balances = {}
    total_supply = 0
    
    for funder, wallets in clusters.items():
        cluster_balance = 0
        for wallet in wallets:
            try:
                params = {
                    "module": "account",
                    "action": "tokenbalance",
                    "contractaddress": contract_address,
                    "address": wallet,
                    "tag": "latest",
                    "apikey": ETHERSCAN_API_KEY
                }
                r = requests.get(ETHERSCAN_BASE, params=params, timeout=10).json()
                result = r.get("result", "0")
                bal = int(result) if result != "Max rate limit reached" else 0
                cluster_balance += bal
                total_supply += bal
            except Exception as e:
                print(f"Error getting balance for {wallet}: {e}")
            
            time.sleep(0.2)
        
        balances[funder] = cluster_balance
    
    if total_supply == 0:
        return 0.0
        
    return max(balances.values()) / total_supply


def solana_rpc(method, params_list):
    payload = {
        "jsonrpc": "2.0", 
        "id": 1, 
        "method": method, 
        "params": params_list
    }
    try:
        r = requests.post(SOLANA_RPC, json=payload, headers=HEADERS, timeout=10).json()
        return r.get("result")
    except Exception as e:
        print(f"Solana RPC error ({method}): {e}")
        return None


def get_top_holders_sol(mint_address, limit=30):
    result = solana_rpc("getTokenLargestAccounts", [mint_address])
    if not result or "value" not in result:
        return []
    
    accounts = result["value"][:limit]
    holders = []
    
    for acc in accounts:
        try:
            acct_info = solana_rpc("getAccountInfo", [
                acc["address"], 
                {"encoding": "jsonParsed"}
            ])
            if acct_info and "value" in acct_info:
                owner = acct_info["value"]["data"]["parsed"]["info"]["owner"]
                holders.append({
                    "wallet": owner, 
                    "token_amount": int(acc["amount"])
                })
        except Exception as e:
            print(f"Error parsing Solana holder: {e}")
    
    return holders


def dormant_wallet_ratio_sol(wallets, dormant_days=30):
    if not wallets:
        return 0.0
        
    now = datetime.now(timezone.utc)
    dormant = 0
    
    for w in wallets:
        try:
            sigs = solana_rpc("getSignaturesForAddress", [w, {"limit": 1}])
            if not sigs:
                dormant += 1
            else:
                ts = sigs[0]["blockTime"]
                last_tx = datetime.fromtimestamp(ts, tz=timezone.utc)
                if (now - last_tx).days >= dormant_days:
                    dormant += 1
        except Exception as e:
            print(f"Error checking Solana dormant {w}: {e}")
            dormant += 1
        
        time.sleep(0.1)
    
    return dormant / len(wallets)


def funding_source_concentration_sol(wallets):
    if not wallets:
        return 0.0
        
    funders = []
    
    for w in wallets:
        try:
            sigs = solana_rpc("getSignaturesForAddress", [w, {"limit": 10}])
            if not sigs:
                continue
            
            oldest_sig = sigs[-1]["signature"]
            tx = solana_rpc("getTransaction", [oldest_sig, {"encoding": "jsonParsed"}])
            
            if tx and "transaction" in tx:
                instructions = tx["transaction"]["message"]["instructions"]
                for ix in instructions:
                    if ix.get("program") == "system" and ix["parsed"]["type"] == "transfer":
                        funders.append(ix["parsed"]["info"]["source"])
                        break
        except Exception as e:
            print(f"Error checking Solana funding {w}: {e}")
        
        time.sleep(0.1)
    
    if not funders:
        return 0.0
        
    counts = Counter(funders)
    total = sum(counts.values())
    hhi = sum((c / total) ** 2 for c in counts.values())
    return hhi


def transaction_sync_score_sol(wallets, window_seconds=300):
    if not wallets:
        return 0.0
        
    timestamps = []
    
    for w in wallets:
        try:
            sigs = solana_rpc("getSignaturesForAddress", [w, {"limit": 1}])
            if sigs and sigs[0].get("blockTime"):
                timestamps.append(sigs[0]["blockTime"])
        except Exception as e:
            print(f"Error checking Solana sync {w}: {e}")
        
        time.sleep(0.1)
    
    timestamps.sort()
    if len(timestamps) < 2:
        return 0.0
        
    clustered = sum(1 for i in range(1, len(timestamps)) 
                   if timestamps[i] - timestamps[i-1] <= window_seconds)
    return clustered / (len(timestamps) - 1)


def funding_clusters_sol(wallets):
    clusters = defaultdict(list)
    
    for w in wallets:
        try:
            sigs = solana_rpc("getSignaturesForAddress", [w, {"limit": 10}])
            if not sigs:
                continue
            
            oldest_sig = sigs[-1]["signature"]
            tx = solana_rpc("getTransaction", [oldest_sig, {"encoding": "jsonParsed"}])
            
            if tx and "transaction" in tx:
                instructions = tx["transaction"]["message"]["instructions"]
                for ix in instructions:
                    if ix.get("program") == "system" and ix["parsed"]["type"] == "transfer":
                        funder = ix["parsed"]["info"]["source"]
                        clusters[funder].append(w)
                        break
        except Exception as e:
            print(f"Error clustering Solana {w}: {e}")
        
        time.sleep(0.1)
    
    return clusters


def cluster_ownership_share_sol(mint_address, clusters):
    if not clusters:
        return 0.0
        
    cluster_balances = {}
    total_balance = 0
    
    for funder, wallets in clusters.items():
        bal = 0
        for w in wallets:
            try:
                resp = solana_rpc("getParsedTokenAccountsByOwner", [
                    w, 
                    {"mint": mint_address}, 
                    {"encoding": "jsonParsed"}
                ])
                if resp and "value" in resp:
                    for acct in resp["value"]:
                        amt = int(acct["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
                        bal += amt
                        total_balance += amt
            except Exception as e:
                print(f"Error getting Solana balance for {w}: {e}")
        
        cluster_balances[funder] = bal
    
    if total_balance == 0:
        return 0.0
        
    return max(cluster_balances.values()) / total_balance


def severe_sybil_score(dormant_ratio, funding_concentration, tx_sync_score, clustered_ownership, weights=None):
    if weights is None:
        weights = {"dormant": 1.0, "funding": 1.0, "sync": 1.0, "ownership": 1.0}
    
    scores = {
        "dormant": np.clip(dormant_ratio, 0, 1),
        "funding": np.clip(funding_concentration, 0, 1),
        "sync": np.clip(tx_sync_score, 0, 1),
        "ownership": np.clip(clustered_ownership, 0, 1)
    }
    
    product_term = 1.0
    for k, s in scores.items():
        product_term *= (1.0 - s) ** weights[k]
    
    return 1.0 - product_term


def sybil_attack_regime(sybil_score, threshold=0.65):
    return int(sybil_score >= threshold)


def sybil_adjusted_jump_intensity(base_lambda, sybil_score, attack_regime, kappa_continuous=1.5, kappa_regime=2.5, lambda_cap=10.0):
    lam = base_lambda * np.exp(kappa_continuous * sybil_score + kappa_regime * attack_regime)
    return min(lam, lambda_cap)


def compute_sybil_ethereum(contract_address, limit=30):
    """
    Native implementation: Real Sybil analysis for ETH tokens.
    """
    try:
        # Get holders
        holders_data = get_top_holders_eth(contract_address, limit)
        if not holders_data:
            return {
                "error": "No holder data retrieved",
                "sybil_score": 0.0,
                "attack_regime": 0,
                "jump_intensity": 0.04
            }
        
        addresses = [h["TokenHolderAddress"] for h in holders_data if "TokenHolderAddress" in h]
        
        if len(addresses) < 2:
            return {
                "error": "Insufficient holders for analysis",
                "sybil_score": 0.0,
                "attack_regime": 0,
                "jump_intensity": 0.04
            }
        
        # Calculate metrics
        print(f"Analyzing {len(addresses)} Ethereum holders...")
        
        dormant_ratio = dormant_wallet_ratio_eth(addresses, dormant_days=30)
        print(f"Dormant ratio: {dormant_ratio:.2f}")
        
        funding_conc = funding_source_concentration_eth(addresses)
        print(f"Funding concentration: {funding_conc:.2f}")
        
        tx_sync = transaction_sync_score_eth(addresses, window_seconds=300)
        print(f"Tx sync score: {tx_sync:.2f}")
        
        clusters = funding_clusters_eth(addresses)
        cluster_ownership = cluster_ownership_share_eth(contract_address, clusters)
        print(f"Cluster ownership: {cluster_ownership:.2f}")
        
        # Calculate composite score
        S = severe_sybil_score(dormant_ratio, funding_conc, tx_sync, cluster_ownership)
        attack_flag = sybil_attack_regime(S)
        lambda_t = sybil_adjusted_jump_intensity(0.04, S, attack_flag)
        
        return {
            "chain": "ethereum",
            "sybil_score": float(S),
            "attack_regime": attack_flag,
            "jump_intensity": float(lambda_t),
            "metrics": {
                "dormant_ratio": float(dormant_ratio),
                "funding_concentration": float(funding_conc),
                "transaction_sync": float(tx_sync),
                "cluster_ownership": float(cluster_ownership),
                "cluster_count": len(clusters)
            },
            "holders_analyzed": len(addresses),
            "mock": False
        }
        
    except Exception as e:
        print(f"Critical error in Ethereum analysis: {e}")
        return {
            "error": str(e),
            "chain": "ethereum",
            "sybil_score": 0.0,
            "attack_regime": 0,
            "jump_intensity": 0.04,
            "mock": False
        }


def compute_sybil_solana(mint_address, limit=30):
    """
    Native implementation: Real Sybil analysis for SOL tokens.
    """
    try:
        # Get holders
        holders_data = get_top_holders_sol(mint_address, limit)
        if not holders_data:
            return {
                "error": "No holder data retrieved from Solana",
                "sybil_score": 0.0,
                "attack_regime": 0,
                "jump_intensity": 0.04
            }
        
        wallets = [h["wallet"] for h in holders_data]
        
        if len(wallets) < 2:
            return {
                "error": "Insufficient holders for analysis",
                "sybil_score": 0.0,
                "attack_regime": 0,
                "jump_intensity": 0.04
            }
        
        # Calculate metrics
        print(f"Analyzing {len(wallets)} Solana holders...")
        
        dormant_ratio = dormant_wallet_ratio_sol(wallets, dormant_days=30)
        print(f"Dormant ratio: {dormant_ratio:.2f}")
        
        funding_conc = funding_source_concentration_sol(wallets)
        print(f"Funding concentration: {funding_conc:.2f}")
        
        tx_sync = transaction_sync_score_sol(wallets, window_seconds=300)
        print(f"Tx sync score: {tx_sync:.2f}")
        
        clusters = funding_clusters_sol(wallets)
        cluster_ownership = cluster_ownership_share_sol(mint_address, clusters)
        print(f"Cluster ownership: {cluster_ownership:.2f}")
        
        # Calculate composite score
        S = severe_sybil_score(dormant_ratio, funding_conc, tx_sync, cluster_ownership)
        attack_flag = sybil_attack_regime(S)
        lambda_t = sybil_adjusted_jump_intensity(0.04, S, attack_flag)
        
        return {
            "chain": "solana",
            "sybil_score": float(S),
            "attack_regime": attack_flag,
            "jump_intensity": float(lambda_t),
            "metrics": {
                "dormant_ratio": float(dormant_ratio),
                "funding_concentration": float(funding_conc),
                "transaction_sync": float(tx_sync),
                "cluster_ownership": float(cluster_ownership),
                "cluster_count": len(clusters)
            },
            "holders_analyzed": len(wallets),
            "mock": False
        }
        
    except Exception as e:
        print(f"Critical error in Solana analysis: {e}")
        return {
            "error": str(e),
            "chain": "solana",
            "sybil_score": 0.0,
            "attack_regime": 0,
            "jump_intensity": 0.04,
            "mock": False
        }


# Example usage for testing
if __name__ == "__main__":
    # Test with a known Ethereum token (USDC)
    print("Testing Ethereum Analysis...")
    eth_result = compute_sybil_metrics("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", limit=10)
    print(f"ETH Result: {eth_result}\n")
    
    # Test with a known Solana token (USDC)
    print("Testing Solana Analysis...")
    sol_result = compute_sybil_metrics("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", limit=10)
    print(f"SOL Result: {sol_result}")
