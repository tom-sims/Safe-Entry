import numpy as np
import pandas as pd
import requests
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone


def detect_blockchain(address):
    """
    Detects whether an address is Ethereum or Solana based on format.

    Ethereum: 42 chars, starts with 0x, hexadecimal
    Solana: 32-44 chars, base58 encoded (no 0, O, I, l)
    """
    address = address.strip()

    # Ethereum check
    if len(address) == 42 and address.startswith("0x"):
        try:
            int(address, 16)  # Valid hex?
            return "ethereum"
        except ValueError:
            pass

    # Solana check
    if 32 <= len(address) <= 44:
        # Base58 alphabet excludes 0, O, I, l
        base58_chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
        if all(c in base58_chars for c in address):
            return "solana"

    return "unknown"


def compute_sybil_metrics(token_address, limit=30):
    """
    Routes to appropriate blockchain-specific implementation.
    Returns: (sybil_score, attack_regime, jump_intensity)
    """
    blockchain = detect_blockchain(token_address)

    if blockchain == "ethereum":
        return compute_sybil_ethereum(token_address, limit)
    elif blockchain == "solana":
        return compute_sybil_solana(token_address, limit)
    else:
        raise ValueError(f"Unsupported or invalid token address: {token_address}")


def compute_sybil_ethereum(contract_address, limit):
    """
    Ethereum-specific Sybil computation.
    """
    holders = get_top_holders(contract_address, limit=limit)
    addresses = [h["HolderAddress"] for h in holders]

    D = dormant_wallet_ratio(addresses)
    F = funding_source_concentration(addresses)
    T = transaction_sync_score(addresses)

    clusters = funding_clusters(addresses)
    C = cluster_ownership_share(contract_address, clusters)

    S = severe_sybil_score(D, F, T, C)
    attack_flag = sybil_attack_regime(S)
    lambda_t = sybil_adjusted_jump_intensity(
        base_lambda=0.05,
        sybil_score=S,
        attack_regime=attack_flag
    )

    return S, attack_flag, lambda_t


def compute_sybil_solana(mint_address, limit):
    """
    Solana-specific Sybil computation.
    """
    holders = get_top_spl_holders(mint_address, limit=limit)
    wallets = [h["wallet"] for h in holders]

    D = dormant_wallet_ratio_solana(wallets)
    F = funding_source_concentration_solana(wallets)
    T = transaction_sync_score_solana(wallets)

    clusters = funding_clusters_solana(wallets)
    C = cluster_ownership_share_solana(mint_address, clusters)

    S = severe_sybil_score(D, F, T, C)
    attack_flag = sybil_attack_regime(S)
    lambda_t = sybil_adjusted_jump_intensity(
        base_lambda=0.05,
        sybil_score=S,
        attack_regime=attack_flag
    )

    return S, attack_flag, lambda_t
