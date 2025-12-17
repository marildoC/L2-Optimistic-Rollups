#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

"""
Clean + Join L2 tx CSVs (Base/Optimism/Arbitrum) → enriched Parquet.

Input: 
  data/raw/*.csv  (files produced by your collect scripts)

Output:
  data/clean/txs_all.parquet
  data/clean/sample_head.csv
  .cache/address_code_state.json  (contract/EOA cache)

Enrichments:
  - chain_id
  - is_contract_from, is_contract_to
  - tx_type  (transfer | contract_call | system | self_transfer | unknown)
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from web3 import Web3
# replace the old import with this block
# replace the old import with this block
# POA middleware import that works across Web3 v5 and v6
# Works across Web3 v5/v6/v7
POA_MIDDLEWARE = None
try:
    # v5 still exposes the helper function
    from web3.middleware import geth_poa_middleware as POA_MIDDLEWARE   # type: ignore
except Exception:
    try:
        # v6/v7 provide the class here
        from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware as POA_MIDDLEWARE  # type: ignore
    except Exception:
        POA_MIDDLEWARE = None



# ---------- Config ----------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"  
CLEAN_DIR = ROOT / "data" / "clean"
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = CACHE_DIR / "address_code_state.json"  # {(chain,address): bool_is_contract}

# Known chain IDs
CHAIN_IDS = {
    "optimism": 10,
    "base": 8453,
    "arbitrum": 42161,
}

# Simple heuristic: addresses that start with 0x4200... are system contracts on OP-stack chains
SYSTEM_ADDR_PREFIX = "0x4200"
HEX_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def _load_cache() -> Dict[str, bool]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, bool]) -> None:
    tmp = CACHE_PATH.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    tmp.replace(CACHE_PATH)


def _normalize_addr(x: str) -> str:
    if pd.isna(x) or not isinstance(x, str):
        return ""
    x = x.strip()
    return x if HEX_ADDR_RE.match(x) else ""


def _build_web3_per_chain(df: pd.DataFrame) -> Dict[str, Web3]:
    """
    Build one Web3 per chain using the first provider URL seen for that chain (from CSV column 'provider').
    Adds POA middleware because these L2s use it.
    """
    chain_to_provider = {}
    for chain in df["chain"].dropna().unique():
        # pick first provider encountered for this chain
        sub = df[df["chain"] == chain]
        prov = sub["provider"].dropna().astype(str).unique()
        if len(prov):
            chain_to_provider[chain] = prov[0]

    w3_map = {}
    for chain, url in chain_to_provider.items():
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 20}))
        # OP & Base & Arbitrum run geth-style engines → POA middleware
        #w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        if POA_MIDDLEWARE:
            w3.middleware_onion.inject(POA_MIDDLEWARE, layer=0)
        if not w3.is_connected():
            print(f"[warn] Could not connect to {chain} provider: {url}", file=sys.stderr)
        w3_map[chain] = w3
    return w3_map


def _is_contract_batch(
    w3: Web3, pairs: Tuple[Tuple[str, str], ...], cache: Dict[str, bool], sleep_s: float = 0.0
) -> None:
    """
    Populate cache for (chain, address) pairs using eth_getCode.
    We keep it simple (sequential) but cache + batching by unique pairs keeps it fast.
    """
    for chain, addr in pairs:
        key = f"{chain}|{addr.lower()}"
        if not addr or key in cache:
            continue
        try:
            code = w3.eth.get_code(Web3.to_checksum_address(addr))
            cache[key] = (code is not None) and (len(code) > 2)  # "0x" means EOA
        except Exception:
            # be robust: assume unknown → False for now (EOA), but don't mark permanently
            # to avoid poisoning cache in case of transient error, skip setting cache here
            pass
        if sleep_s:
            time.sleep(sleep_s)


def infer_tx_type(row) -> str:
    """
    Bucketing logic:
      - system: to begins with 0x4200... OR from==0xDeaD... special system sender used in some batches
      - self_transfer: from == to and not a system address
      - transfer: calldata_bytes == 0 AND value_eth > 0 AND to is EOA
      - contract_call: calldata_bytes > 0 OR to is contract
      - unknown: fallback
    """
    to = row.get("to", "") or ""
    from_ = row.get("from", "") or ""
    calldata_bytes = row.get("calldata_bytes", 0) or 0
    val = row.get("value_eth", 0.0) or 0.0
    is_c_to = row.get("is_contract_to", False)
    is_c_from = row.get("is_contract_from", False)

    if (to.lower().startswith(SYSTEM_ADDR_PREFIX) or
        from_.lower() == "0xdeaddeaddeaddeaddeaddeaddeaddeaddead0001"):
        return "system"

    if to and from_ and to.lower() == from_.lower():
        return "self_transfer"

    if (not is_c_to) and calldata_bytes == 0 and (val or 0) > 0:
        return "transfer"

    if calldata_bytes > 0 or is_c_to or is_c_from:
        return "contract_call"

    return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Clean + Join L2 CSVs → enriched Parquet")
    ap.add_argument("--raw_dir", default=str(RAW_DIR), help="Folder with input CSVs (default: data/raw)")
    ap.add_argument("--out_parquet", default=str(CLEAN_DIR / "txs_all.parquet"), help="Output Parquet path")
    ap.add_argument("--head_csv", default=str(CLEAN_DIR / "sample_head.csv"), help="Small CSV preview")
    ap.add_argument("--max_addr_checks", type=int, default=15000,
                    help="Safety limit: max (chain,address) code checks per run")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Optional sleep (seconds) between eth_getCode calls (avoid rate limits)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    files = list(raw_dir.glob("*.csv"))
    if not files:
        print(f"[error] No CSVs in {raw_dir}. Run the collector first.", file=sys.stderr)
        sys.exit(1)

    # ---- Load & union ----
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"[warn] Skipping {f.name}: {e}", file=sys.stderr)
    if not dfs:
        print("[error] No readable CSV files.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)

    # ---- Normalize columns ----
    # Ensure these columns exist (fill missing with defaults)
    expected_cols = [
        "chain","provider","tx_hash","from","to","value_eth","nonce",
        "l2_block_number","l2_block_ts","l2_block_ts_iso",
        "calldata_bytes","gas_used","effective_gas_price_wei","fee_native_eth","l1_fee_wei",
        "collection_window_id",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Basic dtype cleanup
    for c in ["chain","provider","tx_hash","from","to","collection_window_id","l2_block_ts_iso"]:
        df[c] = df[c].astype(str).fillna("")

    numeric_cols = [
        "value_eth","nonce","l2_block_number","l2_block_ts","calldata_bytes",
        "gas_used","effective_gas_price_wei","fee_native_eth","l1_fee_wei"
    ]
    for c in numeric_cols:
        # convert errors to NaN then fill
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Address normalization (keep original columns too)
    df["from"] = df["from"].map(_normalize_addr)
    df["to"] = df["to"].map(_normalize_addr)

    # Chain id
    df["chain"] = df["chain"].str.lower()
    df["chain_id"] = df["chain"].map(CHAIN_IDS).astype("Int64")

    # Datetime
    # l2_block_ts is seconds; keep as int; also create a datetime column (UTC)
    df["l2_block_ts"] = pd.to_numeric(df["l2_block_ts"], errors="coerce").astype("Int64")
    df["l2_block_datetime"] = pd.to_datetime(df["l2_block_ts"], unit="s", utc=True)

    # Deduplicate (chain, tx_hash) pair is unique
    before = len(df)
    df = df.drop_duplicates(subset=["chain","tx_hash"]).reset_index(drop=True)
    after = len(df)
    print(f"[info] Deduped rows: {before - after} removed, {after} remain.")

    # ---- Build Web3 per chain ----
    w3_map = _build_web3_per_chain(df)

    # ---- Contract detection with cache & budget ----
    cache = _load_cache()
    # Collect unique (chain, addr) pairs we want to check
    addr_pairs = []
    for _, row in df.iterrows():
        ch = row["chain"]
        if ch not in w3_map:  # skip unknown chain
            continue
        for col in ("from", "to"):
            addr = row[col]
            if addr:
                key = f"{ch}|{addr.lower()}"
                if key not in cache:
                    addr_pairs.append((ch, addr))

    # De-duplicate pairs
    seen = set()
    unique_pairs = []
    for ch, addr in addr_pairs:
        k = (ch, addr.lower())
        if k not in seen:
            seen.add(k)
            unique_pairs.append((ch, addr))

    # Respect safety cap
    unique_pairs = unique_pairs[: args.max_addr_checks]
    print(f"[info] Address code checks to perform (capped): {len(unique_pairs)}")

    # Execute per chain to reuse connections
    pairs_by_chain = defaultdict(list)
    for ch, addr in unique_pairs:
        pairs_by_chain[ch].append((ch, addr))

    for ch, pairs in pairs_by_chain.items():
        w3 = w3_map.get(ch)
        if not w3:
            continue
        print(f"[info] Checking {len(pairs)} addresses on {ch}...")
        _is_contract_batch(w3, tuple(pairs), cache, sleep_s=args.sleep)

    # Save cache
    _save_cache(cache)

    # Attach flags to dataframe
    def is_contract(chain: str, addr: str) -> bool:
        if not chain or not addr:
            return False
        return cache.get(f"{chain}|{addr.lower()}", False)

    df["is_contract_from"] = df.apply(lambda r: is_contract(r["chain"], r["from"]), axis=1)
    df["is_contract_to"]   = df.apply(lambda r: is_contract(r["chain"], r["to"]), axis=1)

    # ---- tx_type bucketing ----
    df["calldata_bytes"] = df["calldata_bytes"].fillna(0).astype("Int64")
    df["value_eth"] = df["value_eth"].fillna(0.0).astype(float)
    df["tx_type"] = df.apply(infer_tx_type, axis=1)

    # ---- Final column order (nice & stable) ----
    COLS = [
        "chain","chain_id","provider","collection_window_id",
        "tx_hash","from","to","is_contract_from","is_contract_to","tx_type",
        "value_eth","nonce",
        "l2_block_number","l2_block_ts","l2_block_datetime","l2_block_ts_iso",
        "calldata_bytes","gas_used","effective_gas_price_wei","fee_native_eth","l1_fee_wei",
        "source_file",
    ]
    # Add any missing (future-proof)
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[COLS]

    # ---- Write outputs ----
    out_parquet = Path(args.out_parquet)
    df.to_parquet(out_parquet, index=False)  # requires pyarrow
    print(f"[ok] Wrote {len(df):,} rows → {out_parquet}")

    # Quick human preview
    head_csv = Path(args.head_csv)
    #df.head(50).to_csv(head_csv, index=False)
    #balanced =  (
    #df.groupby("chain", group_keys=False)
     # .apply(lambda g: g.head(min(20, len(g))))
     # .reset_index(drop=True)
  #)
   # balanced.to_csv(head_csv, index=False)
    balanced = pd.concat(
        [g.head(20) for _, g in df.groupby("chain", sort=False)],
        ignore_index=True
    )
    balanced.to_csv(head_csv, index=False)
    print(f"[ok] Wrote preview → {head_csv}")


if __name__ == "__main__":
    main() 




#python clean_join.py --raw_dir .\data\raw //because we have pointed in the file at root 
#add  csv are at scripts inside , 
#so we run that line to work  

""""
🧠 What clean_join.py actually did

Think of it as a data refinery that takes raw blockchain transactions (from collect_recent.py) and turns them into a structured, analytics-ready dataset.

Here’s what each step means logically:

Reads all CSVs
It loads the 3 CSVs from:

scripts/data/raw/


(Optimism, Base, Arbitrum transactions you collected earlier).

Merges and cleans them
Combines all transactions into one big table (≈1609 rows).
It fills missing columns, converts numbers, fixes addresses (only valid 0x... ones), and standardizes field names.


Adds context columns

chain_id → converts chain name to its numeric ID (10, 8453, 42161).

l2_block_datetime → turns block timestamps into readable UTC dates.

Removes duplicate transactions.

Connects to blockchains
For each chain:

Builds a Web3 connection using the RPC provider URL.

Injects the POA middleware (needed for Layer-2 networks like Base, OP, Arbitrum).

Detects contracts vs EOAs

For each unique address (from, to), calls eth_getCode.

If the code length > 2 (“0x”), marks it as a smart contract, else an Externally Owned Account (EOA).

Stores this info in a small cache (.cache/address_code_state.json) to skip rechecking next time.

Classifies transactions
Using infer_tx_type():

system → internal protocol/system contracts

self_transfer → same address sends to itself

transfer → simple ETH/asset transfer

contract_call → involves smart contracts

unknown → fallback case

Exports results

data/clean/txs_all.parquet → the full processed dataset (efficient binary format).

data/clean/sample_head.csv → small human-readable preview (first 50 rows)

"""""



