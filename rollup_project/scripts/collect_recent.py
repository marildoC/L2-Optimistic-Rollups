from __future__ import annotations
import argparse
import csv
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple

from web3 import Web3
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)  


try:
    from dotenv import load_dotenv
    load_dotenv()  
except Exception:
    pass

def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

RPCS: Dict[str, List[str]] = {
    "optimism": _dedup([
        os.environ.get("OP_RPC", ""),                
        "https://mainnet.optimism.io",
        "https://rpc.ankr.com/optimism",
    ]),
    "base": _dedup([
        os.environ.get("BASE_RPC", ""),             
        "https://mainnet.base.org",
        "https://rpc.ankr.com/base",
    ]),
    "arbitrum": _dedup([
        os.environ.get("ARB_RPC", ""),               
        "https://arb1.arbitrum.io/rpc",
        "https://rpc.ankr.com/arbitrum",
    ]),
}

CHAIN_IDS = {"optimism": 10, "base": 8453, "arbitrum": 42161}

def now_utc_iso():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

def iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()

def connect_with_fallback(chain: str, timeout_s: int = 10) -> Tuple[Web3, str]:
    """Prefer the env RPC only; fall back to public only when env is empty."""
    urls = RPCS.get(chain, [])
    if urls and urls[0].strip():
        urls = [urls[0].strip()]
    else:
        urls = [u for u in urls if u.strip()]

    last_err = None
    if not urls:
        raise RuntimeError(f"No RPC URLs configured for {chain}")

    for url in urls:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": timeout_s}))
            if w3.is_connected() and w3.eth.chain_id == CHAIN_IDS[chain]:
                return w3, url
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All providers failed for {chain}. Last error: {last_err}")


def safe_get_receipt(w3: Web3, tx_hash, tries=3, base_sleep=0.3):
    for i in range(tries):
        try:
            return w3.eth.get_transaction_receipt(tx_hash)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base_sleep * (2 ** i))

def extract_l1_fee_from_receipt(chain: str, receipt: Dict[str, Any]) -> int | None:
    
    for key in ("l1Fee", "l1fee", "feeL1", "gasUsedForL1"):
        if key in receipt and receipt[key] is not None:
            try:
                return int(receipt[key])
            except Exception:
                pass
    return None

def compute_fee_eth(effective_gas_price_wei: int | None, gas_used: int | None) -> float | None:
    if effective_gas_price_wei is None or gas_used is None:
        return None
    return (effective_gas_price_wei * gas_used) / 1e18

def collect(chain: str, blocks: int, max_txs: int | None) -> List[Dict[str, Any]]:
    w3, provider = connect_with_fallback(chain)
    head = w3.eth.block_number
    start = max(0, head - blocks + 1)

    rows: List[Dict[str, Any]] = []
    collected = 0
    window_id = f"{chain}_{now_utc_iso()}"

    print(f"[{chain}] Provider: {provider}")
    print(f"[{chain}] Head: {head}, scanning blocks {start}..{head}")

    for bn in range(head, start - 1, -1):
        try:
            block = w3.eth.get_block(bn, full_transactions=True)
        except Exception as e:
            print(f"[{chain}] block {bn} error: {e}")
            continue

        block_ts = int(block["timestamp"])
        for tx in block["transactions"]:
            try:
                tx_hash = tx["hash"].hex()
                from_addr = tx.get("from")
                to_addr = tx.get("to")
                input_data = tx.get("input", "0x")
                calldata_bytes = (len(input_data) - 2) // 2 if isinstance(input_data, str) else None
                value_eth = int(tx.get("value", 0)) / 1e18

                gas_used = None
                eff_gas_price = None
                l1_fee_wei = None
                try:
                    receipt = safe_get_receipt(w3, tx["hash"])
                    gas_used = int(receipt.get("gasUsed")) if receipt.get("gasUsed") is not None else None
                    egp = receipt.get("effectiveGasPrice")
                    eff_gas_price = int(egp) if egp is not None else None
                    l1_fee_wei = extract_l1_fee_from_receipt(chain, receipt)
                except Exception:
                    pass

                total_fee_eth = compute_fee_eth(eff_gas_price, gas_used)

                row = {
                    "chain": chain,
                    "provider": provider,
                    "collection_window_id": window_id,
                    "tx_hash": tx_hash,
                    "from": from_addr,
                    "to": to_addr,
                    "value_eth": value_eth,
                    "nonce": int(tx.get("nonce", 0)),
                    "l2_block_number": bn,
                    "l2_block_ts": block_ts,
                    "l2_block_ts_iso": iso(block_ts),
                    "calldata_bytes": calldata_bytes,
                    "gas_used": gas_used,
                    "effective_gas_price_wei": eff_gas_price,
                    "fee_native_eth": total_fee_eth,
                    "l1_fee_wei": l1_fee_wei,
                }
                rows.append(row)
                collected += 1
                if max_txs and collected >= max_txs:
                    print(f"[{chain}] Reached max_txs={max_txs}, stopping.")
                    return rows
            except Exception as e:
                print(f"[{chain}] tx parse error: {e}")

    return rows

def write_csv(chain: str, rows: List[Dict[str, Any]], out_dir: str = "data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    ts = now_utc_iso()
    path = os.path.join(out_dir, f"{chain}_recent_{ts}.csv")
    if not rows:
        print(f"[{chain}] No rows to write.")
        return None
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[{chain}] Wrote {len(rows)} rows → {path}")
    return path

def main():
    parser = argparse.ArgumentParser(description="Collect recent transactions from L2 rollups.")
    parser.add_argument("--chain", choices=["optimism", "base", "arbitrum", "all"], default="all")
    parser.add_argument("--blocks", type=int, default=40, help="How many recent blocks to scan (per chain).")
    parser.add_argument("--max_txs", type=int, default=600, help="Hard cap on rows per chain to avoid rate limits.")
    args = parser.parse_args()

    chains = ["optimism", "base", "arbitrum"] if args.chain == "all" else [args.chain]
    for c in chains:
        try:
            rows = collect(c, blocks=args.blocks, max_txs=args.max_txs)
            write_csv(c, rows)
        except Exception as e:
            print(f"[{c}] Collection failed: {e}")

if __name__ == "__main__":
    main()


