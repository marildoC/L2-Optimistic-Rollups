# scripts/collect_recent.py
from __future__ import annotations
import argparse
import csv
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple

from web3 import Web3
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)  # <— force override


# ---------- .env (so OP_RPC/BASE_RPC/ARB_RPC are picked up) ----------
try:
    from dotenv import load_dotenv
    load_dotenv()  # auto-load ./.env if present
except Exception:
    pass

# ---------- Providers (env-first with public fallbacks) ----------
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
        os.environ.get("OP_RPC", ""),                # <- your Alchemy/Infura OP endpoint (from .env)
        "https://mainnet.optimism.io",
        "https://rpc.ankr.com/optimism",
    ]),
    "base": _dedup([
        os.environ.get("BASE_RPC", ""),              # <- your Alchemy/Infura Base endpoint (from .env)
        "https://mainnet.base.org",
        "https://rpc.ankr.com/base",
    ]),
    "arbitrum": _dedup([
        os.environ.get("ARB_RPC", ""),               # <- your Alchemy/Infura Arbitrum endpoint (from .env)
        "https://arb1.arbitrum.io/rpc",
        "https://rpc.ankr.com/arbitrum",
    ]),
}

CHAIN_IDS = {"optimism": 10, "base": 8453, "arbitrum": 42161}

# ---------- Helpers ----------
def now_utc_iso():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

def iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()

def connect_with_fallback(chain: str, timeout_s: int = 10) -> Tuple[Web3, str]:
    """Prefer the env RPC only; fall back to public only when env is empty."""
    urls = RPCS.get(chain, [])
    # If the first (env) URL is present, use ONLY that one.
    if urls and urls[0].strip():
        urls = [urls[0].strip()]
    else:
        # env missing → allow the public list
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

# Extract L2-specific L1 fee fields when available
def extract_l1_fee_from_receipt(chain: str, receipt: Dict[str, Any]) -> int | None:
    # OP Stack (Optimism/Base) often exposes 'l1Fee' in the receipt.
    # Arbitrum has different accounting; record only if present.
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

# ---------- Main collection ----------
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

                # Receipt for gasUsed/effectiveGasPrice (best effort)
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


#python scripts\collect_recent.py --chain all --blocks 40 --max_txs 600

#WHAT THIS file of code actually does: 

#connexts to each L2 rollup node via RPC 
#verifies connectivity and chain ID 

#scans recent blocks for transactiosn ( in example we did only 40 blocks scan)
#for each transaction, extracts relevant data (from/to addresses, value, gas used, fees, L1 fee if available)

#for each transaction it tries to get the reciept to get gas used and effective gas price 

#then it stores these data in a csv file for further analysis 



 #CSV EXPLANATION 


 #collection_window_id = unique identifier for this data collection session

 #tx_hash = unique indentifier of the transaction

#from = wallet address sending the transaction 
#often a user wallet or anther contract doing internal logici 

#to = recieption address of the transaction 

#value_eth = amount of ETH (or native token) many contract calls use 0 value because they just call functions and do not transfer funds because they are just executing logic which does not require fund transfer this is about the value being transferred

#nonce = transaction count of the sender address at the time of this transaction it helps prevent replay attacks and ensures correct ordering of transactions from the same sender

#l2_block_number = block number on the L2 where this transaction was included

#l2_block_ts = timestamp of the block in unix epoch format

#l2_block_ts_iso = human readable ISO 8601 format of the block timestamp

#calldata_bytes = size of the calldata in bytes this indicates how much data was sent with the transaction 

#gas_used = amount of gas consumed by the transaction this reflects the computational resources used

#effective_gas_price_wei = actual gas price paid per unit of gas in wei (1 ETH = 10^18 wei )

#fee_native_eth = total fee paid for the transaction in native ETH (or token) calculated as gas_used * effective_gas_price_wei

"""""

Part 1 — Decode every column (the smart way)
1) value_eth

What it is: the native ETH amount transferred from from → to in that transaction.

Why it’s often 0.0: Most Dapp interactions are contract calls (approve, swap, mint, bridge, claim, etc.). Those call functions on a contract and don’t transfer ETH as a direct value — payment is via gas, not value.

When it’s a tiny number (1e-08, 0.00023, etc.):

Micropayments, dust, or test-like “poke” transfers.

Some protocols send tiny ETH amounts to trigger or verify flows.

Smart contracts sometimes forward minuscule ETH for bookkeeping.

What is a contract? An on-chain program (address with code). Interacting with a contract is calling its function. ETH can be sent along with a call (value_eth > 0), but most calls don’t.

Takeaway: value_eth tells you “pure money moved.” Gas is separate. Most DeFi/rollup activity is contract logic with value_eth = 0.

2) calldata_bytes

What it is: the size (in bytes) of the transaction’s input data (the payload sent to a contract).

The payload encodes: function selector + ABI-encoded parameters (e.g., “swap token A→B with minOut=… deadline=…”).

Bigger calldata ⇒ more to store/post as data ⇒ higher data availability (DA) cost on L1 when the rollup posts batches.

After Ethereum’s Dencun/4844, many rollups put their data into blobs (cheaper than calldata). But your tx still has calldata for the contract call on L2; the batch that the sequencer posts to L1 may use blobs.

Takeaway: calldata_bytes is your best L2-side proxy for “how big this tx is to post,” which strongly correlates with L1 data cost of the rollup.

3) gas_used

What it is: how much compute + storage work your tx actually consumed on the chain (measured in gas units).

What is gas? A metering unit. Every EVM op (add, hash, write storage…) costs gas.

Where does gas come from? You fund it with your wallet’s ETH. When you submit a tx you specify a gas limit and price. The chain executes your tx, meters the operations, and charges you gas_used × price.

Does gas “finish” or “get bought”? There’s no global “pool.” You pay per transaction. Unused gas (difference between your limit and actual gas_used) is not charged.

On L2s, gas represents L2 computation. There’s also an L1 data component (for posting batches). Users typically see them bundled as fee components.

Takeaway: gas_used measures how much work your transaction caused. It’s the execution cost side (separate from data cost on L1).

4) effective_gas_price_wei

What it is: the actual price per unit of gas you paid (in wei, where 1 ETH = 1e18 wei).

On Ethereum L1 it’s base_fee + tip after EIP-1559.

On L2s, it’s the L2 node’s priced gas (often with its own EIP-1559-like mechanics).

Sometimes you’ll see 0 — that’s usually system/sequencer transactions or protocol-internal ops where the node reports zero (they don’t pay like a user).

Takeaway: Combine this with gas_used to get the execution fee:
execution_fee_eth = gas_used × effective_gas_price_wei / 1e18.

5) fee_native_eth

What it is: the execution fee you actually paid on the L2 in native ETH.

Usually computed as: gas_used × effective_gas_price_wei / 1e18.

Why it matters: this is what users feel immediately when sending a tx.

Note: This is not the whole story for rollups; there’s also an L1 data cost component charged by the rollup economics (can be shown separately as l1_fee_* if the node exposes it).

Takeaway: fee_native_eth = L2 compute fee. You’ll often add an L1 fee component (below) to get the total.

6) l1_fee_wei

What it is: the L1 data availability fee charged to this tx by the rollup’s economics (in wei). This accounts for the cost to post your tx’s data (or share of the batch) to Ethereum L1.

Why is it empty sometimes?

Many RPCs don’t expose it per-tx.

Different rollups expose it differently (custom receipt fields, trace fields, or fee contracts).

If the batch used blobs (EIP-4844), the L1 fee depends on blob gas; not every node backfills per-tx attribution.

How to get it reliably (three workable paths):

Use rollup-aware RPCs:

OP Stack (Optimism/Base): some nodes return l1Fee (or similarly named) in eth_getTransactionReceipt.

Arbitrum exposes L1 breakdown via arb-specific fields/APIs.
(This is node-specific — not guaranteed everywhere.)

Estimate it yourself (robust, chain-agnostic):

Find the L1 batch tx that included your L2 block(s).

If it used calldata: compute L1 data gas (roughly 4 gas per zero byte / 16 per non-zero byte) × L1 base fee; add overhead.

If it used blobs: use blob_gas_used × blob_base_fee from that L1 tx + amortize overhead.

Allocate per-tx proportionally by each tx’s calldata_bytes.

Use indexers/explorers with fee APIs (if allowed): some services publish per-tx L1 fee for OP Stack chains.

Takeaway: Empty l1_fee_wei doesn’t mean “no L1 fee”; it means “this RPC didn’t return it.” You can recover or estimate it by linking L2 blocks → their L1 batch → reading blob/calldata pricing.
"""