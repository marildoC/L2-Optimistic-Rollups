# scripts/ping_l2s.py
from __future__ import annotations
from web3 import Web3
from datetime import datetime, timezone
import time, os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)  # <— force override

# Load .env so OP_RPC/BASE_RPC/ARB_RPC are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def iso_utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()

CHAIN_IDS = {"optimism": 10, "base": 8453, "arbitrum": 42161}

# Read ENV first (strip whitespace). If empty, we’ll fall back.


ENV_URLS = {
    "optimism": (os.getenv("OP_RPC") or "").strip(),
    "base":     (os.getenv("BASE_RPC") or "").strip(),
    "arbitrum": (os.getenv("ARB_RPC") or "").strip(),
}



FALLBACKS = {
    "optimism": ["https://mainnet.optimism.io", "https://rpc.ankr.com/optimism"],
    "base":     ["https://mainnet.base.org",   "https://rpc.ankr.com/base"],
    "arbitrum": ["https://arb1.arbitrum.io/rpc","https://rpc.ankr.com/arbitrum"],
}

def try_provider(url: str, max_tries: int = 3):
    """
    Connect with timeout, measure round-trip, fetch latest block.
    Returns a dict with chain_id, block_number, block_time, tx_count, base_fee_wei, rtt_ms.
    """
    w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 8}))

    # measure RTT using is_connected()
    start = time.perf_counter()
    ok = w3.is_connected()
    rtt_ms = (time.perf_counter() - start) * 1000.0
    if not ok:
        # force a JSON-RPC call so we see the HTTP error (401/429/etc.)
        try:
            w3.eth.chain_id
        except Exception as e:
            raise RuntimeError(f"is_connected=False; RPC call failed: {e}")
        raise RuntimeError("is_connected=False (no further info)")

    # fetch details with a couple retries
    last_err = None
    for _ in range(max_tries):
        try:
            cid = w3.eth.chain_id
            bn  = w3.eth.block_number
            blk = w3.eth.get_block(bn)
            txc = len(blk["transactions"])
            ts  = int(blk["timestamp"])
            base_fee = blk.get("baseFeePerGas")
            return {
                "rtt_ms": rtt_ms,
                "chain_id": cid,
                "block_number": bn, 
                "block_time": ts,
                "tx_count": txc,
                "base_fee_wei": base_fee,
            }
        except Exception as e:
            last_err = e
            time.sleep(0.4)
    raise RuntimeError(last_err)



def pick_urls(chain: str) -> tuple[list[str], str]:
    """Return (urls, reason). If ENV present → [ENV] only; else fallbacks."""
    env_url = ENV_URLS.get(chain, "")
    if env_url:
        return [env_url], f"ENV {chain.upper()}_RPC"
    return FALLBACKS[chain], "public fallback"

def main():
    print("L2 connectivity probe (round-trip + latest block)\n")
    for chain in ("optimism", "base", "arbitrum"):
        urls, reason = pick_urls(chain)
        print(f"=== {chain.upper()} ===")
        print(f"  Selection   : {reason}")
        for i, u in enumerate(urls, 1):
            print(f"  Candidate {i}: {u}")
        try:
            # Try the first (and only, if ENV) URL
            url = urls[0]
            data = try_provider(url)
            if data["chain_id"] != CHAIN_IDS[chain]:
                print(f"  ❌ Chain ID mismatch: got {data['chain_id']} expected {CHAIN_IDS[chain]}")
            print(f"  Provider    : {url}")
            print(f"  Chain ID    : {data['chain_id']}")
            print(f"  RPC RTT     : {data['rtt_ms']:.1f} ms")
            print(f"  Block       : {data['block_number']}")
            print(f"  Block time  : {iso_utc(data['block_time'])} (UTC)")
            print(f"  Tx count    : {data['tx_count']}")
            if data['base_fee_wei'] is not None:
                print(f"  Base fee    : {data['base_fee_wei']/1e9:.2f} gwei")
        except Exception as e:
            print(f"  Error       : {e}")
        print()

if __name__ == "__main__":
    main()
