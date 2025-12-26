
import time 
import argparse
from pathlib import Path
import os, sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from web3 import Web3 as _Web3
except Exception:
    _Web3 = None

if TYPE_CHECKING:
    from web3 import Web3 as Web3Type
else:
    Web3Type = Any

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
CLEAN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def _require_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"[error] Missing columns in txs_all.parquet: {miss}", file=sys.stderr)
        sys.exit(1)

def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = ["chain", "l2_block_number", "l2_block_ts"]
    _require_cols(df, need)
    df["chain"] = df["chain"].astype(str).str.lower()
    df["l2_block_number"] = pd.to_numeric(df["l2_block_number"], errors="coerce").astype("Int64")
    df["l2_block_ts"] = pd.to_numeric(df["l2_block_ts"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["chain", "l2_block_number", "l2_block_ts"])
    return df

def compute_block_times(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["chain", "l2_block_number"], as_index=False)
           .agg(block_ts=("l2_block_ts", "first")))
    g = g.sort_values(["chain", "l2_block_number"])
    g["inter_block_sec"] = g.groupby("chain")["block_ts"].diff().astype("float")
    return g

def attach_latency_A_upper(df_txs: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    prev = blocks.copy()
    prev["prev_block_ts"] = prev.groupby("chain")["block_ts"].shift(1)
    prev["latencyA_upper_sec"] = (prev["block_ts"] - prev["prev_block_ts"]).astype("float")
    txs = df_txs.merge(
        prev[["chain", "l2_block_number", "latencyA_upper_sec"]],
        on=["chain", "l2_block_number"],
        how="left",
    )
    return txs

@dataclass
class L1Watch:
    chain: str
    address: str
    topic0_sig: Optional[str]
    topic0_hash: Optional[str]

def _checksummed(w3: Web3Type, addr: str) -> str:
    try: return w3.to_checksum_address(addr)
    except Exception: return addr

def _topic0_hash_from_sig(sig: str) -> str:
    """Compute keccak(topic signature) → topic0 hash (0x-prefixed)."""
    if not _Web3:
        raise RuntimeError("web3 is required to compute topic0 hash")
    h = _Web3.keccak(text=sig)
    hx = h.hex() if hasattr(h, "hex") else bytes(h).hex()
    return hx if hx.startswith("0x") else "0x" + hx


def _infer_l1_range_for_ts(w3: Web3Type, ts_min: int, ts_max: int, pad_sec: int = 1800) -> Tuple[int, int]:
    lo_t, hi_t = max(ts_min - pad_sec, 0), ts_max + pad_sec
    latest = w3.eth.get_block("latest", full_transactions=False)
    hi_bn = latest.number
    step, bn = 2500, hi_bn
    while True:
        b = w3.eth.get_block(max(bn - step, 0))
        if b.timestamp <= lo_t or bn - step <= 0:
            lo_bn = b.number; break
        bn -= step
    lo_bn2 = lo_bn
    while True:
        b = w3.eth.get_block(lo_bn2)
        if b.timestamp >= lo_t or lo_bn2 >= hi_bn:
            lo_bn = lo_bn2; break
        lo_bn2 += max(step // 5, 100)
    if latest.timestamp <= hi_t: return (lo_bn, hi_bn)
    lo, hi = lo_bn, hi_bn
    while lo < hi:
        mid = (lo + hi)//2
        t = w3.eth.get_block(mid).timestamp
        if t < hi_t: lo = mid + 1
        else: hi = mid
    return (lo_bn, min(lo, hi_bn))

def _get_logs_safe(
    w3: Web3Type,
    address: str,
    start: int,
    end: int,
    topics: Optional[List[str]] = None,
) -> List[dict]:
    """
    Get logs in chunks and split on provider errors.
    NOTE: Some providers (e.g. Infura) require hex strings for fromBlock/toBlock.
    """
    out: List[dict] = []
    CHUNK = 500  

    a = start
    while a <= end:
        b = min(a + CHUNK - 1, end)
        params = {
            "fromBlock": hex(a),   
            "toBlock": hex(b),     
            "address": address,
        }
        if topics:
            params["topics"] = topics

        try:
            logs = w3.eth.get_logs(params)
            out.extend(logs)
            a = b + 1
            continue
        except Exception as e:
            emsg = str(e)
            if ("Too Many Requests" in emsg) or ("rate limit" in emsg.lower()) or ("429" in emsg):
                time.sleep(0.8)

            if (b - a) <= 16:
                print(f"[warn] get_logs give-up {a}-{b}: {e}")
                a = b + 1 
                continue

            print(f"[warn] get_logs failed {a}-{b}: {e}; splitting…")
            mid = (a + b) // 2
            out.extend(_get_logs_safe(w3, address, a, mid, topics))
            out.extend(_get_logs_safe(w3, address, mid + 1, b, topics))
            a = b + 1

    return out


def _iter_blocks(w3: Web3Type, start: int, end: int, step: int = 5000):
    a = start
    while a <= end:
        b = min(a + step - 1, end)
        for bn in range(a, b + 1):
            yield bn
        a = b + 1

def _get_txs_to_address(w3: Web3Type, address: str, start: int, end: int) -> list[dict]:
    out, addr_lc = [], address.lower()
    for bn in _iter_blocks(w3, start, end, step=2000):
        try:
            blk = w3.eth.get_block(bn, full_transactions=True)
        except Exception as e:
            print(f"[warn] get_block({bn}) failed: {e}"); continue
        for tx in blk.get("transactions", []):
            if isinstance(tx, dict):
                to_addr = (tx.get("to") or "").lower()
            else:
                to_addr = (getattr(tx, "to", "") or "").lower()
            if to_addr == addr_lc:
                out.append({"blockNumber": bn})
    return out

def _assign_next_event_ts(l2_ts: np.ndarray, event_ts: np.ndarray) -> np.ndarray:
    l2_ts = np.asarray(l2_ts, dtype="int64")
    event_ts = np.asarray(event_ts, dtype="int64")
    if len(event_ts) == 0: return np.full(l2_ts.shape, np.nan)
    event_ts_sorted = np.sort(event_ts)
    idx = np.searchsorted(event_ts_sorted, l2_ts, side="left")
    out = np.full(l2_ts.shape, np.nan)
    mask = idx < len(event_ts_sorted)
    out[mask] = event_ts_sorted[idx[mask]]
    return out

def try_compute_latency_C(blocks: pd.DataFrame, eth_rpc: Optional[str], watches: List[L1Watch], max_delay_sec: int = 3*24*3600) -> Optional[pd.DataFrame]:
    if not eth_rpc:
        print("[info] Skipping Latency C (no RPC)."); return None
    if _Web3 is None:
        print("[info] Skipping Latency C (web3 missing)."); return None
    try:
        w3: Web3Type = _Web3(_Web3.HTTPProvider(eth_rpc, request_kwargs={"timeout": 30}))
        if not w3.is_connected():
            print("[warn] L1 RPC not reachable."); return None
    except Exception as e:
        print(f"[warn] L1 RPC init failed: {e}"); return None

    rows = []
    for chain, dfc in blocks.groupby("chain"):
        watch = next((w for w in watches if w.chain.lower()==chain.lower()), None)
        if not watch or not watch.address:
            print(f"[info] Latency C: no watch configured for {chain}; skipping."); continue

        ts_min, ts_max = int(dfc["block_ts"].min()), int(dfc["block_ts"].max())
        start_bn, end_bn = _infer_l1_range_for_ts(w3, ts_min, ts_max)
        addr = _checksummed(w3, watch.address)

        topic0 = None
        if watch.topic0_hash:
            t = watch.topic0_hash.strip()
            topic0 = t if t.startswith("0x") else "0x" + t
        elif watch.topic0_sig:
            try:
                topic0 = _topic0_hash_from_sig(watch.topic0_sig)
            except Exception as e:
                print(f"[warn] could not compute topic0 from signature for {chain}: {e}")

        print(f"[info] {chain}: L1 range {start_bn}..{end_bn}, addr={addr}, topic0={'None' if not topic0 else topic0[:10]+'…'}")

        event_ts: list[int] = []
        if topic0:
            logs = _get_logs_safe(w3, addr, start_bn, end_bn, [topic0])
            for lg in logs:
                try:
                    event_ts.append(int(w3.eth.get_block(lg["blockNumber"]).timestamp))
                except Exception:
                    pass
        else:
            txs = _get_txs_to_address(w3, addr, start_bn, end_bn)
            for t in txs:
                try:
                    event_ts.append(int(w3.eth.get_block(t["blockNumber"]).timestamp))
                except Exception:
                    pass

        if not event_ts:
            print(f"[warn] {chain}: no L1 events/txs found in the range; skipping.")
            continue

        l2_ts = dfc["block_ts"].to_numpy("int64")
        next_ts = _assign_next_event_ts(l2_ts, np.array(event_ts, dtype="int64"))
        lat = next_ts - l2_ts
        lat = np.where((np.isnan(lat)) | (lat<0) | (lat>max_delay_sec), np.nan, lat)

        part = pd.DataFrame({
            "chain": chain,
            "l2_block_number": dfc["l2_block_number"].to_numpy(),
            "l2_block_ts": l2_ts,
            "l1_post_ts": next_ts,
            "latencyC_sec": lat,
            "l1_watch_addr": addr,
            "l1_topic0": topic0 or "",
            "method": "logs" if topic0 else "to_address"
        })
        rows.append(part)

    if not rows: return None
    out = pd.concat(rows, ignore_index=True)
    out_path = CLEAN/"l1_postings.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[ok] Wrote {out_path}")
    return out

def plot_latency_hist_and_cdf(blocks: pd.DataFrame) -> None:
    d = blocks.dropna(subset=["inter_block_sec"])
    d = d[d["inter_block_sec"] >= 0]
    plt.figure(figsize=(9, 6))
    for ch, sub in d.groupby("chain"):
        plt.hist(sub["inter_block_sec"].values, bins=40, alpha=0.45, density=True, label=ch, edgecolor="none")
    plt.xlabel("Inter-block time (sec)")
    plt.ylabel("Density")
    plt.title("L2 Inter-block Time (Histogram)")
    plt.legend()
    out1 = FIG / "latency_hist.png"
    plt.tight_layout(); plt.savefig(out1, dpi=160); plt.close()
    print(f"[ok] Wrote {out1}")

    plt.figure(figsize=(9, 6))
    for ch, sub in d.groupby("chain"):
        vals = np.sort(sub["inter_block_sec"].values)
        if len(vals) == 0: continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        plt.plot(vals, y, label=f"{ch} (N={len(vals)})")
    plt.xlabel("Inter-block time (sec)")
    plt.ylabel("CDF")
    plt.title("L2 Inter-block Time (CDF)")
    plt.grid(alpha=0.3); plt.legend()
    out2 = FIG / "latency_cdf.png"
    plt.tight_layout(); plt.savefig(out2, dpi=160); plt.close()
    print(f"[ok] Wrote {out2}")

def plot_latency_summary_bars(summary: pd.DataFrame, figdir: Path, name: str = "latency_summary_bars.png") -> None:
    cols = [c for c in summary.columns if c.startswith("LatencyB_p") and c[-2:].isdigit()]
    if not cols:
        print("[warn] summary missing columns for bars; skipping."); return
    dfp = summary[["chain"] + cols].set_index("chain").sort_index()
    dfp.rename(columns=lambda c: c.replace("LatencyB_", ""), inplace=True)
    ax = dfp.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Inter-block time summary"); ax.set_ylabel("seconds")
    plt.tight_layout(); out = figdir / name
    plt.savefig(out, dpi=160); plt.close()
    print(f"[ok] Wrote {out}")

def plot_latency_c_cdf(latc: pd.DataFrame) -> None:
    d = latc.dropna(subset=["latencyC_sec"]).copy()
    if d.empty:
        print("[info] No Latency C data to plot."); return
    plt.figure(figsize=(9, 6))
    for ch, sub in d.groupby("chain"):
        vals = np.sort(sub["latencyC_sec"].astype(float).values)
        if len(vals) == 0: continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        plt.plot(vals, y, label=f"{ch} (N={len(vals)})")
    plt.xlabel("L2 → L1 posting delay (sec)")
    plt.ylabel("CDF")
    plt.title("L2 → L1 Posting Delay (Latency C)")
    plt.grid(alpha=0.3); plt.legend()
    out = FIG / "latency_c_cdf.png"
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print(f"[ok] Wrote {out}")

def _q(s: pd.Series, p: float) -> float:
    arr = s.dropna().astype(float).values
    return float(np.nanpercentile(arr, p)) if len(arr) else np.nan

def summarize(blocks: pd.DataFrame, txs_with_A: pd.DataFrame, latC: Optional[pd.DataFrame]) -> pd.DataFrame:
    b = (
        blocks.dropna(subset=["inter_block_sec"])
              .groupby("chain")["inter_block_sec"]
              .agg(count="count",
                   p50=lambda s: _q(s, 50),
                   p90=lambda s: _q(s, 90),
                   p99=lambda s: _q(s, 99),
                   mean="mean",
                   max="max")
              .rename(columns=lambda c: f"LatencyB_{c}")
    )
    a = (
        txs_with_A.dropna(subset=["latencyA_upper_sec"])
                  .groupby("chain")["latencyA_upper_sec"]
                  .agg(count="count",
                       p50=lambda s: _q(s, 50),
                       p90=lambda s: _q(s, 90),
                       p99=lambda s: _q(s, 99),
                       mean="mean",
                       max="max")
                  .rename(columns=lambda c: f"LatencyA_upper_{c}")
    )
    out = b.join(a, how="outer")

    if latC is not None and not latC.empty:
        c = (
            latC.dropna(subset=["latencyC_sec"])
                .groupby("chain")["latencyC_sec"]
                .agg(count="count",
                     p50=lambda s: _q(s, 50),
                     p90=lambda s: _q(s, 90),
                     p99=lambda s: _q(s, 99),
                     mean="mean",
                     max="max")
                .rename(columns=lambda c: f"LatencyC_{c}")
        )
        out = out.join(c, how="outer")

    return out.reset_index()

def main():
    ap = argparse.ArgumentParser(description="Latency mini-suite: A (upper), B (block times), C (L2->L1)")
    ap.add_argument("--parquet", default=str(CLEAN / "txs_all.parquet"))
    ap.add_argument("--eth_rpc", default=os.environ.get("ETH_RPC", ""), help="Ethereum L1 RPC (for Latency C)")
    ap.add_argument("--min_blocks", type=int, default=0, help="Warn if per-chain unique blocks < this")
    ap.add_argument("--op_l1_addr",      default=os.environ.get("OP_L1_ADDR", ""))
    ap.add_argument("--op_l1_topic_sig", default=os.environ.get("OP_L1_TOPIC_SIG", ""))
    ap.add_argument("--op_l1_topic0",    default=os.environ.get("OP_L1_TOPIC0", ""))
    ap.add_argument("--base_l1_addr",      default=os.environ.get("BASE_L1_ADDR", ""))
    ap.add_argument("--base_l1_topic_sig", default=os.environ.get("BASE_L1_TOPIC_SIG", ""))
    ap.add_argument("--base_l1_topic0",    default=os.environ.get("BASE_L1_TOPIC0", ""))
    ap.add_argument("--arb_l1_addr",      default=os.environ.get("ARB_L1_ADDR", ""))
    ap.add_argument("--arb_l1_topic_sig", default=os.environ.get("ARB_L1_TOPIC_SIG", ""))
    ap.add_argument("--arb_l1_topic0",    default=os.environ.get("ARB_L1_TOPIC0", ""))
    args = ap.parse_args()

    df = load_parquet(Path(args.parquet))
    blocks = compute_block_times(df)

    for ch, sub in blocks.groupby("chain"):
        n = sub["l2_block_number"].nunique()
        if args.min_blocks and n < args.min_blocks:
            print(f"[warn] {ch}: only {n} blocks from parquet; plots may look discrete.")

    txsA = attach_latency_A_upper(df, blocks)
    plot_latency_hist_and_cdf(blocks)

    watches: List[L1Watch] = []
    if args.op_l1_addr:
        watches.append(L1Watch("optimism", args.op_l1_addr, args.op_l1_topic_sig or None, args.op_l1_topic0 or None))
    if args.base_l1_addr:
        watches.append(L1Watch("base", args.base_l1_addr, args.base_l1_topic_sig or None, args.base_l1_topic0 or None))
    if args.arb_l1_addr:
        watches.append(L1Watch("arbitrum", args.arb_l1_addr, args.arb_l1_topic_sig or None, args.arb_l1_topic0 or None))

    latC = try_compute_latency_C(blocks, args.eth_rpc.strip() or None, watches)
    if latC is not None:
        plot_latency_c_cdf(latC)

    summary = summarize(blocks, txsA, latC)
    out_csv = CLEAN / "latency_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[ok] Wrote {out_csv}")

    plot_latency_summary_bars(summary, FIG, "latency_summary_bars.png")

if __name__ == "__main__":
    main()



