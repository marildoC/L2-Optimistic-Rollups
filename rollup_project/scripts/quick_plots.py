from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True, parents=True)

REQUIRED = {
    "chain",
    "tx_hash",
    "tx_type",
    "calldata_bytes",
    "fee_native_eth",
}

CHAIN_ORDER = ["optimism", "base", "arbitrum"]

def _fail(msg: str):
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)

def _load() -> pd.DataFrame:
    path = DATA / "txs_all.parquet"
    if not path.exists():
        _fail(f"Missing {path}. Run the clean/join step first.")
    df = pd.read_parquet(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        _fail(f"Parquet missing columns: {sorted(missing)}")

    df["chain"] = df["chain"].astype(str).str.lower().str.strip()
    df["tx_type"] = df["tx_type"].fillna("unknown").astype(str)

    df["calldata_bytes"] = pd.to_numeric(df["calldata_bytes"], errors="coerce")
    df["fee_native_eth"]  = pd.to_numeric(df["fee_native_eth"],  errors="coerce")

    df = df[df["chain"].isin(CHAIN_ORDER)].copy()
    df["chain"] = pd.Categorical(df["chain"], categories=CHAIN_ORDER, ordered=True)
    return df

def _annotate_bars(ax, fmt="{:,.0f}"):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9, xytext=(0,3),
                        textcoords="offset points")

def plot_txs_per_chain(df: pd.DataFrame):
    counts = df["chain"].value_counts().reindex(CHAIN_ORDER).fillna(0).astype(int)
    ax = counts.plot(kind="bar")
    ax.set_title("Transactions per chain")
    ax.set_xlabel("")
    ax.set_ylabel("count")
    _annotate_bars(ax)
    plt.tight_layout()
    out = FIG / "txs_per_chain.png"
    plt.savefig(out, dpi=180)
    plt.clf()
    print(f"[ok] wrote {out}")

def plot_tx_types_by_chain(df: pd.DataFrame):
    pivot_abs = (
        df.pivot_table(index="chain",
                       columns="tx_type",
                       values="tx_hash",
                       aggfunc="count",
                       fill_value=0,
                       observed=False)  
          .reindex(CHAIN_ORDER, fill_value=0)
    )
    pivot_abs.plot(kind="bar", stacked=True)
    plt.title("Transaction types by chain (stacked counts)")
    plt.xlabel("")
    plt.ylabel("count")
    plt.tight_layout()
    out1 = FIG / "tx_types_by_chain.png"
    plt.savefig(out1, dpi=180)
    plt.clf()
    print(f"[ok] wrote {out1}")

    row_sums = pivot_abs.sum(axis=1).replace(0, np.nan)
    pivot_pct = (pivot_abs.T / row_sums).T * 100.0
    pivot_pct = pivot_pct.fillna(0.0)
    pivot_pct.plot(kind="bar", stacked=True)
    plt.title("Transaction types by chain (percent)")
    plt.xlabel("")
    plt.ylabel("%")
    plt.tight_layout()
    out2 = FIG / "tx_types_by_chain_percent.png"
    plt.savefig(out2, dpi=180)
    plt.clf()
    print(f"[ok] wrote {out2}")

def _safe_linregress(x: np.ndarray, y: np.ndarray):
    """
    Robust-ish linear fit:
      * finite-only
      * need >=10 points and non-zero variance
      * fit on standardized x with lstsq, fallback to polyfit
      * returns (slope, intercept, r2) or (nan, nan, nan)
    """
    msk = np.isfinite(x) & np.isfinite(y)
    x = x[msk]; y = y[msk]
    if x.size < 10:
        return np.nan, np.nan, np.nan

    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return np.nan, np.nan, np.nan

    x_mean = float(np.mean(x))
    x_s = (x - x_mean) / x_std

    A = np.vstack([x_s, np.ones_like(x_s)]).T
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        m_norm, b = float(sol[0]), float(sol[1])
        m = m_norm / x_std  
    except Exception:
        try:
            m, b = np.polyfit(x, y, 1)
        except Exception:
            return np.nan, np.nan, np.nan

    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m, b, r2

def plot_fee_vs_payload_and_leaderboard(df: pd.DataFrame):
    sub = df.dropna(subset=["calldata_bytes", "fee_native_eth"]).copy()
    sub = sub[(sub["calldata_bytes"] >= 0) & (sub["fee_native_eth"] > 0)].copy()

    if len(sub) > 20_000:
        sub = sub.sample(20_000, random_state=1)

    plt.figure(figsize=(8.5, 6.0))
    rows = []

    for ch, g in sub.groupby("chain", observed=False):
        x = g["calldata_bytes"].to_numpy(dtype=float)
        y = g["fee_native_eth"].to_numpy(dtype=float)

        plt.scatter(x, y, s=10, alpha=0.45, label=str(ch))

        if x.size >= 10:
            qlo, qhi = np.nanpercentile(x, [2.5, 97.5])
            fit_mask = (x >= qlo) & (x <= qhi)
            xf = x[fit_mask]; yf = y[fit_mask]

            m, b, r2 = _safe_linregress(xf, yf)
            if np.isfinite(m):
                xs = np.linspace(xf.min(), xf.max(), 60)
                plt.plot(xs, m*xs + b)  

                rows.append({
                    "chain": str(ch),
                    "slope_eth_per_byte": m,
                    "slope_eth_per_kib": m * 1024.0,
                    "intercept_eth": b,
                    "r2": r2,
                    "n": int(x.size)
                })

    title_lines = ["Fee vs Payload (scatter + per-chain linear fit)"]
    plt.title("\n".join(title_lines))
    plt.xlabel("calldata_bytes")
    plt.ylabel("fee_native_eth (ETH)")
    plt.legend()
    plt.tight_layout()
    out = FIG / "fee_vs_payload.png"
    plt.savefig(out, dpi=180)
    plt.clf()
    print(f"[ok] wrote {out}")

    if (sub["calldata_bytes"] > 0).any() and (sub["fee_native_eth"] > 0).any():
        plt.figure(figsize=(8.5, 6.0))
        for ch, g in sub.groupby("chain", observed=False):
            plt.scatter(g["calldata_bytes"], g["fee_native_eth"], s=10, alpha=0.45, label=str(ch))
        plt.xscale("log"); plt.yscale("log")
        plt.title("Fee vs Payload (log-log)")
        plt.xlabel("calldata_bytes (log)")
        plt.ylabel("fee_native_eth (ETH, log)")
        plt.legend()
        plt.tight_layout()
        out_log = FIG / "fee_vs_payload_log.png"
        plt.savefig(out_log, dpi=180)
        plt.clf()
        print(f"[ok] wrote {out_log}")

    if rows:
        tb = pd.DataFrame(rows).sort_values("slope_eth_per_kib", ascending=True)
        outcsv = DATA / "cost_per_kib.csv"
        tb.to_csv(outcsv, index=False)
        print(f"[ok] wrote {outcsv}")
    else:
        print("[warn] could not compute per-chain slopes (insufficient or degenerate data).")

def main():
    df = _load()
    plot_txs_per_chain(df)
    plot_tx_types_by_chain(df)
    plot_fee_vs_payload_and_leaderboard(df)
    print(f"[ok] figures written to {FIG}")

if __name__ == "__main__":
    main()


