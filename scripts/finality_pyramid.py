

import sys, math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

try:
    import yaml
except Exception:
    yaml = None  

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
META  = ROOT / "data" / "meta"
FIG   = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = CLEAN / "latency_summary.csv"
ASSUME_YAML = META  / "finality_assumptions.yaml"

CHAIN_ORDER = ["base", "optimism", "arbitrum"]

def _fmt(x: float) -> str:
    if x is None:
        return "TBD"
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return "TBD"
        if 0 <= xf < 1.0:
            return "<1 s"
        if abs(xf - round(xf)) < 1e-6:
            return f"{int(round(xf))} s"
        return f"{xf:.1f} s"
    except Exception:
        return "TBD"

def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        print(f"[error] Missing {SUMMARY_CSV}. Run latency_suite.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(SUMMARY_CSV)
    df["chain"] = df["chain"].astype(str).str.lower().str.strip()
    df = df[df["chain"].isin(CHAIN_ORDER)].copy()
    for col in ["LatencyB_p50", "LatencyC_p50"]:
        if col not in df.columns:
            df[col] = float("nan")
    df["chain"] = pd.Categorical(df["chain"], categories=CHAIN_ORDER, ordered=True)
    return df.sort_values("chain")

def load_assumptions() -> dict:
    out = {"challenge_window": {c: "TBD" for c in CHAIN_ORDER}}
    if yaml and ASSUME_YAML.exists():
        y = yaml.safe_load(open(ASSUME_YAML, "r", encoding="utf-8")) or {}
        cw = y.get("challenge_window", {})
        out["challenge_window"].update({(k or "").lower(): v for k, v in cw.items()})
    else:
        print("[warn] YAML not found or PyYAML not installed; using generic labels.")
    return out

def _trap(ax, xc, yb, w_bot, w_top, h, face="#3b82f6", edge="#1f2937", alpha=0.12, lw=1.6):
    x0, x1   = xc - w_bot/2.0, xc + w_bot/2.0
    xt0, xt1 = xc - w_top/2.0, xc + w_top/2.0
    y0, y1   = yb, yb + h
    poly = Polygon([[x0,y0],[x1,y0],[xt1,y1],[xt0,y1]], closed=True,
                   facecolor=face, edgecolor=edge, linewidth=lw, antialiased=True, alpha=alpha)
    ax.add_patch(poly)
    return (xc, (y0+y1)/2.0)

def build_pyramid(df: pd.DataFrame, assumptions: dict):
    b = dict(zip(df["chain"], df["LatencyB_p50"]))
    c = dict(zip(df["chain"], df["LatencyC_p50"]))
    cw = assumptions.get("challenge_window", {})

    inc_txt  = "\n".join([f"{ch.capitalize()}: p50(B) = {_fmt(b.get(ch))}" for ch in CHAIN_ORDER])
    post_txt = "\n".join([f"{ch.capitalize()}: p50(C) = {_fmt(c.get(ch))}" for ch in CHAIN_ORDER])
    final_txt= "\n".join([f"{ch.capitalize()}: {cw.get(ch,'TBD')}" for ch in CHAIN_ORDER])

    plt.rcParams.update({"font.size": 11, "text.color": "#0f172a"})
    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_axis_off()

    x = 0.5
    H   = 0.18      
    GAP = 0.055     
    w_bot = [0.90, 0.78, 0.64, 0.50]
    w_top = [0.82, 0.70, 0.56, 0.44]
    y0s = [0.78, 0.78-(H+GAP), 0.78-2*(H+GAP), 0.78-3*(H+GAP)]


    centers = [
        _trap(ax, x, y0s[0], w_bot[0], w_top[0], H),
        _trap(ax, x, y0s[1], w_bot[1], w_top[1], H),
        _trap(ax, x, y0s[2], w_bot[2], w_top[2], H),
        _trap(ax, x, y0s[3], w_bot[3], w_top[3], H),
    ]

    titles = [
        "L2 Included (user-visible)",
        "L2 Safe (data posted to L1)",
        "L1 Posted (anchored on L1)",
        "L1 Final (withdrawable)",
    ]
    subs = [
        inc_txt,
        "Begins once the posting transaction lands on Ethereum (batch available)",
        post_txt,
        final_txt,
    ]

    for (cx, cy), ttl, sub in zip(centers, titles, subs):
        ax.text(cx, cy + 0.042, ttl, ha="center", va="center",
               fontsize=14, fontweight="bold")
        ax.text(cx, cy - 0.012, sub, ha="center", va="center",
               fontsize=10.0, linespacing=1.5)  

    ax.text(0.5, 0.945, "Finality Pyramid (Data-driven)",
        ha="center", va="center", fontsize=17, fontweight="bold")

    footer = (
        "B = L2 inter-block median (p50).   "
        "C = L2→L1 posting delay median (p50)."
    )
    ax.text(0.5, 0.035, footer, ha="center", va="center", fontsize=9.6)

    out_png = FIG / "finality_pyramid.png"
    out_svg = FIG / "finality_pyramid.svg"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    print(f"[ok] Wrote {out_png}")
    print(f"[ok] Wrote {out_svg}")

def main():
    df = load_summary()
    assumptions = load_assumptions()
    build_pyramid(df, assumptions)

if __name__ == "__main__":
    main()
