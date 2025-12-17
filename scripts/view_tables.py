#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "clean" / "overview_tables.html"

df = pd.read_parquet(ROOT / "data" / "clean" / "txs_all.parquet")

# Per-chain counts
per_chain = df.groupby("chain", dropna=False).size().rename("tx_count").reset_index()

# Tx-type breakdown per chain
tx_by_chain = (
    df.pivot_table(index="chain", columns="tx_type", values="tx_hash", aggfunc="count", fill_value=0)
      .reset_index()
)

# Fee / payload summary per chain
fee_cols = ["fee_native_eth", "calldata_bytes", "gas_used", "effective_gas_price_wei"]
fee_summary = (
    df.groupby("chain")[fee_cols]
      .agg(["count", "median", "mean", "max"])
)

# Pretty styling
def style_table(t, caption):
    sty = (t.style
              .set_caption(caption)
              .format(precision=6)
              .background_gradient(axis=None)
              .set_table_styles([{
                  "selector": "caption",
                  "props": [("font-size", "16px"), ("font-weight", "600"), ("text-align", "left"),
                            ("margin", "10px 0")]
              }])
          )
    return sty.to_html()

html = """
<html>
<head>
<meta charset="utf-8">
<title>Rollup Tables Overview</title>
<style>
 body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
 h1 { margin-bottom: 6px; }
 .note { color:#555; margin-bottom:18px; }
 table { border-collapse: collapse; margin-bottom: 24px; }
 th, td { padding: 8px 10px; border: 1px solid #eee; }
 th { background: #fafafa; }
</style>
</head>
<body>
<h1>Rollup Tables Overview</h1>
<div class="note">Source: data/clean/txs_all.parquet</div>
"""

html += style_table(per_chain, "Transactions per Chain")
html += style_table(tx_by_chain, "Transaction Types by Chain")
html += style_table(fee_summary, "Fee / Payload Summary by Chain")

html += "</body></html>"

OUT.write_text(html, encoding="utf-8")
print(f"[ok] Wrote {OUT}")



#