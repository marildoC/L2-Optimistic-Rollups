#!/usr/bin/env python3
# Builds:
# - report/trust_matrix.html (Jinja2 if available; else simple HTML)
# - figures/trust_matrix.png  (Matplotlib fallback, zero extra deps)

from pathlib import Path
import sys
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
META   = ROOT / "data" / "meta"
REPORT = ROOT / "report"
TPL_DIR= REPORT / "templates"
FIG    = ROOT / "figures"

CSV       = META / "trust_matrix.csv"
OUT_HTML  = REPORT / "trust_matrix.html"
OUT_PNG   = FIG / "trust_matrix.png"

CHAINS = ["base", "optimism", "arbitrum"]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal schema (no sources, no brackets):
      dimension,
      <chain>_status, <chain>_note, <chain>_date   for each chain
    If *_source columns exist, they are ignored.
    """
    required = ["dimension"]
    for c in CHAINS:
        required += [f"{c}_status", f"{c}_note", f"{c}_date"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        print(f"[error] trust_matrix.csv missing cols: {miss}", file=sys.stderr)
        sys.exit(1)
    return df

def css_class(status: str) -> str:
    s = (status or "").strip().lower()
    if s == "live":    return "live"
    if s == "partial": return "partial"
    return "planned"

def render_html(rows, verified_on: str):
    # Prefer Jinja2 template; fall back to simple HTML if not available
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        env = Environment(loader=FileSystemLoader(str(TPL_DIR)),
                          autoescape=select_autoescape())
        tpl = env.get_template("trust_matrix.html")
        html = tpl.render(rows=rows, chains=CHAINS, verified_on=verified_on)
    except Exception as e:
        print(f"[warn] Jinja2 not available or template error ({e}); using minimal HTML.")
        head = "<html><head><meta charset='utf-8'><title>Trust & Maturity</title></head><body>"
        th = "<tr><th>Dimension</th>" + "".join(f"<th>{c.title()}</th>" for c in CHAINS) + "</tr>"
        trs = []
        for r in rows:
            tds = [f"<td><b>{r['dimension']}</b></td>"]
            for c in CHAINS:
                cell = r[c]
                parts = [f"<b>{cell['status']}</b>"]
                if cell.get("note"): parts.append(cell["note"])
                if cell.get("date"): parts.append(f"({cell['date']})")
                tds.append("<td>" + "<br>".join(parts) + "</td>")
            trs.append("<tr>" + "".join(tds) + "</tr>")
        tail = f"</table><p>Data verified as of {verified_on}.</p></body></html>"
        html = head + "<table border=1>" + th + "".join(trs) + tail

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT_HTML}")

def render_png(rows):
    # Matplotlib fallback: simple, clean table (no sources)
    import matplotlib.pyplot as plt
    from matplotlib.table import Table

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_axis_off()

    header = ["Dimension"] + [c.title() for c in CHAINS]
    grid = [header]
    for r in rows:
        row = [r["dimension"]]
        for c in CHAINS:
            cell = r[c]
            txt = cell["status"]
            if cell.get("note"): txt += f"\n{cell['note']}"
            if cell.get("date"): txt += f"\n({cell['date']})"
            row.append(txt)
        grid.append(row)

    nrows, ncols = len(grid), len(grid[0])
    table = Table(ax, bbox=[0, 0, 1, 1])
    widths  = [0.28] + [0.24] * (ncols - 1)
    heights = [1.0 / nrows] * nrows

    for i in range(nrows):
        for j in range(ncols):
            face = "#f8fafc" if i == 0 else "white"
            table.add_cell(i, j, widths[j], heights[i], text=grid[i][j],
                           loc="center", facecolor=face)

    for j in range(ncols):
        table[(0, j)].set_text_props(weight="bold")
    ax.add_table(table)
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {OUT_PNG}")

def main():
    if not CSV.exists():
        print(f"[error] missing {CSV}", file=sys.stderr); sys.exit(1)

    df = pd.read_csv(CSV)
    df = normalize(df)

    # compute "verified_on" as the max date present in any column
    try:
        dates = []
        for c in CHAINS:
            if f"{c}_date" in df.columns:
                dates += df[f"{c}_date"].astype(str).tolist()
        verified_on = max(pd.to_datetime(dates)).date().isoformat()
    except Exception:
        verified_on = ""

    rows = []
    for _, r in df.iterrows():
        row = {"dimension": r["dimension"]}
        for c in CHAINS:
            cell = {
                "status": (r[f"{c}_status"] or "").strip().title(),
                "note":   (r.get(f"{c}_note","") or "").strip(),
                "date":   (str(r.get(f"{c}_date","")).strip() or ""),
                "css":    css_class(r[f"{c}_status"])
            }
            row[c] = cell
        rows.append(row)

    render_html(rows, verified_on)
    render_png(rows)

if __name__ == "__main__":
    main()
