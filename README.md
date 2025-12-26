# Empirical Analysis of Ethereum Layer-2 Optimistic Rollups

This repository presents a **data-driven empirical analysis of Ethereum Layer-2 Optimistic Rollups**, with a focused comparison of **Optimism**, **Base**, and **Arbitrum**.  

The study evaluates how architectural design choices of optimistic rollups manifest in practice by measuring **latency**, **finality behavior**, **fee characteristics**, and **transaction composition** using live on-chain data.

The repository contains the **complete analytical implementation** required to reproduce the results discussed in the accompanying **PDF report**, which provides the theoretical background, research questions, methodology, and interpretation of findings.

---

## 🔍 Project Overview

Optimistic Rollups scale Ethereum by executing transactions off-chain while relying on Layer 1 for data availability and dispute resolution. Rather than providing a purely descriptive comparison, this project adopts an **empirical approach**, focusing on **observable performance and cost metrics** derived from real transaction data.

### Key Analytical Areas:
* **Methodology:** Comparable measurement across diverse rollup systems.
* **Reproducibility:** Standardized data collection and normalization.
* **Performance:** Quantitative evaluation of user-visible metrics.
* **Maturity:** Interpretation of trust and architectural characteristics.

---

## 📁 Repository Contents

The repository includes the following components:
* **Collection Pipeline:** Python-based scripts for fetching recent Layer-2 transactions.
* **Data Processing:** Scripts for normalization and aggregation.
* **Analytical Modules:** Logic for latency, fee, and transaction-type evaluation.
* **Visualizations:** Generated figures used directly in the final PDF report.
* **Documentation:** Configuration files documenting analytical assumptions.

*Note: Large raw datasets, virtual environments, and private credentials are excluded to maintain security and repository clarity.*

---

## ⚙️ Reproducing the Analysis

The analysis can be reproduced by executing the pipeline locally.

### Installation & Execution

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment (Windows)
.venv\Scripts\activate
# Activate the environment (macOS / Linux)
source .venv/bin/activate

# Install required dependencies
pip install -r rollup_project/requirements.txt

# Execute the pipeline
python rollup_project/scripts/collect_recent.py
python rollup_project/scripts/clean_join.py
python rollup_project/scripts/latency_suite.py
python rollup_project/scripts/quick_plots.py
[!IMPORTANT] Running the pipeline regenerates the figures and metrics discussed in the report. Since the data is collected from live blockchain endpoints, numerical results may vary slightly depending on the exact time of execution.

📄 Relation to the PDF Report
This repository represents the implementation layer that produces the empirical evidence referenced throughout the accompanying PDF report. The report provides:

Conceptual background on Optimistic Rollups.

Formal research questions and methodology.

Detailed comparative interpretation of the empirical results.

🎓 Academic Context
This project was developed within the Blockchain and Distributed Ledger Technologies course at Sapienza University of Rome, with an emphasis on empirical system evaluation and performance analysis.

Supervision
Massimo La Morgia - Sapienza University of Rome

Author
Marildo Cani - MSc Computer Science (Blockchain & Distributed Systems)
