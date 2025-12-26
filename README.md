# Empirical Analysis of Ethereum Layer-2 Optimistic Rollups

This repository presents a data-driven empirical analysis of Ethereum Layer-2 Optimistic Rollups, with a focused comparison of Optimism, Base, and Arbitrum. The study evaluates how architectural design choices of optimistic rollups manifest in practice by measuring transaction latency, finality anchoring behavior, fee characteristics, and transaction composition using live on-chain data.

The repository contains the complete analytical implementation required to reproduce the results discussed in the accompanying PDF report, which provides the theoretical background, research questions, methodological justification, and interpretation of findings.

Rather than offering a purely descriptive comparison, the project adopts an empirical approach based on observable performance and cost metrics derived from real transaction data. A uniform measurement methodology is applied across rollup systems to enable fair and reproducible cross-chain comparison.

The analysis emphasizes user-visible performance, L2-to-L1 settlement behavior, and economic characteristics related to calldata usage and transaction batching. Generated datasets and figures correspond directly to those referenced in the report.

Large raw datasets, virtual environments, and private credentials are intentionally excluded to maintain clarity, security, and reproducibility.

This project was developed within the academic context of Blockchain and Distributed Ledger Technologies at Sapienza University of Rome, with a focus on empirical system evaluation and performance analysis.

**Supervision:**  
Massimo La Morgia — Sapienza University of Rome

**Author:**  
Marildo Cani — MSc Computer Science  
Blockchain & Distributed Systems
