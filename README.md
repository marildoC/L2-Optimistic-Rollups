# Empirical Analysis of Ethereum Layer-2 Optimistic Rollups

This repository presents a rigorous empirical analysis of Ethereum Layer-2 Optimistic Rollups, with a comparative focus on Optimism, Base, and Arbitrum. The study investigates how optimistic rollup design choices translate into observable system behavior by measuring transaction latency, finality anchoring, fee characteristics, transaction composition using live on-chain data and so on.

The repository contains the full analytical implementation supporting the accompanying PDF research report, which details the theoretical background, research questions, methodology, and interpretation of results. All metrics and figures discussed in the report are derived directly from the analyses implemented here.

The project adopts a measurement-driven approach rather than a purely architectural comparison. A uniform data collection and normalization methodology is applied across rollup systems to enable fair, reproducible cross-chain evaluation. The analysis emphasizes user-visible performance, L2-to-L1 settlement behavior, and economic properties related to calldata usage and transaction batching.

This codebase is intended to be used as a reproducible analytical framework. By executing the provided data collection, processing, and analysis scripts in sequence, users can regenerate the datasets and figures presented in the report or extend the methodology to additional rollup systems or alternative observation windows.

Large raw datasets, environment-specific artifacts, and private credentials are excluded to preserve clarity, security, and reproducibility.

This work was developed in an academic context within the field of Blockchain and Distributed Ledger Technologies at Sapienza University of Rome.

**Supervision**  
Massimo La Morgia — Sapienza University of Rome

**Author**  
Marildo Cani — MSc Computer Science  
Blockchain & Distributed Systems
