# 🇮🇳 MPLAD AI Intelligence System

### AI-Powered Anomaly, Fraud-Risk & Inefficiency Detection for MPLAD Scheme Implementation

> **Smart India Hackathon 2026 — Software Solution**

---

## 📌 Overview

The **MPLAD AI Intelligence System** is an AI-assisted monitoring platform designed to help identify **anomalies, financial irregularities, implementation delays, contractor concentration, duplicate/similar projects, geographic patterns, and data-quality issues** in projects implemented under the **Members of Parliament Local Area Development Scheme (MPLADS)**.

The system combines:

- 🤖 Machine Learning
- 📊 Data Analytics
- 🔎 Rule-Based Anomaly Detection
- 🗺️ Geographic Analysis
- 📈 Risk Scoring
- 💡 Explainable AI
- 👤 Human-in-the-Loop Decision Support

The objective is to help authorities **prioritize projects requiring further review** rather than automatically declaring a project fraudulent.

---

# 🎯 Problem Statement

Large-scale public infrastructure schemes involve thousands of projects, financial transactions, contractors, implementing agencies, and geographical locations.

Manually identifying suspicious patterns across such datasets can be difficult and time-consuming.

The system addresses this challenge by automatically analysing project-level data and highlighting records that exhibit unusual or potentially concerning patterns.

### Key detection areas

| Detection Area | Purpose |
|---|---|
| Financial Anomalies | Detect unusual expenditure, utilization and fund patterns |
| Project Delays | Identify delayed or overdue projects |
| Duplicate Projects | Detect identical or highly similar project records |
| Contractor Concentration | Identify unusual concentration of projects among contractors |
| Geographic Patterns | Detect spatial clusters of potentially suspicious projects |
| Data Quality | Identify invalid, incomplete or inconsistent records |
| ML Anomalies | Detect unusual multidimensional project behaviour |

---

# 🚀 Key Features

## 🤖 AI-Powered Anomaly Detection

The platform uses **Isolation Forest** to identify projects whose characteristics differ significantly from normal project patterns.

The model analyses project-level features including financial, implementation and operational information.

---

## 📊 Multi-Dimensional Risk Scoring

Each project receives a transparent risk score based on multiple indicators.

The system combines:

```text
AI Anomaly Detection
        +
Financial Analysis
        +
Delay Detection
        +
Duplicate Detection
        +
Contractor Analysis
        +
Geographic Analysis
        +
Data Quality Analysis
        ↓
   Overall Risk Score
