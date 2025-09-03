# Sovereign Credit Risk — PoC v2

Proof-of-concept project for **predicting sovereign credit risk** using World Bank indicators, built with Python + Streamlit.

---

## Features
- Fetches macroeconomic indicators from **World Bank API**
- Feature engineering (debt ratios, inflation volatility, lagged values, thresholds)
- **Unsupervised anomaly detection** with IsolationForest
- Heuristic **0–10 rating scale** (10=best, 0=worst)
- Interactive **Streamlit dashboard** for multiple countries (default: DEU, GRC, ARG, ESP, ITA)

---

## Indicators Used
- **Debt & Fiscal**: Central government debt (% GDP), budget balance (% GDP)
- **External Sector**: Debt service/exports, current account (% GDP), reserves (months of imports), trade openness
- **Growth & Income**: GDP growth, GDP per capita
- **Monetary**: Inflation (CPI), lending/deposit rates, money supply (M2 % GDP)
- **Labor & Governance**: Unemployment rate, Political stability (WGI)

---

## Installation

Clone this repo and install requirements:

```bash
git clone https://github.com/<kullanıcı-adı>/<repo-ismi>.git
cd <repo-ismi>
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt


- Kurulum: `pip install -r requirements.txt`
- Çalıştır: `python quick_test.py DEU` veya `streamlit run app.py`
