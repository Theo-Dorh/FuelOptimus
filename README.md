# FuelOptimus — Pricing Decision Support for Downstream Petroleum Retail

A machine learning-based fuel price optimisation framework for oil marketing companies (OMCs) operating in Ghana's deregulated downstream petroleum retail sector.

## What It Does

FuelOptimus takes a station, product, NPA pricing window, and local competitor price as inputs and returns the **margin-maximising price premium** above the NPA regulatory floor — constrained to stay at or below the local competitive average.

The tool answers the question every OMC pricing manager faces every two weeks: *how much above the NPA floor should we charge at this station?*

## How It Works

**Stage 1 — Predict:** Three ML models (Ridge Regression, Random Forest, XGBoost) are trained on 39,312 daily station-level observations to predict sales volume as a function of the price premium and 26 engineered features. The best model is selected automatically by test-set R².

**Stage 2 — Optimise:** For 201 candidate premiums (GHS 0.00 to 2.00 in steps of 0.01), the engine predicts volume at each price point and selects the premium where margin (premium × volume) is highest, subject to the competition constraint that the pump price stays at or below the local competitor mean.

## Key Findings

| Metric | Value |
|--------|-------|
| Best model | Ridge Regression (R² = 0.865) |
| PMS underpriced | 88% of observations below competitive average |
| AGO underpriced | 54% of observations below competitive average |
| Estimated improvement | GHS 2,060 / station / day |
| Network-wide | ≈ GHS 2.2 million / month (36 stations) |
| DiD: uniform pricing cost | GHS −2,039 / station / day (p < 0.001) |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10+ and `ICON_MAT_Clean.csv` in the same directory.

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit application — trains model at startup, serves recommendations |
| `requirements.txt` | Python dependencies |
| `ICON_MAT_Clean.csv` | Master Analytical Table (39,312 rows × 26 columns) |

## Academic Context

This tool is the software artefact for:

> **A Machine Learning-Based Fuel Price Optimisation Framework for the Downstream Petroleum Retail Sector**
>
> Theophilus Dorh (22425676), supervised by Prof. Solomon Mensah
>
> MSc Computer Science, University of Ghana, Legon — September 2026

## Tech Stack

- Python 3.12
- Streamlit
- scikit-learn (Ridge, Random Forest)
- XGBoost
- pandas, NumPy
- Plotly (charts)

## Licence

This repository is submitted as part of an academic project. The data is anonymised and used with permission. Contact theodorh123@gmail.com for enquiries.
