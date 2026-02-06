# ML Project

## Deployed Webpage
https://monitor-price-estimator-git-clngo.streamlit.app

## Creating a new environment
Recommended (venv):
```bash
python3.8 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Conda (optional):
```bash
conda create -p venv python==3.8 -y
conda activate ./venv
pip install -r requirements.txt
```

## Quick run
```bash
python -m src.pipeline.train_pipeline
```

## Data collection (repeatable)
Environment variable:
- `EBAY_QUERY` (default: `gaming monitor`) — search query string and folder name under `notebook/data/ebay/`.

1) Fetch listings (Browse API):
```bash
python -m src.components.product_ebay
```
This saves `notebook/data/ebay/<dataset>/listings_<timestamp>.json`.

2) Parse HTML cache into `monitors.json` (from your own HTML downloads):
```bash
python -m src.components.html_offline
```

3) Combine specs + listings and create train/test:
```bash
python -m src.components.data_ingestion
```

## Streamlit demo (local)
```bash
streamlit run streamlit_app.py
```

## Docker demo
```bash
docker build -t monitor-price-estimator .
docker run -p 8501:8501 monitor-price-estimator
```
Then open `http://localhost:8501` in your browser.

If you update the model artifacts or code, rebuild the image:
```bash
docker build -t monitor-price-estimator .
```

## Deploy (owner-only)
This section is intended for the project owner to publish the public demo URL.
Others can still run the app locally via Docker or Streamlit.

Streamlit Community Cloud steps:
1) Push to a public GitHub repo.
2) Create a new app in Streamlit Cloud and set entrypoint to `streamlit_app.py`.
3) Add any secrets in the Streamlit Cloud settings (see `secrets.toml` note below).

`secrets.toml` is a Streamlit Cloud config file for sensitive values (API keys). It
is **not** committed to git; Streamlit stores it securely.

## Project Overview

**Problem:**  
Online sellers often misprice products on eBay due to lack of visibility into marketplace listings. Mispricing leads to lost revenue, unsold inventory, and missed opportunities.

**Solution:**  
This MVP provides a **pricing intelligence tool** that:

- Suggests a **recommended price** for a product
- Shows the **price percentile** relative to similar listings
- Flags **pricing risk** (overpriced or underpriced)

**Scope for MVP:**

- Focus on **one product category** (e.g., gaming monitors)
- Use **active listing data only** via the eBay Browse API
- Expose results via a **minimal web interface** or CLI
- Sell-through prediction is **planned for future work**

---

## Technical Stack

- **Language:** Python  
- **APIs:** eBay Browse API  
- **Data Handling:** pandas / JSON  
- **Machine Learning:** scikit-learn (Linear Regression, optional Random Forest)  
- **Interface (optional MVP):** minimal webpage (Streamlit / Flask / FastAPI)
- **Environment Variables:** `.env` for `EBAY_OAUTH_TOKEN`  
- **Logging & Exceptions:** Custom logging

---

## Authentication

1. Use your **AppID + CertID** to generate an OAuth token (Client Credentials flow)  
2. Store the token in `.env`:
```bash
EBAY_OAUTH_TOKEN=v1.1.xxxxxxxxxxxxxxxxx
```
