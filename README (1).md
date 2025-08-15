
# Mangia Mangia — Nonna's Kitchen (Streamlit Demo)

A clean, neutral, Italian-home–inspired Streamlit demo that recommends dishes you can make **right now** based on **today’s inventory**, **earliest expiry**, **surplus**, and **margin** — with Nonna cheering you on 👵🍝.

## Features
- Dynamic ranking: priority + margin + urgency (from expiry) + surplus (over par) + availability
- Query boosts: “tomatoes and eggs” nudges matching dishes
- Insights: Top Expiring, 86 Soon, High‑Margin Specials, Low Stock, Overstock + Matches
- Planning: Weekly Specials Plan, Waste Forecast, Batch Prep Suggestions
- Boosters: Server Script, Social Caption, quick wine hints (LLM)
- Optional: Tasty API for recipe/video ideas

## Data
```
data/authentic_menu_priced_2025.csv
data/authentic_inventory.csv
```
Menu headers (synonyms OK): dish_name, ingredients, price, cost, profit_margin_pct, priority_score, urgency_score, surplus_score  
Inventory headers (synonyms OK): ingredient (required), quantity, unit, expiry_date, par_level, incoming_qty

## Secrets (Streamlit Cloud → Settings → Secrets)
```toml
OPENAI_API_KEY = "sk-..."
TASTY_API_KEY  = "xxxxxxxx"
USE_TASTY      = "false"
MODEL          = "gpt-4o-mini"
```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Assets (optional)
Place PNGs in `assets/` named:
```
assets/nonna_hello.png
assets/nonna_happy.png
assets/nonna_alert.png
```
They’ll show automatically in the sidebar.

## Roadmap
- FastAPI wrapper for Bubble UI
- Supplier deliveries calendar integration
- POS sales integration for demand‑aware specials
