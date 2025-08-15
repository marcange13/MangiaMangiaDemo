# 🍝 Mangia Maniga – Restaurant Assistant – MVP

**Your AI Restaurant Assistant.**

This Streamlit app ranks dishes using real‑time inventory signals, chef queries, and business goals. It also includes planning tools for a quick weekly workflow.

---

## ✨ Features

### Core behaviors
- **Inventory‑aware recommendations** – dishes missing ingredients are penalized.
- **Expiry boosting** – dishes using ingredients close to expiry move up.
- **Surplus boosting** – dishes that help reduce overstock get promoted.
- **Dynamic scoring** – blends `priority_score`, expiry/surplus, margin, and chef‑query match.
- **Chef queries** – type natural phrases (e.g., “seafood special”, “tomato lunch”).

### Operational insights
- **Top Expiring Ingredients**
- **🛑 86 Soon (3–5 days)**
- **High‑Margin Specials**
- **Low Stock Watchlist**
- **Overstock Clearance**

### Planning add‑ons (included)
- **Waste Forecast** (≤7 days)
- **Weekly Specials Planner (prototype)**
- **Prep Impact suggestions**

### Optional integrations
- **OpenAI** for short social captions + server upsell scripts.
- **Tasty (RapidAPI)** for quick recipe/video ideas.

---

## 🚀 Quickstart

```bash
# 1) Python 3.10+ recommended
pip install -r requirements.txt

# 2) (Optional) Put CSVs alongside app.py or in /data
#    - menu_with_viability.csv  (or authentic_menu.csv)
#    - data/authentic_inventory.csv

# 3) Run
streamlit run app.py
```

The app auto‑loads from the repo root or `/data`. You can also upload CSVs from the sidebar.

---

## 🔐 Secrets (optional)

Create `.streamlit/secrets.toml` to enable captions and Tasty suggestions:

```toml
# OpenAI
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL   = "gpt-4o-mini"  # optional

# Tasty via RapidAPI
TASTY_API_KEY  = "your_rapidapi_key"
# RAPIDAPI_KEY = "your_rapidapi_key"  # either key name works
```

> If secrets are not set, those integrations are simply hidden.

---

## 📦 Data expectations

Place CSVs in the repo root or `/data`:

- **Menu**: `menu_with_viability.csv` (or `authentic_menu.csv`)
- **Inventory**: `data/authentic_inventory.csv`

Common columns (auto‑normalized if names vary):

**Menu**
- `dish_name` (aka `dish`/`name`), `ingredients` (comma/semicolon list)
- optional: `priority_score`, `price`, `cost`, `margin_pct` (0–1)

**Inventory**
- `ingredient`, `quantity`, optional: `unit`, `par_level`, `expiry`, `surplus_qty`

---

## ⚙️ Scoring (high level)

Final score ≈ `priority_score`
`+ (w_query × query_match)`
`+ (w_expiry × expiry_urgency)`
`+ (w_surplus × surplus_signal)`
`+ (w_margin × margin_pct)`
`− (w_availability × missing_ingredient_penalty)`

Weights are adjustable from the sidebar.

---

## 🧭 Using the app

1. Enter a **chef query** (e.g., “tomato lunch”).  
2. Tune **weights** in the sidebar.  
3. Review the **Dashboard** table (status, score, margin).  
4. Explore **Operational Insights** and **Planning** sections.  
5. (Optional) Expand **Recipe & video ideas (Tasty)** for inspiration and use OpenAI‑generated captions.

---

## 🗂 Repo layout

```
.
├─ app.py
├─ logic.py
├─ prompts.py
├─ requirements.txt
├─ menu_with_viability.csv            # optional; or use uploads
├─ data/
│  └─ authentic_inventory.csv
└─ .streamlit/
   └─ secrets.toml                    # optional (keys)
```

---

## 🧰 Troubleshooting

- **No results / missing availability**: ensure menu ingredient names match inventory (case‑insensitive exact words).  
- **Margins look off**: use `margin_pct` in **0–1** range (e.g., `0.72`, not `72`).  
- **Dates**: `YYYY‑MM‑DD` format recommended for `expiry`.
