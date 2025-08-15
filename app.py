import os
import io
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# Optional: OpenAI + Tasty integrations (will no-op if secrets are missing)
OPENAI_AVAILABLE = False
TASTY_AVAILABLE = False
client = None

def _maybe_init_openai():
    global OPENAI_AVAILABLE, client
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if api_key:
            client = OpenAI(api_key=api_key)
            OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False

def _tasty_headers():
    key = st.secrets.get("TASTY_API_KEY", None) or st.secrets.get("RAPIDAPI_KEY", None)
    if not key:
        return None
    return {
        "x-rapidapi-host": "tasty.p.rapidapi.com",
        "x-rapidapi-key": key,
    }

def tasty_search(query: str, size: int = 3):
    import requests
    headers = _tasty_headers()
    if not headers:
        return []
    try:
        url = "https://tasty.p.rapidapi.com/recipes/list"
        params = {"q": query, "from": 0, "size": size}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("results", [])
            out = []
            for it in items:
                out.append({
                    "name": it.get("name"),
                    "description": it.get("description"),
                    "thumbnail_url": it.get("thumbnail_url"),
                    "video_url": it.get("original_video_url"),
                    "total_time_tier": (it.get("total_time_tier") or {}).get("display_tier")
                })
            return out
    except Exception:
        pass
    return []

def llm_caption_and_script(dish_name: str, wine: str = "") -> Tuple[str, str]:
    \"\"\"Returns (caption, server_script). Fallbacks if OpenAI not configured.\"\"\"
    if not OPENAI_AVAILABLE:
        caption = f\"Today's special: {dish_name}! 🍝 A cozy classic bursting with fresh flavours. #DailySpecial #MangiaMangia\"
        script = f\"Guest loves {dish_name}? Offer to pair it with our house {wine or 'red wine'} and suggest a dessert finish.\"
        return caption, script
    try:
        model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
        prompt = (
            \"You are a restaurant marketer. Write a fun, punchy Instagram caption (≤25 words) for the dish: "
            f"{dish_name}. Include 1-2 food emojis and a gentle call to action.\\n\\n"
            "Also provide a 1-2 sentence server upsell script highlighting a pairing and a premium add-on. "
            "Return JSON with keys 'caption' and 'script'.\""
        )
        # Try Chat Completions API first
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {\"role\": \"system\", \"content\": \"You are helpful and concise.\"},
                    {\"role\": \"user\", \"content\": prompt},
                ],
                temperature=0.7,
            )
            text = resp.choices[0].message.content.strip()
        except Exception:
            # Fallback to Responses API for newer SDKs
            resp = client.responses.create(
                model=model,
                input=[
                    {\"role\": \"system\", \"content\": \"You are helpful and concise.\"},
                    {\"role\": \"user\", \"content\": prompt},
                ],
            )
            text = resp.output_text.strip()
        # Parse JSON if present
        try:
            data = json.loads(text)
            return data.get(\"caption\", \"\"), data.get(\"script\", \"\")
        except Exception:
            # Heuristic split
            parts = text.split(\"\\n\", 1)
            caption = parts[0][:200]
            script = parts[1] if len(parts) > 1 else \"Suggest a wine pairing and a premium add-on.\"
            return caption, script
    except Exception:
        caption = f\"Today's special: {dish_name}! 🍝 A cozy classic bursting with fresh flavours. #DailySpecial #MangiaMangia\"
        script = f\"Offer to pair {dish_name} with a glass of {wine or 'Chianti'}; suggest a dessert to finish.\"
        return caption, script

# -------------------- App Config -------------------- #
st.set_page_config(
    page_title=\"Mangia Mangia – Kitchen Brain (MVP)\",
    page_icon=\"🍝\",
    layout=\"wide\",
)

st.title(\"🍝 Mangia Mangia – Kitchen Brain (MVP)\")
st.caption(\"Inventory-aware specials, waste reduction, and a dash of Nonna.\")

# Nonna assets (will fallback gracefully if missing)
NONNA_ASSETS = {
    \"smile\": \"assets/nonna_smile.gif\",
    \"point\": \"assets/nonna_point.png\",
    \"wave\": \"assets/nonna_wave.gif\",
}

# -------------------- Sidebar Controls -------------------- #
with st.sidebar:
    st.header(\"⚙️ Controls\")
    st.write(\"**Scoring Weights** (tune live)\")
    w_query = st.slider(\"Chef query weight\", 0.0, 3.0, 1.5, 0.1)
    w_urg = st.slider(\"Expiry (urgency) weight\", 0.0, 3.0, 1.2, 0.1)
    w_sur = st.slider(\"Surplus weight\", 0.0, 3.0, 1.0, 0.1)
    w_mrg = st.slider(\"Margin weight\", 0.0, 3.0, 1.0, 0.1)
    w_avl = st.slider(\"Availability penalty\", 0.0, 5.0, 3.0, 0.1)

    st.divider()
    st.subheader(\"📦 Data Sources\")
    st.write(\"The app will auto-load CSVs if present. Otherwise, upload below.\")
    menu_file = st.file_uploader(\"Menu CSV (e.g., authentic_menu.csv or menu_with_viability.csv)\", type=[\"csv\"], key=\"menu\")
    inv_file = st.file_uploader(\"Inventory CSV (e.g., authentic_inventory.csv)\", type=[\"csv\"], key=\"inv\")

    st.caption(\"OpenAI + Tasty keys are read from `st.secrets`.")

# -------------------- Helpers -------------------- #
def _read_csv_auto(fp: io.BytesIO | str) -> pd.DataFrame:
    try:
        return pd.read_csv(fp)
    except Exception:
        try:
            return pd.read_csv(fp, encoding=\"latin-1\")
        except Exception:
            return pd.DataFrame()

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try provided uploads first
    if menu_file is not None:
        menu = _read_csv_auto(menu_file)
    else:
        for p in [\"menu_with_viability.csv\", \"authentic_menu.csv\", \"data/menu_with_viability.csv\", \"data/authentic_menu.csv\"]:
            if os.path.exists(p):
                menu = _read_csv_auto(p)
                break
        else:
            menu = pd.DataFrame()

    if inv_file is not None:
        inv = _read_csv_auto(inv_file)
    else:
        for p in [\"authentic_inventory.csv\", \"data/authentic_inventory.csv\"]:
            if os.path.exists(p):
                inv = _read_csv_auto(p)
                break
        else:
            inv = pd.DataFrame()

    return menu, inv

def _now():
    # Use local server time
    return datetime.now()

def parse_date(s):
    if pd.isna(s):
        return None
    try:
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        try:
            return datetime.strptime(str(s), \"%Y-%m-%d\")
        except Exception:
            return None

def ensure_menu_schema(menu: pd.DataFrame) -> pd.DataFrame:
    # Expected columns: dish_id, dish_name, ingredients, price, cost, margin_pct, priority_score
    out = menu.copy()
    cols = {c.lower(): c for c in out.columns}
    # Normalize potential variants
    if \"dish\" in cols and \"dish_name\" not in cols:
        out.rename(columns={cols[\"dish\"]: \"dish_name\"}, inplace=True)
    if \"name\" in cols and \"dish_name\" not in out.columns:
        out.rename(columns={cols[\"name\"]: \"dish_name\"}, inplace=True)
    if \"ingredients\" not in out.columns and \"ingredient_list\" in cols:
        out.rename(columns={cols[\"ingredient_list\"]: \"ingredients\"}, inplace=True)

    if \"ingredients\" not in out.columns:
        out[\"ingredients\"] = \"\"
    if \"price\" not in out.columns:
        out[\"price\"] = np.nan
    if \"cost\" not in out.columns:
        out[\"cost\"] = np.nan
    if \"margin_pct\" not in out.columns:
        # compute if cost/price present
        with np.errstate(divide='ignore', invalid='ignore'):
            out[\"margin_pct\"] = np.where(out[\"price\"].gt(0) & out[\"cost\"].notna(),
                                          (out[\"price\"] - out[\"cost\"]) / out[\"price\"],
                                          np.nan)
    if \"priority_score\" not in out.columns:
        out[\"priority_score\"] = 0.5
    if \"dish_id\" not in out.columns:
        out[\"dish_id\"] = np.arange(1, len(out) + 1)
    return out

def ensure_inventory_schema(inv: pd.DataFrame) -> pd.DataFrame:
    # Expected: ingredient, quantity, unit, par_level, expiry_date, surplus_qty, cost_per_unit
    out = inv.copy()
    # Normalize likely variants
    rename_map = {}
    for c in list(out.columns):
        lc = c.lower().strip()
        if lc in {\"item\", \"name\"}:
            rename_map[c] = \"ingredient\"
        elif lc in {\"qty\", \"stock\", \"on_hand\"}:
            rename_map[c] = \"quantity\"
        elif lc in {\"par\", \"parlevel\"}:
            rename_map[c] = \"par_level\"
        elif lc in {\"expiry\", \"expires\", \"best_before\"}:
            rename_map[c] = \"expiry_date\"
        elif lc in {\"surplus\", \"overstock\", \"excess_qty\"}:
            rename_map[c] = \"surplus_qty\"
        elif lc in {\"unit_cost\", \"cost\"}:
            rename_map[c] = \"cost_per_unit\"
    if rename_map:
        out.rename(columns=rename_map, inplace=True)

    for col, default in [
        (\"ingredient\", \"\"), (\"quantity\", 0.0), (\"unit\", \"\"), (\"par_level\", 0.0),
        (\"expiry_date\", None), (\"surplus_qty\", 0.0), (\"cost_per_unit\", np.nan)
    ]:
        if col not in out.columns:
            out[col] = default

    # Parse dates + helper features
    out[\"expiry_dt\"] = out[\"expiry_date\"].apply(parse_date)
    now = _now().date()
    out[\"days_to_expiry\"] = out[\"expiry_dt\"].apply(lambda d: (d.date() - now).days if d else 9999)
    out[\"below_par\"] = out[\"quantity\"] < out[\"par_level\"].clip(lower=0)
    out[\"overstock\"] = out[\"surplus_qty\"].fillna(0) > 0
    out[\"over_par2x\"] = out[\"quantity\"] > (out[\"par_level\"].fillna(0) * 2)
    out[\"is_surplus\"] = out[[\"overstock\", \"over_par2x\"]].any(axis=1)
    return out

def ingredient_tokens(s: str) -> List[str]:
    return [t.strip().lower() for t in (s or \"\").replace(\";\", \",\").split(\",\") if t.strip()]

def has_all_ingredients(ing_list: List[str], inv: pd.DataFrame) -> bool:
    if not ing_list:
        return False
    inv_names = set(inv[\"ingredient\"].str.lower().str.strip())
    return all(ing.lower().strip() in inv_names for ing in ing_list)

def min_days_to_expiry_for_dish(ing_list: List[str], inv: pd.DataFrame) -> int:
    if not ing_list or inv.empty:
        return 9999
    df = inv.copy()
    df[\"_norm\"] = df[\"ingredient\"].str.lower().str.strip()
    days = []
    for ing in ing_list:
        row = df.loc[df[\"_norm\"] == ing.lower().strip()]
        if not row.empty:
            days.append(int(row[\"days_to_expiry\"].iloc[0]))
        else:
            days.append(9999)
    return min(days) if days else 9999

def surplus_ratio_for_dish(ing_list: List[str], inv: pd.DataFrame) -> float:
    if not ing_list or inv.empty:
        return 0.0
    df = inv.copy()
    df[\"_norm\"] = df[\"ingredient\"].str.lower().str.strip()
    flags = []
    for ing in ing_list:
        row = df.loc[df[\"_norm\"] == ing.lower().strip()]
        if not row.empty:
            flags.append(bool(row[[\"is_surplus\"]].iloc[0].values[0]))
        else:
            flags.append(False)
    return float(sum(flags)) / max(1, len(flags))

def query_match_score(text: str, dish_name: str, ingredients: str) -> float:
    if not text:
        return 0.0
    q = text.lower().strip()
    target = f\"{dish_name} {ingredients}\".lower()
    # Use fuzzy if available; otherwise, simple token overlap
    if fuzz:
        return fuzz.token_set_ratio(q, target) / 100.0
    tokens_q = set(q.split())
    tokens_t = set(target.split())
    return len(tokens_q & tokens_t) / max(1, len(tokens_q))

def margin_norm(margin_pct: float) -> float:
    if pd.isna(margin_pct):
        return 0.5
    return float(np.clip(margin_pct, 0.0, 1.0))

def dynamic_score(row, inv: pd.DataFrame, chef_query: str) -> Tuple[float, Dict[str, float]]:
    ing_list = ingredient_tokens(row.get(\"ingredients\", \"\"))
    available = has_all_ingredients(ing_list, inv)
    # Core features
    urg_days = min_days_to_expiry_for_dish(ing_list, inv)
    urg = 0.0 if urg_days >= 10 else max(0.0, (10 - urg_days) / 10.0)  # closer expiry => higher
    sur = surplus_ratio_for_dish(ing_list, inv)
    mrg = margin_norm(row.get(\"margin_pct\", np.nan))
    qry = query_match_score(chef_query, row.get(\"dish_name\", \"\"), row.get(\"ingredients\", \"\"))

    base = float(row.get(\"priority_score\", 0.5))
    score = (base
             + w_query * qry
             + w_urg * urg
             + w_sur * sur
             + w_mrg * mrg)
    if not available:
        score -= w_avl  # penalize heavily if missing any ingredient

    details = {\"base\": base, \"query\": qry, \"urgency\": urg, \"surplus\": sur, \"margin\": mrg, \"available\": float(available)}
    return float(score), details

def recommend(menu: pd.DataFrame, inv: pd.DataFrame, chef_query: str, top_n: int = 12) -> pd.DataFrame:
    rows = []
    for _, r in menu.iterrows():
        sc, d = dynamic_score(r, inv, chef_query)
        rows.append({\"dish_id\": r[\"dish_id\"], \"dish_name\": r[\"dish_name\"], \"score\": sc, **d,
                     \"ingredients\": r.get(\"ingredients\", \"\"), \"price\": r.get(\"price\", np.nan), \"margin_pct\": r.get(\"margin_pct\", np.nan)})
    df = pd.DataFrame(rows)
    df.sort_values(\"score\", ascending=False, inplace=True)
    return df.head(top_n)

def dishes_using(ingredients: List[str], menu: pd.DataFrame) -> List[str]:
    out = []
    names = set(i.lower().strip() for i in ingredients)
    for _, r in menu.iterrows():
        ing_list = ingredient_tokens(r.get(\"ingredients\", \"\"))
        if any(i in names for i in ing_list):
            out.append(r[\"dish_name\"])
    return out

def weekly_planner(recos: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    # Simple round-robin from top recos with diminishing returns
    items = recos[\"dish_name\"].tolist()
    result = []
    for i in range(days):
        if not items:
            break
        pick = items[i % len(items)]
        result.append({\"day\": (i + 1), \"dish\": pick})
    return pd.DataFrame(result)

# -------------------- Load Data -------------------- #
menu, inv = load_data()
if menu.empty or inv.empty:
    st.warning(\"Upload or place your menu & inventory CSVs to unlock full functionality.\")
else:
    menu = ensure_menu_schema(menu)
    inv = ensure_inventory_schema(inv)

# Initialize OpenAI state
_maybe_init_openai()

# -------------------- Nonna Mood -------------------- #
def infer_nonna_mood(inv: pd.DataFrame) -> Tuple[str, str]:
    if inv.empty:
        return \"smile\", \"Benvenuti! Upload your data and Nonna will help. 😄\"
    soon_expiring = (inv[\"days_to_expiry\"] <= 3).sum()
    overstock = inv[\"is_surplus\"].sum()
    if soon_expiring >= 5:
        return \"point\", \"Mamma mia! These are expiring soon—use them first!\"
    if overstock >= 5:
        return \"wave\", \"We have plenty! Mangia! Mangia!\"
    return \"smile\", \"Stock looks good—let's cook something delicious!\"

mood, mood_text = infer_nonna_mood(inv if not inv.empty else pd.DataFrame())
nonna_asset = NONNA_ASSETS.get(mood, NONNA_ASSETS[\"smile\"])
col_non, col_hdr = st.columns([1, 5])
with col_non:
    if os.path.exists(nonna_asset):
        st.image(nonna_asset, use_container_width=True)
    else:
        st.markdown(\"### 👵 Nonna\")
with col_hdr:
    st.subheader(mood_text)

# -------------------- Chef Query + Recommendations -------------------- #
st.markdown(\"---\")
chef_query = st.text_input(\"👨‍🍳 Chef query (e.g., 'seafood special', 'tomato lunch', 'gluten-free'):\", value=\"\")

if menu.empty or inv.empty:
    st.info(\"Waiting for data… See sidebar to upload CSVs.\")
else:
    recos = recommend(menu, inv, chef_query, top_n=20)

    # Real-Time Kitchen Dashboard
    st.subheader(\"🧭 Real-Time Kitchen Dashboard\")
    def dish_status(row) -> str:
        if not row.get(\"available\"):
            return \"❌ Not Available\"
        if row.get(\"urgency\", 0.0) >= 0.5:
            return \"⚠️ Use Soon\"
        return \"✅ Ready\"

    recos[\"status\"] = recos.apply(dish_status, axis=1)
    st.dataframe(recos[[\"dish_name\", \"score\", \"status\", \"margin\":=recos[\"margin_pct\"].round(2)]].rename(columns={\"margin\": \"margin_pct\"}), use_container_width=True)

    # Select a dish for details
    picked = st.selectbox(\"Focus dish:\", options=recos[\"dish_name\"].tolist())
    picked_row = recos.loc[recos[\"dish_name\"] == picked].iloc[0]
    st.markdown(f\"**Why this ranks well:**  \") 
    st.write({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in picked_row.to_dict().items() if k in [\"query\", \"urgency\", \"surplus\", \"margin\", \"available\"]})

    # Social + server script
    caption, upsell = llm_caption_and_script(picked)
    st.markdown(\"**📸 Social caption**\") 
    st.write(caption)
    st.markdown(\"**🗣 Server 'sales script'**\")
    st.write(upsell)

    # Tasty suggestions
    with st.expander(\"🍳 Recipe & video ideas (Tasty)\"):
        sugg = tasty_search(picked) if _tasty_headers() else []
        if not sugg:
            st.caption(\"Connect Tasty RapidAPI key in secrets to enable.\")
        else:
            for s in sugg:
                st.markdown(f\"- **{s.get('name','')}** — {s.get('total_time_tier','')} \"
                            f\"{('[video]' if s.get('video_url') else '')}\")
                if s.get(\"thumbnail_url\"): st.image(s[\"thumbnail_url\"], width=160)

    st.markdown(\"---\")
    # -------------------- Operational Insights -------------------- #
    st.subheader(\"📊 Operational Insights\")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(\"**Top 5 Expiring Ingredients**\") 
        soon = inv.sort_values(\"days_to_expiry\").head(5)[[\"ingredient\", \"days_to_expiry\", \"quantity\", \"unit\"]]
        st.dataframe(soon, use_container_width=True)

        st.markdown(\"**Low Stock Watchlist**\")
        low = inv.loc[inv[\"below_par\"]].sort_values(\"par_level\", ascending=False)[[\"ingredient\", \"quantity\", \"par_level\"]].head(10)
        st.dataframe(low, use_container_width=True)

    with col2:
        st.markdown(\"**🛑 86 Soon (3–5 days)**\")
        # Dishes with min days to expiry <= 5
        rows = []
        for _, r in menu.iterrows():
            ing = ingredient_tokens(r.get(\"ingredients\", \"\"))
            days = min_days_to_expiry_for_dish(ing, inv)
            if days <= 5:
                rows.append({\"dish\": r[\"dish_name\"], \"min_days\": days})
        soon_out = pd.DataFrame(rows).sort_values(\"min_days\")
        st.dataframe(soon_out.head(15), use_container_width=True)

        st.markdown(\"**Overstock Clearance**\")
        over = inv.loc[inv[\"is_surplus\"]][[\"ingredient\", \"quantity\", \"par_level\"]].head(10)
        st.dataframe(over, use_container_width=True)

    with col3:
        st.markdown(\"**High-Margin Specials**\")
        avail = recos.loc[recos[\"status\"] != \"❌ Not Available\"]
        hm = avail.sort_values(\"margin\", ascending=False)[[\"dish_name\", \"margin\"]].head(10).rename(columns={\"margin\": \"margin_pct\"})
        st.dataframe(hm, use_container_width=True)

        st.markdown(\"**Waste Forecast (≤7 days)**\")
        waste = inv.loc[(inv[\"days_to_expiry\"] <= 7) & (inv[\"quantity\"] > 0)].copy()
        waste[\"suggested_dishes\"] = waste[\"ingredient\"].apply(lambda x: \", \".join(dishes_using([x], menu))[:120])
        st.dataframe(waste[[\"ingredient\", \"days_to_expiry\", \"suggested_dishes\"]].head(10), use_container_width=True)

    st.markdown(\"---\")
    # -------------------- Menu Planning Tools -------------------- #
    st.subheader(\"📅 Menu Planning Tools\")
    with st.expander(\"Weekly Specials Planner (prototype)\", expanded=True):
        plan = weekly_planner(recos, days=7)
        st.dataframe(plan, use_container_width=True)

    with st.expander(\"Prep Impact suggestions\", expanded=False):
        # naive shared-ingredient grouping
        rec_short = recos.head(12).merge(menu[[\"dish_name\", \"ingredients\"]], on=\"dish_name\", how=\"left\")
        pairs = []
        for i in range(len(rec_short)):
            for j in range(i + 1, len(rec_short)):
                a = set(ingredient_tokens(rec_short.iloc[i][\"ingredients\"])) 
                b = set(ingredient_tokens(rec_short.iloc[j][\"ingredients\"])) 
                overlap = a & b
                if len(overlap) >= 2:
                    pairs.append({
                        \"dish_A\": rec_short.iloc[i][\"dish_name\"],
                        \"dish_B\": rec_short.iloc[j][\"dish_name\"],
                        \"shared_preps\": \", \".join(sorted(list(overlap)))[:100]
                    })
        pairs_df = pd.DataFrame(pairs)
        if pairs_df.empty:
            st.caption(\"No obvious shared prep across top picks. Try adjusting weights or query.\")
        else:
            st.dataframe(pairs_df, use_container_width=True)

st.markdown(\"\\n---\\nMade with ❤️ by Mangia Mangia MVP.\")
