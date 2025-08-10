# 🍝 Mangia Mangia – Full Streamlit App (Phase 5)
# Save this file as: app.py
# Run locally: pip install streamlit pandas numpy
#              streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import ast
from datetime import datetime

st.set_page_config(page_title="Mangia Mangia AI", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def parse_list_maybe(val):
    """Safely parse list-like strings from CSV into Python lists."""
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    s = str(val).strip()
    try:
        # parse JSON-like Python list strings
        return ast.literal_eval(s) if (s.startswith("[") and s.endswith("]")) else [x.strip() for x in s.split(",")]
    except Exception:
        return [x.strip() for x in s.split(",")]

def ensure_columns(df, columns_with_defaults):
    for col, default in columns_with_defaults.items():
        if col not in df.columns:
            df[col] = default(df) if callable(default) else default
    return df

def calc_priority_score(df):
    return df["urgency_score"].fillna(0)*1.5 + df["surplus_score"].fillna(0)*1.0

def viability_fallback(df):
    pm_pct = df.get("profit_margin", pd.Series(np.nan, index=df.index)) / df.get("price_usd", pd.Series(1.0, index=df.index))
    pm_pct = pm_pct.fillna(pm_pct.median() if not pm_pct.dropna().empty else 0.4)
    return (df["urgency_score"].fillna(0)*0.5 + df["surplus_score"].fillna(0)*0.4 + pm_pct*0.1)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ----------------------------
# Sidebar – Data sources & Uploads
# ----------------------------
st.sidebar.title("⚙️ Data & Controls")

with st.sidebar.expander("📥 Load / Upload Data", expanded=True):
    uploaded_menu = st.file_uploader("Upload menu_with_viability.csv", type=["csv"])
    uploaded_inventory = st.file_uploader("Upload authentic_inventory.csv", type=["csv"])
    use_sample = st.checkbox("Use local files if uploads are empty", value=True)

# Load menu
if uploaded_menu is not None:
    menu_df = pd.read_csv(uploaded_menu)
elif use_sample:
    try:
        menu_df = pd.read_csv("menu_with_viability.csv")
    except Exception:
        st.error("menu_with_viability.csv not found. Upload it in the sidebar.")
        st.stop()
else:
    st.error("Please upload menu_with_viability.csv")
    st.stop()

# Load inventory
if uploaded_inventory is not None:
    inventory_df = pd.read_csv(uploaded_inventory)
elif use_sample:
    try:
        inventory_df = pd.read_csv("authentic_inventory.csv")
    except Exception:
        st.warning("authentic_inventory.csv not found. You can still browse recommendations.")
        inventory_df = pd.DataFrame({"ingredient": []})
else:
    inventory_df = pd.DataFrame({"ingredient": []})

# Normalize expected columns
menu_df = ensure_columns(menu_df, {
    "dish_name": "",
    "ingredients": [],
    "price_usd": lambda df: pd.Series([np.nan]*len(df)),
    "urgency_score": 0.0,
    "surplus_score": 0.0,
    "profit_margin": lambda df: pd.Series([np.nan]*len(df)),
    "pairings": [],
    "suggested_wine": "",
})
# Parse list columns
menu_df["ingredients"] = menu_df["ingredients"].apply(parse_list_maybe)
if "pairings" in menu_df.columns:
    menu_df["pairings"] = menu_df["pairings"].apply(parse_list_maybe)

# Ensure numeric types
for c in ["price_usd", "urgency_score", "surplus_score", "profit_margin"]:
    menu_df[c] = menu_df[c].apply(safe_float)

# Derive priority & viability if missing
if "priority_score" not in menu_df.columns:
    menu_df["priority_score"] = calc_priority_score(menu_df)
if "viability_score" not in menu_df.columns:
    menu_df["viability_score"] = viability_fallback(menu_df)

# Inventory normalize
if "ingredient" in inventory_df.columns:
    inv_col = "ingredient"
else:
    for col in ["ingredient_name", "item_name", "name"]:
        if col in inventory_df.columns:
            inv_col = col
            break
    else:
        inv_col = None

available_ingredients = set()
if inv_col is not None:
    inventory_df[inv_col] = inventory_df[inv_col].astype(str).str.lower()
    available_ingredients = set(inventory_df[inv_col].tolist())

# ----------------------------
# App Title
# ----------------------------
st.title("👨‍🍳 Mangia Mangia: Daily Special Recommender")
st.caption("Reduce food waste, optimize inventory, and pick profitable specials with AI.")

# ----------------------------
# Sidebar – Filters
# ----------------------------
st.sidebar.subheader("🔎 Filters")
n_recs = st.sidebar.slider("Number of top dishes", 1, 20, 10)
max_price = st.sidebar.number_input(
    "Max price ($)", min_value=0.0,
    value=float(np.nanmax(menu_df["price_usd"]) if not np.isnan(menu_df["price_usd"]).all() else 0.0)
)
min_viability = st.sidebar.slider(
    "Min viability score", 0.0, float(max(100.0, np.nanmax(menu_df["viability_score"]) if not np.isnan(menu_df["viability_score"]).all() else 10.0)), 0.0
)
show_pairings = st.sidebar.checkbox("Show pairings & wine", value=True)

# Scenario toggles
st.sidebar.subheader("🧪 Scenarios")
scenario = st.sidebar.selectbox("Scenario", ["Default", "Low Budget", "Surplus Emergency", "High Urgency Priority"])

def apply_scenario(df, scenario_name):
    df = df.copy()
    if scenario_name == "Low Budget":
        inv = df["price_usd"].replace(0, np.nan)
        df["scenario_score"] = (df["viability_score"]*0.5 + (1.0 / inv) * 10.0).fillna(df["viability_score"])
    elif scenario_name == "Surplus Emergency":
        df["scenario_score"] = df["viability_score"] + df["surplus_score"]*2.0
    elif scenario_name == "High Urgency Priority":
        df["scenario_score"] = df["viability_score"] + df["urgency_score"]*1.5
    else:
        df["scenario_score"] = df["viability_score"]
    return df

menu_df = apply_scenario(menu_df, scenario)

# ----------------------------
# Tabs
# ----------------------------
tab_dash, tab_recs, tab_inventory, tab_pair, tab_ideas, tab_chat, tab_admin = st.tabs(
    ["📊 Dashboard", "🍝 Recommendations", "📦 Inventory Match", "🍷 Pairings", "🧠 New Ideas", "💬 Chatbot", "🛠 Admin/Export"]
)

# ----------------------------
# Dashboard
# ----------------------------
with tab_dash:
    st.title("📊 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Dishes", len(menu_df))
    with c2:
        st.metric("Median Price", f"${np.nanmedian(menu_df['price_usd']):.2f}" if not np.isnan(menu_df['price_usd']).all() else "N/A")
    with c3:
        st.metric("Avg Viability", f"{np.nanmean(menu_df['viability_score']):.2f}")
    with c4:
        st.metric("Inventory Items", len(available_ingredients))

    st.subheader("Top Viability (Scenario-adjusted)")
    show_df = menu_df.sort_values("scenario_score", ascending=False).head(n_recs)
    st.dataframe(show_df[["dish_name","price_usd","viability_score","priority_score","urgency_score","surplus_score","scenario_score"]])

    st.subheader("Viability Distribution")
    st.bar_chart(menu_df["viability_score"])

# ----------------------------
# Recommendations
# ----------------------------
with tab_recs:
    st.title("🍝 Recommendations")
    df = menu_df.copy()
    if max_price > 0:
        df = df[df["price_usd"] <= max_price]
    df = df[df["viability_score"] >= min_viability]

    top = df.sort_values("scenario_score", ascending=False).head(n_recs)

    if show_pairings:
        for _, row in top.iterrows():
            st.markdown(f"**{row['dish_name']}**")
            st.write(f"• Viability: {row['viability_score']:.2f}  |  Scenario Score: {row['scenario_score']:.2f}")
            if not np.isnan(row["price_usd"]): st.write(f"• Price: ${row['price_usd']:.2f}")
            if row.get("suggested_wine"): st.write(f"• Wine: {row['suggested_wine']}")
            if isinstance(row.get("pairings"), list) and row["pairings"]:
                st.write(f"• Pairings: {', '.join(row['pairings'])}")
            st.divider()
    else:
        cols = [c for c in ["dish_name","viability_score","scenario_score","price_usd","suggested_wine"] if c in top.columns]
        st.dataframe(top[cols])

    st.caption("Scenario logic: Default, Low Budget, Surplus Emergency, High Urgency Priority.")

# ----------------------------
# Inventory Match
# ----------------------------
with tab_inventory:
    st.title("📦 Inventory Match")
    if available_ingredients:
        can_make = []
        missing = []
        for ings in menu_df["ingredients"]:
            ings_lower = [i.lower() for i in ings]
            ok = all(i in available_ingredients for i in ings_lower)
            can_make.append(ok)
            missing.append([i for i in ings_lower if i not in available_ingredients])
        menu_df["can_make_now"] = can_make
        menu_df["missing_ingredients"] = missing

        st.subheader("Make Right Now")
        ready = menu_df[menu_df["can_make_now"]].sort_values("scenario_score", ascending=False).head(n_recs)
        st.dataframe(ready[["dish_name","price_usd","viability_score","scenario_score"]])

        st.subheader("Closest (Minimal Missing)")
        almost = menu_df[~menu_df["can_make_now"]].copy()
        almost["missing_count"] = almost["missing_ingredients"].apply(len)
        almost = almost.sort_values(["missing_count","scenario_score"], ascending=[True, False]).head(n_recs)
        st.dataframe(almost[["dish_name","missing_ingredients","price_usd","viability_score","scenario_score"]])
    else:
        st.info("No inventory loaded. Upload authentic_inventory.csv on the sidebar to see matches.")

# ----------------------------
# Pairings (Food + Wine)
# ----------------------------
with tab_pair:
    st.title("🍷 Pairings")
    st.write("Pairings and wine suggestions imported from Phase 4.2 results (heuristic/NLP-lite).")
    cols = [c for c in ["dish_name","ingredients","pairings","suggested_wine"] if c in menu_df.columns]
    st.dataframe(menu_df[cols].head(50))

# ----------------------------
# New Ideas (Tasty mock)
# ----------------------------
with tab_ideas:
    st.title("🧠 New Dish Ideas (Mock Tasty Integration)")
    st.caption("Offline mock to simulate Tasty suggestions by seed ingredient.")

    mock_tasty_recipes = {
        "spaghetti": [
            "Spaghetti Carbonara by Tasty",
            "One-Pot Spaghetti w/ Tomato & Basil",
            "Spaghetti Aglio e Olio"
        ],
        "gnocchi": [
            "Baked Gnocchi Alfredo",
            "Gnocchi with Pesto & Sun-dried Tomatoes"
        ],
        "pizza": [
            "Stuffed Crust Margherita Pizza",
            "Neapolitan Pizza w/ Buffalo Mozzarella"
        ],
        "risotto": [
            "Creamy Mushroom Risotto",
            "Saffron Risotto with Parmesan"
        ],
        "lasagna": [
            "Classic Beef Lasagna",
            "Spinach Lasagna Roll-ups"
        ],
    }

    seed = st.selectbox("Pick a seed ingredient", list(mock_tasty_recipes.keys()))
    ideas = mock_tasty_recipes.get(seed, ["No recipes found."])
    st.write("**Suggested recipes:**")
    for i in ideas:
        st.write("• " + i)

# ----------------------------
# Chatbot (FAQ-style)
# ----------------------------
with tab_chat:
    st.title("💬 Assistant")
    st.caption("Lightweight FAQ logic; can be swapped for OpenAI later.")

    def handle_query(q):
        ql = q.lower()
        if "what can i cook" in ql or "suggest" in ql:
            t = menu_df.sort_values("scenario_score", ascending=False).head(5)["dish_name"].tolist()
            return "🍽️ Try: " + ", ".join(t)
        if "viability" in ql:
            return "📈 Viability combines urgency, surplus, and profit margin."
        if "waste" in ql:
            return "♻️ Cook high-urgency & high-surplus items first to cut waste."
        if "pairing" in ql:
            return "🍷 Pairings use heuristic matching from Phase 4.2."
        if "special" in ql or "daily" in ql:
            t = menu_df.sort_values("scenario_score", ascending=False).head(1)["dish_name"].tolist()
            return f"⭐ Today's special: {t[0]}" if t else "No special available."
        return "🤖 Ask about: what to cook, viability, waste reduction, pairing, or specials."

    user_q = st.text_input("Ask the assistant")
    if user_q:
        st.success(handle_query(user_q))

# ----------------------------
# Admin / Export
# ----------------------------
with tab_admin:
    st.title("🛠 Admin / Export")
    st.write("Download scenario-adjusted recommendations for the day.")

    export_cols = [c for c in ["dish_name","ingredients","price_usd","urgency_score","surplus_score","profit_margin","viability_score","priority_score","scenario_score","suggested_wine","pairings"] if c in menu_df.columns]
    export_df = menu_df.sort_values("scenario_score", ascending=False)[export_cols].head(50).copy()
    export_df["generated_at"] = datetime.utcnow().isoformat()

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Today's Specials (CSV)", data=csv_bytes, file_name="mangia_mangia_todays_specials.csv", mime="text/csv")

    st.subheader("Raw Data Preview")
    st.dataframe(menu_df.head(50))
