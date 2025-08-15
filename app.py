import os
import io
from typing import List

import pandas as pd
import numpy as np
import streamlit as st

from logic import (
    load_data,
    normalize_menu,
    normalize_inventory,
    compute_scores,
    tokenize_ingredients,
    insight_top_expiring,
    insight_dishes_86_soon,
    insight_high_margin,
    insight_low_stock,
    insight_overstock,
)
from prompts import caption_and_script, tasty_search

# Rebranded app config
st.set_page_config(page_title="Mangia Maniga – Restaurant Assistant – MVP", page_icon="🍝", layout="wide")
st.title("🍝 Mangia Maniga – Restaurant Assistant – MVP")
st.caption("Your AI Restaurant Assistant.")

# -------------------- Sidebar -------------------- #
with st.sidebar:
    st.header("⚙️ Weights")
    w_query = st.slider("Chef query weight", 0.0, 3.0, 1.5, 0.1)
    w_expiry = st.slider("Expiry (urgency) weight", 0.0, 3.0, 1.2, 0.1)
    w_surplus = st.slider("Surplus weight", 0.0, 3.0, 1.0, 0.1)
    w_margin = st.slider("Margin weight", 0.0, 3.0, 1.0, 0.1)
    w_avail_penalty = st.slider("Availability penalty", 0.0, 5.0, 3.0, 0.1)

    st.divider()
    st.subheader("📦 Data Sources")
    st.write("Auto-loads from repo root or /data. You can also upload:")
    menu_up = st.file_uploader("Menu CSV", type=["csv"], key="menu_up")
    inv_up = st.file_uploader("Inventory CSV", type=["csv"], key="inv_up")

def _read_csv(fp: io.BytesIO | str) -> pd.DataFrame:
    try:
        return pd.read_csv(fp)
    except Exception:
        try:
            return pd.read_csv(fp, encoding="latin-1")
        except Exception:
            return pd.DataFrame()

# -------------------- Load Data -------------------- #
menu_df, inv_df = load_data()
if menu_up is not None:
    menu_df = _read_csv(menu_up)
if inv_up is not None:
    inv_df = _read_csv(inv_up)

menu_df = normalize_menu(menu_df) if not menu_df.empty else pd.DataFrame()
inv_df = normalize_inventory(inv_df) if not inv_df.empty else pd.DataFrame()

# -------------------- Main UI -------------------- #
chef_query = st.text_input("👨‍🍳 Chef query (e.g., 'seafood special', 'tomato lunch', 'gluten-free'):", value="")

if menu_df.empty or inv_df.empty:
    st.warning("Upload or place your menu & inventory CSVs (root or /data) to enable all features.")
    st.stop()

recos = compute_scores(
    menu_df, inv_df, query_text=chef_query,
    w_query=w_query, w_expiry=w_expiry, w_surplus=w_surplus,
    w_margin=w_margin, w_avail_penalty=w_avail_penalty, top_n=20
)

# Dashboard
st.subheader("🧭 Real‑Time Kitchen Dashboard")

def dish_status(row) -> str:
    if row.get("available_ratio", 0.0) < 1.0:
        return "❌ Not Available"
    if row.get("urgency", 0.0) >= 0.5:
        return "⚠️ Use Soon"
    return "✅ Ready"

recos = recos.copy()
recos["status"] = recos.apply(dish_status, axis=1)
recos_display = recos.copy()
recos_display["margin(%)"] = (recos_display["margin"] * 100).round(1)
st.dataframe(recos_display[["dish_name", "score", "status", "margin(%)"]], use_container_width=True)

# Focus dish details + content
picked = st.selectbox("Focus dish:", options=recos["dish_name"].tolist())
pr = recos.loc[recos["dish_name"] == picked].iloc[0]

colA, colB = st.columns([2, 3])
with colA:
    st.markdown("**Why this ranks well:**")
    st.write({
        "query": float(pr["query"]),
        "urgency": float(pr["urgency"]),
        "surplus": float(pr["surplus"]),
        "margin": float(pr["margin"]),
        "available_ratio": float(pr["available_ratio"]),
    })

with colB:
    cap, script = caption_and_script(picked)
    st.markdown("**📸 Social caption**")
    st.write(cap)
    st.markdown("**🗣 Server 'sales script'**")
    st.write(script)
    with st.expander("🍳 Recipe & video ideas (Tasty)"):
        sugg = tasty_search(picked)
        if not sugg:
            st.caption("Add TASTY_API_KEY/RAPIDAPI_KEY to Streamlit secrets to enable.")
        else:
            for s in sugg:
                st.markdown(f"- **{s.get('name','')}** — {s.get('total_time_tier','')} {'[video]' if s.get('video_url') else ''}")
                if s.get("thumbnail_url"):
                    st.image(s["thumbnail_url"], width=160)

st.markdown("---")

# -------------------- Operational Insights -------------------- #
st.subheader("📊 Operational Insights")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Top 5 Expiring Ingredients**")
    st.dataframe(insight_top_expiring(inv_df, n=5), use_container_width=True)

    st.markdown("**Low Stock Watchlist**")
    st.dataframe(insight_low_stock(inv_df, threshold=3), use_container_width=True)

with c2:
    st.markdown("**🛑 86 Soon (3–5 days)**")
    st.dataframe(insight_dishes_86_soon(menu_df, inv_df, window_max_days=5), use_container_width=True)

    st.markdown("**Overstock Clearance**")
    st.dataframe(insight_overstock(inv_df, qty_threshold=20), use_container_width=True)

with c3:
    st.markdown("**High‑Margin Specials**")
    hm = insight_high_margin(menu_df, threshold=0.7)
    st.dataframe(hm, use_container_width=True)

st.markdown("---")

# -------------------- Planning Add‑ons -------------------- #
st.subheader("📅 Menu Planning Tools")

# Waste Forecast (≤7 days)
def dishes_using(ingredients: List[str], menu: pd.DataFrame) -> List[str]:
    out = []
    names = set(i.lower().strip() for i in ingredients)
    for _, r in menu.iterrows():
        ings = tokenize_ingredients(r.get("ingredients", ""))
        if any(i in names for i in ings):
            out.append(r.get("dish_name",""))
    return out

waste = inv_df.loc[(inv_df["days_to_expiry"] <= 7) & (inv_df["quantity"].astype(float) > 0)].copy()
waste["suggested_dishes"] = waste["ingredient"].apply(lambda x: ", ".join(dishes_using([str(x)], menu_df))[:160])
with st.expander("♻️ Waste Forecast (≤7 days)", expanded=True):
    st.dataframe(waste[["ingredient", "days_to_expiry", "quantity", "suggested_dishes"]].head(15), use_container_width=True)

# Weekly Specials Planner
def weekly_planner(recs: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    items = recs["dish_name"].tolist()
    rows = []
    for i in range(days):
        if not items: break
        rows.append({"Day": i + 1, "Dish": items[i % len(items)]})
    return pd.DataFrame(rows)

with st.expander("🗓️ Weekly Specials Planner (prototype)", expanded=True):
    plan = weekly_planner(recos.head(12), days=7)
    st.dataframe(plan, use_container_width=True)

# Prep Impact (shared preps across top picks) — FIXED to avoid KeyError on 'ingredients'
def prep_impact(recs: pd.DataFrame, menu: pd.DataFrame) -> pd.DataFrame:
    top = recos.head(12).copy()

    # If recommendations already include 'ingredients', use them; otherwise merge from menu.
    ing_col = None
    if "ingredients" in top.columns:
        ing_col = "ingredients"
    else:
        merged = top.merge(menu[["dish_name", "ingredients"]], on="dish_name", how="left")
        if "ingredients" in merged.columns:
            top = merged
            ing_col = "ingredients"
        elif "ingredients_x" in merged.columns:
            top = merged
            ing_col = "ingredients_x"
        elif "ingredients_y" in merged.columns:
            top = merged
            ing_col = "ingredients_y"
        else:
            return pd.DataFrame(columns=["Dish A", "Dish B", "Shared preps"])

    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a = set(tokenize_ingredients(top.iloc[i][ing_col]))
            b = set(tokenize_ingredients(top.iloc[j][ing_col]))
            overlap = a & b
            if len(overlap) >= 2:
                pairs.append({
                    "Dish A": top.iloc[i]["dish_name"],
                    "Dish B": top.iloc[j]["dish_name"],
                    "Shared preps": ", ".join(sorted(list(overlap)))[:120]
                })
    return pd.DataFrame(pairs)

with st.expander("🔪 Prep Impact suggestions", expanded=False):
    pi = prep_impact(recos, menu_df)
    if pi.empty:
        st.caption("No obvious shared prep across the top picks. Adjust query/weights and try again.")
    else:
        st.dataframe(pi, use_container_width=True)

st.markdown("Built for your demo — Mangia Maniga: Your AI Restaurant Assistant (no animations).")
