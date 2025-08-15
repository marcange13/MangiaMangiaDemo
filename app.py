import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from rapidfuzz import fuzz
import plotly.express as px

# --------------------
# Helper functions
# --------------------

def load_data():
    menu_file = None
    inventory_file = None
    
    if os.path.exists('menu_with_viability.csv'):
        menu_file = 'menu_with_viability.csv'
    elif os.path.exists('authentic_menu.csv'):
        menu_file = 'authentic_menu.csv'
        
    if os.path.exists('authentic_inventory.csv'):
        inventory_file = 'authentic_inventory.csv'
    
    menu_df = pd.read_csv(menu_file) if menu_file else None
    inv_df = pd.read_csv(inventory_file) if inventory_file else None
    
    return menu_df, inv_df

def normalize_columns(df):
    col_map = {
        'dish': 'dish_name',
        'name': 'dish_name',
        'ingredients_list': 'ingredients',
        'ingredient_list': 'ingredients',
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    return df

def compute_scores(menu_df, inv_df, query, w_query, w_expiry, w_surplus, w_margin, w_avail_penalty):
    today = datetime.today()
    scores = []
    for idx, row in menu_df.iterrows():
        ingredients = str(row.get('ingredients', '')).split(',')
        ingredients = [i.strip().lower() for i in ingredients if i.strip()]
        
        avail_count = 0
        expiry_score = 0
        surplus_score = 0
        for ing in ingredients:
            inv_row = inv_df[inv_df['ingredient'].str.lower() == ing]
            if not inv_row.empty:
                avail_count += 1
                exp_date = pd.to_datetime(inv_row.iloc[0]['expiry'], errors='coerce')
                qty = inv_row.iloc[0].get('quantity', 0)
                if pd.notnull(exp_date):
                    days_to_expiry = (exp_date - today).days
                    if days_to_expiry <= 5:
                        expiry_score += (5 - days_to_expiry) / 5
                if qty > 10:
                    surplus_score += min(qty/50, 1.0)
        
        avail_ratio = avail_count / max(len(ingredients), 1)
        base_score = row.get('priority_score', 0.5)
        
        query_score = 0
        if query:
            query_score = fuzz.partial_ratio(query.lower(), row.get('dish_name','').lower())/100
        
        margin_score = row.get('margin_pct', 0)
        
        final_score = (
            base_score +
            w_query * query_score +
            w_expiry * expiry_score +
            w_surplus * surplus_score +
            w_margin * margin_score -
            w_avail_penalty * (1 - avail_ratio)
        )
        scores.append(final_score)
    menu_df['final_score'] = scores
    return menu_df.sort_values('final_score', ascending=False)

def top_expiring(inv_df, n=5):
    today = datetime.today()
    inv_df['days_to_expiry'] = (pd.to_datetime(inv_df['expiry'], errors='coerce') - today).dt.days
    return inv_df.sort_values('days_to_expiry').head(n)

def dishes_86_soon(menu_df, inv_df):
    soon = top_expiring(inv_df, len(inv_df))
    soon_ings = set(soon[soon['days_to_expiry'] <= 5]['ingredient'].str.lower())
    mask = menu_df['ingredients'].apply(lambda x: any(ing.strip().lower() in soon_ings for ing in str(x).split(',')))
    return menu_df[mask]

def high_margin_specials(menu_df, threshold=0.7):
    return menu_df[menu_df['margin_pct'] >= threshold]

def low_stock_watchlist(inv_df, threshold=3):
    return inv_df[inv_df['quantity'] <= threshold]

def overstock_clearance(inv_df, threshold=20):
    return inv_df[inv_df['quantity'] >= threshold]

# --------------------
# Streamlit UI
# --------------------

st.title("🍝 Kitchen Brain – Inventory-Aware Menu Planner")

menu_df, inv_df = load_data()

with st.sidebar:
    st.header("Data Upload")
    if menu_df is None:
        menu_file = st.file_uploader("Upload menu CSV", type=['csv'])
        if menu_file:
            menu_df = pd.read_csv(menu_file)
    if inv_df is None:
        inv_file = st.file_uploader("Upload inventory CSV", type=['csv'])
        if inv_file:
            inv_df = pd.read_csv(inv_file)
    
    st.header("Weights")
    w_query = st.slider("Chef query weight", 0.0, 2.0, 0.5, 0.1)
    w_expiry = st.slider("Expiry urgency weight", 0.0, 2.0, 1.0, 0.1)
    w_surplus = st.slider("Surplus weight", 0.0, 2.0, 0.5, 0.1)
    w_margin = st.slider("Margin weight", 0.0, 2.0, 0.5, 0.1)
    w_avail_penalty = st.slider("Penalty for missing ingredients", 0.0, 2.0, 1.0, 0.1)

if menu_df is not None and inv_df is not None:
    menu_df = normalize_columns(menu_df)
    inv_df = normalize_columns(inv_df)
    
    st.subheader("Chef Query")
    query = st.text_input("Describe what you want (e.g., 'seafood special')", "")
    
    ranked_menu = compute_scores(menu_df, inv_df, query, w_query, w_expiry, w_surplus, w_margin, w_avail_penalty)
    st.subheader("Ranked Recommendations")
    st.dataframe(ranked_menu[['dish_name','final_score'] + [col for col in ranked_menu.columns if col not in ['dish_name','final_score']]])
    
    st.subheader("Operational Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Expiring Ingredients**")
        st.dataframe(top_expiring(inv_df))
    with col2:
        st.markdown("**86 Soon Dishes**")
        st.dataframe(dishes_86_soon(menu_df, inv_df))
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**High-Margin Specials**")
        st.dataframe(high_margin_specials(menu_df))
    with col4:
        st.markdown("**Low Stock Watchlist**")
        st.dataframe(low_stock_watchlist(inv_df))
    
    st.markdown("**Overstock Clearance**")
    st.dataframe(overstock_clearance(inv_df))
else:
    st.warning("Please upload both menu and inventory CSVs.")
