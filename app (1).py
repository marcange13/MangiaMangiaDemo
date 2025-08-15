
import io
import os
import time
import pandas as pd
import streamlit as st

from logic import (
    # core
    load_menu, load_inventory, clean_menu_df, handle_user_query,
    # insights
    top_expiring_ingredients, at_risk_dishes_86, high_margin_specials,
    low_stock_watchlist, overstock_clearance, waste_forecast_summary,
    overstock_dish_matches, weekly_specials_plan, batch_prep_suggestions,
    social_caption_text, server_script_text, tasty_for_top, chat_complete,
)

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, CHEF_TONE_REMINDER

st.set_page_config(page_title="Mangia Mangia — Nonna's Kitchen", page_icon="👵🍝", layout="wide")

# ======== Subtle 'Nonna's house' theme (neutral linens + charcoal, simple font) ========
st.markdown(
    """
    <style>
      :root{
        --linen:#faf7f2;
        --porcelain:#f4f2ee;
        --paper:#fffdf8;
        --ink:#2f2f2f;
        --shadow:0 10px 30px rgba(0,0,0,.06);
        --radius:16px;
      }
      html, body, [data-testid="stAppViewContainer"]{
        background:
          radial-gradient(1200px 800px at 0% 0%, var(--porcelain), var(--linen));
      }
      .main .block-container{padding-top:1rem; padding-bottom:2rem;}
      * { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'; }
      h1,h2,h3 { letter-spacing:.3px; color: var(--ink); }
      .card{
        background: var(--paper);
        border: 1px solid rgba(0,0,0,.06);
        border-radius: var(--radius);
        padding: 1rem;
        box-shadow: var(--shadow);
        margin-bottom: .75rem;
      }
      .pill{
        display:inline-block; padding:.25rem .6rem; border-radius:999px;
        background: rgba(0,0,0,.05); color: var(--ink); font-size:.85rem;
        border: 1px solid rgba(0,0,0,.08);
      }
      .soft{opacity:.85}
      .nonna-card { background: var(--paper); border: 1px dashed rgba(0,0,0,.12); border-radius: 16px; padding: .8rem 1rem; box-shadow: var(--shadow); }
      .muted{color:#6b6b6b}
      .metric-row{display:flex; gap:.75rem; flex-wrap:wrap}

      /* Pulse animation for alert mood */
      @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(200,0,0,.12); }
        50% { transform: scale(1.01); box-shadow: 0 10px 32px rgba(200,0,0,.08); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(200,0,0,.12); }
      }
      .pulse { animation: pulse 1.5s ease-in-out infinite; }
    </style>
    """, unsafe_allow_html=True
)

# ======== Header ========
c1, c2 = st.columns([0.72, 0.28])
with c1:
    st.title("Mangia Mangia · Nonna’s Kitchen")
    st.caption("Simple. Seasonal. Nothing wasted. 🍞🫒🧄")
with c2:
    st.metric("Today", "Less waste, more margin", delta="Balanced")

# ======== Sidebar: Nonna + Data + Options ========
with st.sidebar:
    st.subheader("👵 Nonna")

    # Session state for mood + wave timer
    st.session_state.setdefault("nonna_mood", "hello")
    st.session_state.setdefault("nonna_wave_until", 0.0)

    def trigger_wave(seconds: float = 2.5):
        st.session_state["nonna_wave_until"] = time.time() + seconds

    # Asset paths
    def _asset(path):
        return path if os.path.exists(path) else None

    NONNA = {
        "hello": _asset("assets/nonna_hello.png"),
        "happy": _asset("assets/nonna_happy.png"),
        "alert": _asset("assets/nonna_alert.png"),
        "wave":  _asset("assets/nonna_wave.gif"),
    }

    # Decide what to show now
    now = time.time()
    wave_active = NONNA["wave"] and (now < st.session_state.get("nonna_wave_until", 0.0))
    mood = st.session_state["nonna_mood"]
    mood_img = NONNA.get(mood) or NONNA["hello"]
    is_alert = (mood == "alert")
    css_classes = "nonna-card" + (" pulse" if is_alert else "")

    st.markdown(f'<div class="{css_classes}">', unsafe_allow_html=True)
    if wave_active and NONNA["wave"]:
        st.image(NONNA["wave"], use_container_width=True)
    else:
        st.image(mood_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Manual wave test button
    if st.button("Wave at me 👋", use_container_width=True):
        trigger_wave()

    st.markdown("---")
    st.header("Data & Options")
    menu_upload = st.file_uploader("Menu CSV", type=["csv"], help="Needs dish_name & ingredients (synonyms OK).")
    menu_path = st.text_input("…or menu path", os.getenv("MENU_CSV_PATH", "data/authentic_menu_priced_2025.csv"))

    inv_upload = st.file_uploader("Inventory CSV", type=["csv"], help="Needs an 'ingredient' column (synonyms OK).")
    inv_path = st.text_input("…or inventory path", os.getenv("INVENTORY_CSV_PATH", "data/authentic_inventory.csv"))

    st.markdown("###### Ranking Weights")
    w_cols = st.columns(5)
    with w_cols[0]:
        w_priority = st.slider("Priority", 0.0, 1.0, 0.45, 0.05)
    with w_cols[1]:
        w_margin   = st.slider("Margin",   0.0, 1.0, 0.25, 0.05)
    with w_cols[2]:
        w_urgency  = st.slider("Urgency",  0.0, 1.0, 0.15, 0.05)
    with w_cols[3]:
        w_surplus  = st.slider("Surplus",  0.0, 1.0, 0.05, 0.05)
    with w_cols[4]:
        w_avail    = st.slider("Avail",    0.0, 1.0, 0.10, 0.05)

    top_k = st.slider("Top K dishes", 1, 12, 6)
    horizon_days = st.slider("86 Horizon (days)", 1, 14, 4)
    use_llm = st.checkbox("Assistant Notes (OpenAI)", value=True)
    use_tasty = st.checkbox("Recipe Ideas (Tasty API)", value=False)
    st.caption("Tip: Put PNG/GIFs in ./assets named nonna_hello/happy/alert.png and nonna_wave.gif")

# ======== Load data ========
@st.cache_data(show_spinner=False)
def _load_menu(path: str) -> pd.DataFrame:
    return load_menu(path)

@st.cache_data(show_spinner=False)
def _load_inv(path: str) -> pd.DataFrame:
    return load_inventory(path)

try:
    menu_df = clean_menu_df(pd.read_csv(menu_upload)) if menu_upload else _load_menu(menu_path)
    menu_source = "Uploaded menu ✅" if menu_upload else f"Loaded from {menu_path}"
except Exception as e:
    st.error(f"Failed to load Menu CSV: {e}")
    st.stop()

try:
    inv_df = _load_inv(inv_upload) if inv_upload else _load_inv(inv_path)
    inv_source = "Uploaded inventory ✅" if inv_upload else f"Loaded from {inv_path}"
except Exception as e:
    st.error(f"Failed to load Inventory CSV: {e}")
    st.stop()

st.caption(f"Data — Menu: {menu_source} | Inventory: {inv_source}")

# ======== Query ========
query = st.text_input("What should we cook today?", placeholder="e.g., ‘tomatoes and eggs’, ‘zucchini lunch’, ‘seafood special’")
run = st.button("Ask Nonna 🍽️")

def set_nonna(mood:str):
    st.session_state["nonna_mood"] = mood

# ======== Run ========
if run:
    # Make Nonna wave on click
    st.session_state["nonna_wave_until"] = time.time() + 2.5

    weights = {"priority": w_priority, "margin": w_margin, "urgency": w_urgency, "surplus": w_surplus, "availability": w_avail}
    ranked = handle_user_query(menu_df, query=query or "", top_k=top_k, weights=weights, inventory_df=inv_df)

    # Mood logic
    ready_count = int(ranked.get("can_be_made", pd.Series(dtype=bool)).sum())
    urgent_avg = float(ranked.get("dyn_urgency", pd.Series([0])).mean())
    if ready_count >= max(1, top_k//2) and urgent_avg < 0.4:
        set_nonna("happy")
    elif urgent_avg >= 0.7:
        set_nonna("alert")
    else:
        set_nonna("hello")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Top Picks", "Insights", "Planning", "Dashboard", "Boosters"])

    with tab1:
        st.subheader("Top Picks (Inventory‑Aware) 🍝🍕🍅🍷")
        for _, r in ranked.iterrows():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns([0.42, 0.16, 0.22, 0.20])
            with c1:
                badge = "✅ Ready" if r.get("can_be_made", False) else ("⚠️ Use Soon" if r.get("coverage_pct",0)>0 else "❌ 86")
                st.markdown(f"**{r['dish_name']}**  <span class='pill'>{badge}</span>", unsafe_allow_html=True)
                miss = r.get("missing_ingredients", [])
                if miss:
                    st.caption("Missing: " + ", ".join(miss[:3]))
                st.caption(r.get("ingredients",""))
            with c2:
                st.metric("Margin %", f"{r.get('profit_margin_pct',0):.0f}%")
            with c3:
                st.caption("Scores")
                st.write(
                    f"Priority: **{r.get('priority_score',0):.2f}**  \n"
                    f"Urgency (dyn): **{r.get('dyn_urgency',0):.2f}**  \n"
                    f"Surplus (dyn): **{r.get('dyn_surplus',0):.2f}**  \n"
                    f"Avail: **{(r.get('coverage_pct',0)*100):.0f}%**"
                )
            with c4:
                st.caption("Price / Cost")
                st.write(f"Price: **${r.get('price',0):.2f}**")
                st.write(f"Cost: **${r.get('cost',0):.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("See raw table"):
            st.dataframe(ranked.reset_index(drop=True), use_container_width=True)

        if use_tasty:
            st.divider()
            st.subheader("Recipe/Video Ideas")
            ideas_map = tasty_for_top(ranked, max_items=min(3, len(ranked)))
            for dish, ideas in ideas_map.items():
                st.markdown(f"**{dish}**")
                for it in ideas:
                    st.markdown(f"- {it.get('name','(unnamed)')} — {it.get('desc','')}")

    with tab2:
        st.subheader("Operational Insights")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("#### 🍂 Top Expiring Ingredients")
            exp_df = top_expiring_ingredients(inv_df, top_n=5)
            st.dataframe(exp_df, use_container_width=True)
        with cB:
            st.markdown("#### 🛑 86 Soon")
            risk_df = at_risk_dishes_86(menu_df, inv_df, horizon_days=horizon_days)
            st.dataframe(risk_df, use_container_width=True)

        cC, cD = st.columns(2)
        with cC:
            st.markdown("#### 💰 High‑Margin Specials (Ready)")
            hm = high_margin_specials(ranked, n=6)
            st.dataframe(hm, use_container_width=True)
        with cD:
            st.markdown("#### 🛒 Low Stock Watchlist")
            low = low_stock_watchlist(inv_df)
            st.dataframe(low, use_container_width=True)

        st.markdown("#### 📦 Overstock Clearance & Matches")
        over = overstock_clearance(inv_df)
        st.dataframe(over, use_container_width=True)
        matches = overstock_dish_matches(over, menu_df)
        if not matches.empty:
            st.dataframe(matches, use_container_width=True)

    with tab3:
        st.subheader("Menu Planning Tools")
        st.markdown("##### 📅 Weekly Specials Planner")
        plan = weekly_specials_plan(menu_df, inv_df, days=7)
        st.dataframe(plan, use_container_width=True)

        st.markdown("##### ♻️ Waste Forecast (7 Days)")
        wf = waste_forecast_summary(inv_df, days=7)
        st.dataframe(wf, use_container_width=True)

        st.markdown("##### 🔄 Batch Prep Suggestions")
        batch = batch_prep_suggestions(menu_df)
        st.dataframe(batch, use_container_width=True)

    with tab4:
        st.subheader("Real‑Time Kitchen Dashboard")
        ready = int(ranked.get("can_be_made", pd.Series(dtype=bool)).sum())
        total = len(ranked)
        use_soon = int((ranked.get("coverage_pct", pd.Series([0]*total)) > 0).sum() - ready)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
          <span class="pill">✅ Ready: <b>{ready}</b></span>
          <span class="pill">⚠️ Use Soon: <b>{use_soon}</b></span>
          <span class="pill">❌ 86: <b>{total - ready - use_soon}</b></span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        try:
            st.bar_chart(ranked.set_index("dish_name")[["demo_score"]])
            st.bar_chart(ranked.set_index("dish_name")[["coverage_pct"]])
        except Exception:
            pass

    with tab5:
        st.subheader("Customer Experience Boosters")
        picks = list(ranked["dish_name"].head(3).values)
        colx, coly = st.columns(2)
        with colx:
            st.markdown("###### 🗣 Server Script")
            st.write(server_script_text(picks))
        with coly:
            st.markdown("###### 📸 Social Caption")
            st.write(social_caption_text(picks))

        st.markdown("###### 🍷 Pairing Hints")
        if use_llm:
            user_prompt = USER_PROMPT_TEMPLATE.format(query=f"Pair wines with: {', '.join(picks)}")
            user_prompt += "\n\nFocus on short, practical pairings for a Roman trattoria."
            st.write(chat_complete(SYSTEM_PROMPT, user_prompt))
        else:
            st.caption("Enable Assistant Notes to generate pairings.")

else:
    st.info("Type what you have (e.g., ‘tomatoes and eggs’) then click **Ask Nonna** to get picks, insights, and plans.")
