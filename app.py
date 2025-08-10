import io
import pandas as pd
import streamlit as st

from logic import (
    REQUIRED_COLUMNS,
    load_menu,
    clean_menu_df,
    handle_user_query,
    tasty_for_top,
    chat_complete,
    validate_schema,
)
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, CHEF_TONE_REMINDER

st.set_page_config(
    page_title="Mangia Mangia — Cucina Assistant",
    page_icon="🍝",
    layout="wide",
)

# ============= Italian trattoria theme =============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Italiana&family=Crimson+Text:wght@400;600;700&display=swap');

:root{
  --cream:#f7f1e3;
  --parchment:#f4e7cd;
  --parchment-ink:#312a23;
  --olive:#6a7b4f;
  --tomato:#b23a2a;
  --chianti:#7a2d2d;
  --gold:#c9a227;
  --card:#fffaf0;
  --shadow:0 12px 30px rgba(49,42,35,.08);
}

html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 800px at 10% 10%, var(--parchment), var(--cream)) fixed;
}

.main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

h1, h2, h3 {
  font-family: 'Italiana', serif !important;
  letter-spacing:.3px;
  color: var(--parchment-ink);
}
p, div, span, label {
  font-family: 'Crimson Text', serif !important;
  font-size: 18px;
  color: var(--parchment-ink);
}

.m-brand {
  display:flex; align-items:center; gap:.6rem;
  background: linear-gradient(180deg, #fff9ef, #f8eed8);
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 18px;
  padding: .85rem 1rem;
  box-shadow: var(--shadow);
}
.m-brand .dot {
  width:14px; height:14px; border-radius:999px; background: var(--tomato);
  box-shadow: 0 0 0 3px rgba(178,58,42,.18);
}

.m-card {
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 18px;
  padding: 1rem 1rem 1rem 1rem;
  background: var(--card);
  box-shadow: var(--shadow);
  margin-bottom:.75rem;
}
.m-card h3 { margin: 0 0 .25rem 0; }

.m-pill {
  display:inline-block;
  padding:.2rem .6rem;
  border-radius:999px;
  font-size: .9rem;
  background: rgba(106,123,79,.10);
  border: 1px solid rgba(106,123,79,.25);
  color: var(--olive);
  margin-right:.35rem;
}

.stButton>button {
  background: linear-gradient(180deg, #fff3e6, #ffe2cc);
  border: 1px solid #e5c2a6;
  color:#5a2c24;
  font-weight:600;
  border-radius:14px;
  padding:.55rem 1rem;
}
.stButton>button:hover { filter: contrast(1.05); transform: translateY(-1px); }

div.stTabs [data-baseweb="tab-list"] { gap:.5rem; }
div.stTabs [data-baseweb="tab"] {
  padding: .5rem .9rem; border-radius: 12px;
  background: rgba(201,162,39,.08); border:1px solid rgba(201,162,39,.25);
}
</style>
""", unsafe_allow_html=True)

# ============= Sidebar =============
st.sidebar.title("Mangia Mangia 🍝")
st.sidebar.caption("Inventory-aware menu suggestions to cut waste & boost margin.")

# Upload OR use default CSV path
uploaded_file = st.sidebar.file_uploader(
    "Upload your menu CSV",
    type=["csv"],
    help="Must include columns: " + ", ".join(REQUIRED_COLUMNS)
)
data_path = st.sidebar.text_input("…or use CSV path", "data/menu_with_viability.csv")

# Template download
template_buf = io.BytesIO()
pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(template_buf, index=False)
st.sidebar.download_button(
    label="Download template CSV",
    data=template_buf.getvalue(),
    file_name="menu_template.csv",
    mime="text/csv"
)

weights = {
    "priority": st.sidebar.slider("Weight: Priority", 0.0, 1.0, 0.55, 0.05),
    "margin":   st.sidebar.slider("Weight: Margin",   0.0, 1.0, 0.25, 0.05),
    "urgency":  st.sidebar.slider("Weight: Urgency",  0.0, 1.0, 0.10, 0.05),
    "surplus":  st.sidebar.slider("Weight: Surplus",  0.0, 1.0, 0.10, 0.05),
}
top_k = st.sidebar.slider("Top K dishes", 1, 10, 5)
use_llm = st.sidebar.checkbox("Use ChatGPT (if key set)", value=True)
st.sidebar.divider()
st.sidebar.write("**Tips**")
st.sidebar.caption("• Upload your CSV or use the default.\n• Narrow your query (e.g., “chicken pasta”).\n• Tune weights to today’s goals.")

# ============= Header =============
c1, c2 = st.columns([0.72, 0.28])
with c1:
    st.markdown("""
<div class="m-brand">
  <div class="dot"></div>
  <div>
    <h1 style="margin:0;">Mangia Mangia — Cucina Assistant</h1>
    <div style="opacity:.8;">Reduce waste • Protect margins • Serve authentic Italian fare.</div>
  </div>
</div>
""", unsafe_allow_html=True)
with c2:
    st.metric("Today’s Goal", "Waste ↓ + Margin ↑", delta="Balanced")

# ============= Data loading =============
@st.cache_data(show_spinner=False)
def _load_from_path(path: str) -> pd.DataFrame:
    return load_menu(path)

df = None
source_note = ""
try:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        # Schema check panel
        missing, extra = validate_schema(raw.columns)
        with st.expander("CSV schema check (uploaded)", expanded=(len(missing) > 0 or len(extra) > 0)):
            if not missing and not extra:
                st.success("Schema OK ✅ All required columns present.")
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            if extra:
                st.warning(f"Extra columns (ignored by ranking): {', '.join(extra)}")
            st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))

        df = clean_menu_df(raw)  # raises if missing critical columns
        source_note = "Using **uploaded** dataset ✅"
    else:
        df = _load_from_path(data_path)
        source_note = f"Using dataset from path: `{data_path}`"
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# quick preview
with st.expander("Sample preview", expanded=False):
    st.dataframe(df.head(8), use_container_width=True)

# ============= Query + Run =============
query = st.text_input(
    "What should we cook today?",
    placeholder="e.g., ‘broccoli pasta’, ‘vegetarian lunch’, ‘seafood special’"
)
st.caption(source_note)
run = st.button("Generate Suggestions")

if run:
    ranked = handle_user_query(df, query=query or "", top_k=top_k, weights=weights)

    tab1, tab2, tab3 = st.tabs(["Top Picks", "Recipe Ideas", "Assistant Notes"])

    with tab1:
        st.subheader("Top Picks (inventory-aware) 🍷🍅")
        for _, r in ranked.iterrows():
            st.markdown("<div class='m-card'>", unsafe_allow_html=True)
            cc1, cc2, cc3, cc4 = st.columns([0.42, 0.16, 0.20, 0.22])
            with cc1:
                st.markdown(f"### {r['dish_name']}")
                st.caption(r.get("ingredients",""))
            with cc2:
                st.metric("Margin %", f"{r['profit_margin_pct']:.0f}%")
            with cc3:
                st.caption("Scores")
                st.write(
                    f"Priority: **{r['priority_score']:.2f}**  \n"
                    f"Urgency: **{r['urgency_score']:.2f}**  \n"
                    f"Surplus: **{r['surplus_score']:.2f}**"
                )
            with cc4:
                st.caption("Price / Cost")
                st.write(f"Price: **${r['price']:.2f}**")
                st.write(f"Cost: **${r['cost']:.2f}**")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        with st.expander("See raw table"):
            st.dataframe(ranked.reset_index(drop=True), use_container_width=True)

    with tab2:
        st.subheader("Recipe/Video Ideas (Tasty)")
        ideas_map = tasty_for_top(ranked, max_items=min(3, len(ranked)))
        for dish, ideas in ideas_map.items():
            st.markdown(f"**{dish}**")
            if not ideas:
                st.caption("No ideas available.")
                continue
            for it in ideas:
                st.markdown(f"- {it.get('name','(unnamed)')} — {it.get('desc','')}")
                if it.get("video_url"):
                    st.markdown(f"[Video]({it['video_url']})")
            st.write("")

    with tab3:
        st.subheader("Assistant Notes")
        if use_llm:
            user_prompt = USER_PROMPT_TEMPLATE.format(query=query or "chef’s choice")
            user_prompt += "\n\n" + CHEF_TONE_REMINDER
            response = chat_complete(SYSTEM_PROMPT, user_prompt)
            st.write(response)
        else:
            st.caption("LLM disabled. Enable in sidebar to generate notes.")
else:
    st.info("Enter a query (or leave blank) and click **Generate Suggestions** to see top picks.")

