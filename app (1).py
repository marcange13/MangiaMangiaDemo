import io
import os
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
    page_title="Mangia Mangia — Roma Cucina Assistant",
    page_icon="🍝",
    layout="wide",
)

# ======================= Roman Trattoria Theme (no sidebar) =======================
st.markdown(\"\"\"
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Cormorant+Garamond:wght@400;500;600&display=swap');

/* ===== Roman Trattoria Theme ===== */
:root{
  --marble:#f6f3ee;         /* marble base */
  --travertine:#efe6d6;     /* warmer stone */
  --ink:#2a2622;            /* deep brown-black */
  --terracotta:#b45736;     /* terra cotta clay */
  --imperial:#7c2f2f;       /* imperial red wine */
  --olive:#6f7b4b;          /* Roman olive leaves */
  --gold:#bda46a;           /* antique gold */
  --card:#fffaf2;           /* parchment card */
  --shadow:0 10px 28px rgba(42,38,34,.08);
}

/* Hide default sidebar/hamburger */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important;}

/* Background: subtle marble & stone */
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 800px at 10% 15%, var(--travertine), var(--marble)),
    repeating-linear-gradient(135deg, rgba(255,255,255,.0) 0px, rgba(255,255,255,.0) 14px, rgba(0,0,0,.015) 15px, rgba(0,0,0,.015) 16px);
  background-attachment: fixed;
}
.main .block-container {padding-top: 1rem; padding-bottom: 2rem;}

/* Typography */
h1, h2, h3 {
  font-family: 'Cinzel', serif !important;
  letter-spacing:.4px; color: var(--ink);
}
p, div, span, label, input, textarea, button {
  font-family: 'Cormorant Garamond', serif !important;
  font-size: 18px; color: var(--ink);
}

/* Brand header – Roman border & laurel accent */
.m-brand {
  display:flex; align-items:center; gap:.8rem;
  background: linear-gradient(180deg, #fffdf7, #f9efdc);
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 18px;
  padding: .9rem 1.1rem;
  box-shadow: var(--shadow);
  position: relative;
}
.m-brand:before, .m-brand:after {
  content:""; position:absolute; inset:-6px;
  border-radius: 22px;
  border: 2px solid rgba(189,164,106,.35); /* antique gold line */
  pointer-events:none;
}
.m-laurel { font-size: 20px; color: var(--olive); filter: drop-shadow(0 1px 0 rgba(0,0,0,.06)); }

/* Toolbar & buttons */
.toolbar {display:flex; gap:.75rem; align-items:center; flex-wrap:wrap; margin:.9rem 0 1.1rem 0;}
.stButton>button, .m-btn {
  background: linear-gradient(180deg, #fff1e8, #ffe1d3);
  border: 1px solid #e0c1ad;
  color: var(--terracotta);
  font-weight:700;
  border-radius:14px; padding:.55rem 1rem; cursor:pointer;
}
.stButton>button:hover, .m-btn:hover { filter:contrast(1.06); transform: translateY(-1px); }

/* Cards */
.m-card {
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 18px;
  padding: 1rem;
  background: var(--card);
  box-shadow: var(--shadow);
  margin-bottom:.8rem;
}
.m-card h3 { margin: 0 0 .25rem 0; }

/* Tabs */
div.stTabs [data-baseweb="tab-list"] { gap:.5rem; }
div.stTabs [data-baseweb="tab"] {
  padding:.55rem .95rem; border-radius:12px;
  background: rgba(189,164,106,.10);
  border: 1px solid rgba(189,164,106,.35);
}
div.stTabs [aria-selected="true"] {
  background: rgba(111,123,75,.12);
  border-color: rgba(111,123,75,.45);
  color: var(--olive);
}

/* Metrics + labels */
[data-testid="stMetricValue"]{ color: var(--imperial) !important; }
small, .caption, .stCaption, [data-testid="stMarkdownContainer"] .caption {
  color: #5b564f !important; opacity:.9;
}
</style>
\"\"\", unsafe_allow_html=True)

# =================================== Header ===================================
c1, c2 = st.columns([0.72, 0.28])
with c1:
    st.markdown(\"\"\"
<div class="m-brand">
  <div class="m-laurel">❦</div>
  <div>
    <h1 style="margin:0;">Mangia Mangia · Roma Cucina</h1>
    <div style="opacity:.82;">Autentico • Sostenibile • Margini Protetti</div>
  </div>
  <div class="m-laurel" style="margin-left:auto;">❦</div>
</div>
\"\"\", unsafe_allow_html=True)
with c2:
    st.metric("Obiettivo di Oggi", "Spreco ↓  ·  Margine ↑", delta="Equilibrato")

# ============================ Top Toolbar (no sidebar) =========================
st.markdown('<div class="toolbar">', unsafe_allow_html=True)
query = st.text_input(
    "What should we cook today?",
    placeholder="e.g., ‘broccoli pasta’, ‘vegetarian lunch’, ‘seafood special’",
    label_visibility="collapsed"
)
run_btn = st.button("Generate Suggestions", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Options (Upload CSV, Scoring Weights, LLM)", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload your menu CSV",
        type=["csv"],
        help="Must include columns: " + ", ".join(REQUIRED_COLUMNS)
    )
    data_path = st.text_input("…or use CSV path", "data/menu_with_viability.csv")

    # Template download
    template_buf = io.BytesIO()
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(template_buf, index=False)
    st.download_button(
        label="Download template CSV",
        data=template_buf.getvalue(),
        file_name="menu_template.csv",
        mime="text/csv"
    )

    cA, cB, cC, cD, cE = st.columns(5)
    with cA:
        w_priority = st.slider("Weight: Priority", 0.0, 1.0, 0.55, 0.05)
    with cB:
        w_margin   = st.slider("Weight: Margin",   0.0, 1.0, 0.25, 0.05)
    with cC:
        w_urgency  = st.slider("Weight: Urgency",  0.0, 1.0, 0.10, 0.05)
    with cD:
        w_surplus  = st.slider("Weight: Surplus",  0.0, 1.0, 0.10, 0.05)
    with cE:
        top_k = st.slider("Top K dishes", 1, 10, 5)

    use_llm = st.checkbox("Use ChatGPT (if key set)", value=True)

weights = {"priority": w_priority if 'w_priority' in locals() else 0.55,
           "margin":   w_margin   if 'w_margin'   in locals() else 0.25,
           "urgency":  w_urgency  if 'w_urgency'  in locals() else 0.10,
           "surplus":  w_surplus  if 'w_surplus'  in locals() else 0.10}
top_k = top_k if 'top_k' in locals() else 5

# ================================ Fallback Dataset ================================
def _fallback_menu_df():
    return pd.DataFrame([
        {
            "dish_name":"Spaghetti Aglio e Olio","price":14.0,"cost":3.2,
            "profit_margin_pct":77.1,"priority_score":0.86,
            "urgency_score":0.30,"surplus_score":0.20,
            "ingredients":"spaghetti, garlic, olive oil, parsley, chili"
        },
        {
            "dish_name":"Panzanella Toscana","price":12.0,"cost":2.5,
            "profit_margin_pct":79.2,"priority_score":0.74,
            "urgency_score":0.45,"surplus_score":0.40,
            "ingredients":"stale bread, tomato, cucumber, onion, basil"
        },
        {
            "dish_name":"Frittata di Verdure","price":13.0,"cost":2.9,
            "profit_margin_pct":77.7,"priority_score":0.70,
            "urgency_score":0.55,"surplus_score":0.35,
            "ingredients":"eggs, onion, zucchini, spinach, pecorino"
        }
    ])

# ================================ Data Loading ================================
@st.cache_data(show_spinner=False)
def _load_from_path(path: str) -> pd.DataFrame:
    return load_menu(path)

df = None
source_note = ""
try:
    if 'uploaded_file' in locals() and uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        missing, extra = validate_schema(raw.columns)
        with st.expander("CSV schema check (uploaded)", expanded=(len(missing) > 0 or len(extra) > 0)):
            if not missing and not extra:
                st.success("Schema OK ✅ All required columns present.")
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            if extra:
                st.warning(f"Extra columns (ignored by ranking): {', '.join(extra)}")
            st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
        df = clean_menu_df(raw)
        source_note = "Using **uploaded** dataset ✅"
    else:
        default_path = "data/menu_with_viability.csv"
        if os.path.exists(default_path):
            df = _load_from_path(default_path)
            source_note = f"Using dataset from: `{default_path}`"
        else:
            df = clean_menu_df(_fallback_menu_df())
            source_note = "Using built-in fallback sample (place your CSV at data/menu_with_viability.csv)"
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# Preview (folded by default)
with st.expander("Sample preview (first 8 rows)", expanded=False):
    st.dataframe(df.head(8), use_container_width=True)

st.caption(source_note)

# =================================== Run ======================================
if run_btn:
    ranked = handle_user_query(df, query=query or "", top_k=top_k, weights=weights)

    tab1, tab2, tab3 = st.tabs(["Top Picks", "Recipe Ideas", "Assistant Notes"])

    with tab1:
        st.subheader("Top Picks (inventory‑aware) 🍝🍕🍅🍷")
        for _, r in ranked.iterrows():
            st.markdown("<div class='m-card'>", unsafe_allow_html=True)
            cc1, cc2, cc3, cc4 = st.columns([0.42, 0.16, 0.20, 0.22])
            with cc1:
                st.markdown(f"### {r['dish_name']}")
                st.caption(r.get("ingredients",""))
            with cc2:
                st.metric("Margin %", f"{r['profit_margin_pct']:.0f}%")
            with cc3:
                st.caption("Scores 🍝🍕")
                st.write(
                    f"Priority: **{r['priority_score']:.2f}**  \\n"
                    f"Urgency: **{r['urgency_score']:.2f}**  \\n"
                    f"Surplus: **{r['surplus_score']:.2f}**"
                )
            with cc4:
                st.caption("Price / Cost 🍅🍷")
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
        if 'use_llm' in locals() and use_llm:
            user_prompt = USER_PROMPT_TEMPLATE.format(query=query or "chef’s choice")
            user_prompt += "\\n\\n" + CHEF_TONE_REMINDER
            response = chat_complete(SYSTEM_PROMPT, user_prompt)
            st.write(response)
        else:
            st.caption("LLM disabled. Enable in Options → Use ChatGPT.")
else:
    st.info("Enter a query and click **Generate Suggestions**.")
