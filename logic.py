import time
from typing import List, Dict, Any, Iterable

import numpy as np
import pandas as pd
import requests
import streamlit as st

# === Secrets ===
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
TASTY_API_KEY = st.secrets.get("TASTY_API_KEY", "")
USE_TASTY = str(st.secrets.get("USE_TASTY", "true")).lower() == "true"
MODEL = st.secrets.get("MODEL", "gpt-4o-mini")

# === Schema ===
REQUIRED_COLUMNS: List[str] = [
    "dish_name", "price", "cost", "profit_margin_pct", "priority_score",
    "urgency_score", "surplus_score", "ingredients"
]

def validate_schema(cols: Iterable[str]):
    cols_set = set([c.strip() for c in cols])
    req_set = set(REQUIRED_COLUMNS)
    missing = sorted(list(req_set - cols_set))
    extra = sorted(list(cols_set - req_set))
    return missing, extra

# ---------- Data helpers ----------
def clean_menu_df(df: pd.DataFrame) -> pd.DataFrame:
    missing, _ = validate_schema(df.columns)
    if missing:
        raise ValueError(f"menu CSV missing columns: {missing}")
    df = df.copy()
    numeric_cols = ["profit_margin_pct", "priority_score", "urgency_score", "surplus_score", "price", "cost"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["ingredients"] = df["ingredients"].fillna("").astype(str)
    return df

def load_menu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_menu_df(df)

def filter_by_keywords(df: pd.DataFrame, query: str) -> pd.DataFrame:
    toks = [t.strip().lower() for t in query.split() if t.strip()]
    if not toks:
        return df
    mask = np.ones(len(df), dtype=bool)
    for t in toks:
        mask &= (
            df["dish_name"].str.lower().str.contains(t, na=False) |
            df["ingredients"].str.lower().str.contains(t, na=False)
        )
    return df[mask]

def compute_demo_score(row, weights=None):
    w = {"priority": 0.55, "margin": 0.25, "urgency": 0.10, "surplus": 0.10}
    if weights:
        w.update(weights)
    margin = row.get("profit_margin_pct", 0) / 100.0
    return (
        w["priority"] * row.get("priority_score", 0) +
        w["margin"]   * margin +
        w["urgency"]  * row.get("urgency_score", 0) +
        w["surplus"]  * row.get("surplus_score", 0)
    )

def rank_dishes(df: pd.DataFrame, top_k: int = 5, weights=None) -> pd.DataFrame:
    temp = df.copy()
    temp["demo_score"] = temp.apply(lambda r: compute_demo_score(r, weights), axis=1)
    temp = temp.sort_values(by=["demo_score","priority_score","profit_margin_pct"], ascending=False)
    return temp.head(top_k)

# ---------- Public entrypoints ----------
def handle_user_query(df: pd.DataFrame, query: str, top_k: int = 5, weights=None) -> pd.DataFrame:
    sub = filter_by_keywords(df, query)
    if sub.empty:
        sub = df  # fallback to full set
    ranked = rank_dishes(sub, top_k=top_k, weights=weights)
    return ranked

def tasty_for_top(ranked_df: pd.DataFrame, max_items: int = 3):
    names = ranked_df["dish_name"].head(max_items).tolist()
    out: Dict[str, Any] = {}
    for n in names:
        out[n] = tasty_search(n, num=2)
        time.sleep(0.1)
    return out

# ---------- Tasty API (live + fallback) ----------
def tasty_search(query: str, num: int = 3) -> List[Dict[str, Any]]:
    if not USE_TASTY or not TASTY_API_KEY:
        return tasty_mock(query, num)
    try:
        url = "https://tasty.p.rapidapi.com/recipes/list"
        headers = {
            "x-rapidapi-host": "tasty.p.rapidapi.com",
            "x-rapidapi-key": TASTY_API_KEY
        }
        params = {"q": query, "size": num}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get("results", [])[:num]
        results = []
        for it in items:
            results.append({
                "name": it.get("name"),
                "desc": it.get("description"),
                "video_url": (it.get("original_video_url") or ""),
                "thumbnail_url": (it.get("thumbnail_url") or ""),
                "yields": it.get("yields")
            })
        return results
    except Exception:
        return tasty_mock(query, num)

def tasty_mock(query: str, num: int = 3) -> List[Dict[str, Any]]:
    samples = [
        {"name": f"{query.title()} alla Casa", "desc": "Simple, seasonal, waste-smart.", "video_url": "", "thumbnail_url": "", "yields": "Serves 2"},
        {"name": f"{query.title()} Rustica", "desc": "Comforting, high-margin classic.", "video_url": "", "thumbnail_url": "", "yields": "Serves 4"},
        {"name": f"{query.title()} Verde", "desc": "Herb-forward, uses surplus greens.", "video_url": "", "thumbnail_url": "", "yields": "Serves 3"},
    ]
    return samples[:num]

# ---------- OpenAI (assistant response) ----------
def chat_complete(system_prompt: str, user_prompt: str, model: str = None) -> str:
    if not OPENAI_API_KEY:
        return ("(Offline demo)\\n"
                "• Top picks balance margin + waste.\\n"
                "• Use near-expiry greens today.\\n\\n"
                "Dishes:\\n1) Spaghetti Aglio e Olio — pantry-first, fast.\\n"
                "2) Panzanella — uses surplus bread/tomatoes.\\n"
                "3) Frittata di Verdure — absorbs misc. veg.\\n\\n"
                "Waste tips: rotate parsley & basil; pre-chop stems for soffritto.\\n"
                "Pairing: light Chianti works broadly.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=(model or MODEL),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error fallback) {e}\\nReturning concise template:\\n- Use high-urgency items first.\\n- Keep margin ≥ target.\\n- Offer 3 dishes + 2 waste tips."
