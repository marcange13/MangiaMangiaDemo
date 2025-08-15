
import re
import time
from typing import List, Dict, Any, Iterable, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

# === Secrets (APIs) ===
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
TASTY_API_KEY = st.secrets.get("TASTY_API_KEY", "")
USE_TASTY = str(st.secrets.get("USE_TASTY", "true")).lower() == "true"
MODEL = st.secrets.get("MODEL", "gpt-4o-mini")

# === Required menu schema (flexible via coercion) ===
REQUIRED_COLUMNS: List[str] = [
    "dish_name", "price", "cost", "profit_margin_pct", "priority_score",
    "urgency_score", "surplus_score", "ingredients"
]

def validate_schema(cols: Iterable[str]):
    cols_set = set([str(c).strip() for c in cols])
    req_set = set(REQUIRED_COLUMNS)
    missing = sorted(list(req_set - cols_set))
    extra = sorted(list(cols_set - req_set))
    return missing, extra

# -------------------- schema coercion (menu) --------------------
def _norm(s: str) -> str:
    s = re.sub(r'[%]', 'pct', str(s).lower())
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')

SYNONYMS = {
    "dish_name": {"dish_name","dish","name","item","menu_item","title","recipe"},
    "ingredients": {"ingredients","ingredient_list","recipe_ingredients","items","components"},
    "price": {"price","menu_price","sale_price","list_price","price_usd"},
    "cost": {"cost","food_cost","unit_cost","cogs","cost_usd"},
    "profit_margin_pct": {"profit_margin_pct","margin_pct","profit_margin","profit_percent","gross_margin_pct"},
    "priority_score": {"priority_score","priority","viability","score","weighted_score","rank_score"},
    "urgency_score": {"urgency_score","urgency","perishability","expiry_score","spoilage_risk"},
    "surplus_score": {"surplus_score","surplus","overstock","excess_score","inventory_pressure"},
}

def coerce_menu_schema(df: pd.DataFrame) -> pd.DataFrame:
    canon = {k: {_norm(x) for x in v} for k, v in SYNONYMS.items()}
    rename_map = {}
    seen_targets = set()
    for col in df.columns:
        n = _norm(col)
        target = None
        for want, variants in canon.items():
            if n in variants and want not in seen_targets:
                target = want
                seen_targets.add(want)
                break
        if target:
            rename_map[col] = target
    out = df.rename(columns=rename_map)

    # defaults
    for col in ["price", "cost"]:
        if col not in out.columns: out[col] = 0.0

    # profit margin pct
    if "profit_margin_pct" not in out.columns:
        if "price" in out.columns and "cost" in out.columns:
            with pd.option_context("mode.use_inf_as_na", True):
                out["profit_margin_pct"] = (
                    (pd.to_numeric(out["price"], errors="coerce") - pd.to_numeric(out["cost"], errors="coerce"))
                    / pd.to_numeric(out["price"], errors="coerce").replace(0, pd.NA)
                ) * 100
        out["profit_margin_pct"] = pd.to_numeric(out["profit_margin_pct"], errors="coerce").fillna(0.0)

    for col in ["priority_score","urgency_score","surplus_score"]:
        if col not in out.columns: out[col] = 0.0

    if "dish_name" not in out.columns:
        raise ValueError("menu CSV missing a dish/name column I can map to 'dish_name'.")
    if "ingredients" not in out.columns:
        out["ingredients"] = ""

    # cleanup
    num_cols = ["price","cost","profit_margin_pct","priority_score","urgency_score","surplus_score"]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["dish_name"] = out["dish_name"].astype(str)
    out["ingredients"] = out["ingredients"].fillna("").astype(str)
    return out

# -------------------- inventory coercion --------------------
INV_SYNONYMS = {
    "ingredient": {"ingredient","item","name","product"},
    "quantity": {"quantity","qty","on_hand","stock","amount"},
    "unit": {"unit","uom","measure"},
    "expiry_date": {"expiry","expiration","expiry_date","best_before","use_by","expires"},
    "par_level": {"par","par_level","target_stock","parqty"},
    "incoming_qty": {"incoming","incoming_qty","on_order","arriving"}
}

def coerce_inventory_schema(inv: pd.DataFrame) -> pd.DataFrame:
    inv = inv.copy()
    # ingredient
    cand_ing = [c for c in inv.columns if _norm(c) in {_norm(x) for x in INV_SYNONYMS["ingredient"]}]
    if not cand_ing: raise ValueError("Inventory CSV needs an 'ingredient' column (or synonym).")
    inv = inv.rename(columns={cand_ing[0]: "ingredient"})
    # optionals
    for key in ["quantity","unit","expiry_date","par_level","incoming_qty"]:
        if key not in inv.columns:
            cands = [c for c in inv.columns if _norm(c) in {_norm(x) for x in INV_SYNONYMS[key]}]
            if cands: inv = inv.rename(columns={cands[0]: key})
    # normalize
    inv["ingredient_norm"] = inv["ingredient"].astype(str).str.lower().str.strip()
    for col in ["quantity","par_level","incoming_qty"]:
        inv[col] = pd.to_numeric(inv[col], errors="coerce").fillna(0.0) if col in inv.columns else 0.0
    # parse date
    def parse_date(x):
        if pd.isna(x): return pd.NaT
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return pd.to_datetime(str(x), format=fmt, errors="raise")
            except Exception:
                continue
        return pd.to_datetime(x, errors="coerce")
    inv["expiry_date"] = inv["expiry_date"].apply(parse_date) if "expiry_date" in inv.columns else pd.NaT
    return inv

# -------------------- parsing + features --------------------
def _parse_ingredient_list(s: str) -> List[str]:
    if isinstance(s, list):
        return [str(x).strip().lower() for x in s]
    return [x.strip().lower() for x in str(s).split(",") if x.strip()]

def clean_menu_df(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_menu_schema(df)
    df["ingredients_list"] = df["ingredients"].apply(_parse_ingredient_list)
    return df

def load_menu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_menu_df(df)

def load_inventory(path: str) -> pd.DataFrame:
    inv = pd.read_csv(path)
    return coerce_inventory_schema(inv)

def availability_features(menu: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    avail = set(inv["ingredient_norm"].dropna().astype(str).tolist())
    # ingredient -> earliest expiry + quantities
    by_ing = (inv.sort_values("expiry_date")
                .groupby("ingredient_norm")
                .agg({"quantity":"sum","par_level":"max","incoming_qty":"sum","expiry_date":"min"})
                ).reset_index()
    meta = {row["ingredient_norm"]: row for _, row in by_ing.iterrows()}

    def coverage(items: List[str]):
        miss, pres = [], []
        for it in items:
            if it in avail: pres.append(it)
            else: miss.append(it)
        cov = 1.0 - (len(miss) / max(len(items),1))
        return cov, miss, pres

    def days_to_exp(it: str) -> float:
        row = meta.get(it)
        if row is not None and pd.notna(row["expiry_date"]):
            delta = (row["expiry_date"] - pd.Timestamp.now()).days
            return float(delta)
        return np.inf

    def surplus_ratio(it: str) -> float:
        row = meta.get(it)
        if row is None: return 0.0
        par = float(row.get("par_level", 0) or 0.0)
        qty = float(row.get("quantity", 0) or 0.0) + float(row.get("incoming_qty", 0) or 0.0)
        if par <= 0: return 0.0
        return max(0.0, (qty - par) / par)

    menu = menu.copy()
    feats = menu["ingredients_list"].apply(coverage)
    menu["coverage_pct"] = feats.apply(lambda x: x[0])
    menu["missing_ingredients"] = feats.apply(lambda x: x[1])
    menu["present_ingredients"] = feats.apply(lambda x: x[2])
    menu["can_be_made"] = menu["coverage_pct"] >= 0.999

    def dynamic_urgency(items: List[str]) -> float:
        finite_days = [d for d in (days_to_exp(i) for i in items) if np.isfinite(d)]
        if not finite_days: return 0.0
        dmin = max(0.0, min(finite_days))
        return float(max(0.0, min(1.0, (2.0 / max(dmin, 0.01)))))

    def dynamic_surplus(items: List[str]) -> float:
        if not items: return 0.0
        sr = [surplus_ratio(i) for i in items]
        if not sr: return 0.0
        return float(max(0.0, min(1.0, float(np.mean(sr)))))

    menu["dyn_urgency"] = menu["present_ingredients"].apply(dynamic_urgency)
    menu["dyn_surplus"] = menu["present_ingredients"].apply(dynamic_surplus)
    return menu

# -------------------- ranking --------------------
def compute_demo_score(row, weights=None):
    w = {"priority": 0.45, "margin": 0.25, "urgency": 0.15, "surplus": 0.05, "availability": 0.10}
    if weights: w.update(weights)
    margin = row.get("profit_margin_pct", 0) / 100.0
    urg = max(row.get("urgency_score", 0.0), row.get("dyn_urgency", 0.0))
    sur = max(row.get("surplus_score", 0.0), row.get("dyn_surplus", 0.0))
    return (
        w["priority"] * row.get("priority_score", 0.0) +
        w["margin"]   * margin +
        w["urgency"]  * urg +
        w["surplus"]  * sur +
        w["availability"] * row.get("coverage_pct", 0.0)
    ) + (0.06 if row.get("can_be_made", False) else 0.0)

def parse_items_from_query(query: str):
    if not query: return set()
    tokens = set()
    for _, name in re.findall(r"\b(\d+(?:\.\d+)?\s*[a-zA-Z]+)?\s*([a-zA-Z][a-zA-Z\s\-]{2,})\b", query):
        for w in re.split(r"\s+", name.lower()):
            if len(w) >= 3:
                tokens.add(w)
    return tokens

def ingredient_match_boost(text: str, tokens: set) -> float:
    if not tokens or not text: return 0.0
    s = str(text).lower()
    hits = sum(1 for t in tokens if t in s)
    return min(hits * 0.08, 0.24)

def filter_by_keywords(df: pd.DataFrame, query: str) -> pd.DataFrame:
    toks = [t.strip().lower() for t in (query or '').split() if t.strip()]
    if not toks:
        return df
    mask = np.zeros(len(df), dtype=bool)
    name = df["dish_name"].astype(str).str.lower()
    ing  = df["ingredients"].astype(str).str.lower()
    for t in toks:
        mask |= name.str.contains(t, na=False) | ing.str.contains(t, na=False)
    return df[mask] if mask.any() else df

def rank_dishes(df: pd.DataFrame, top_k: int = 5, weights=None, query_tokens: Optional[set] = None) -> pd.DataFrame:
    temp = df.copy()
    temp["demo_score"] = temp.apply(lambda r: compute_demo_score(r, weights), axis=1)
    if query_tokens:
        temp["demo_score"] += temp["ingredients"].apply(lambda s: ingredient_match_boost(s, query_tokens))
        temp["demo_score"] += temp["dish_name"].apply(lambda s: ingredient_match_boost(s, query_tokens))
    temp = temp.sort_values(by=["can_be_made","demo_score","priority_score","profit_margin_pct"], ascending=[False, False, False, False])
    return temp.head(top_k)

# -------------------- entrypoint --------------------
def handle_user_query(menu_df: pd.DataFrame, query: str, top_k: int = 5, weights=None, inventory_df: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    df = clean_menu_df(menu_df)
    if inventory_df is not None and not inventory_df.empty:
        df = availability_features(df, inventory_df)
    else:
        df["coverage_pct"] = 0.0
        df["missing_ingredients"] = [[] for _ in range(len(df))]
        df["present_ingredients"] = [[] for _ in range(len(df))]
        df["can_be_made"] = False
        df["dyn_urgency"] = 0.0
        df["dyn_surplus"] = 0.0

    tokens = parse_items_from_query(query)
    sub = filter_by_keywords(df, query)
    if sub.empty: sub = df
    ranked = rank_dishes(sub, top_k=top_k, weights=weights, query_tokens=tokens)
    return ranked

# -------------------- insights helpers --------------------
def top_expiring_ingredients(inv_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    inv = inv_df.copy()
    if "expiry_date" not in inv.columns:
        return pd.DataFrame(columns=["ingredient","expiry_date","days_to_exp"])
    inv = inv[inv["expiry_date"].notna()].copy()
    inv["days_to_exp"] = (inv["expiry_date"] - pd.Timestamp.now()).dt.days
    inv = inv.sort_values("days_to_exp").head(top_n)
    return inv[["ingredient","expiry_date","days_to_exp"]]

def at_risk_dishes_86(menu_df: pd.DataFrame, inv_df: pd.DataFrame, horizon_days: int = 4) -> pd.DataFrame:
    inv = inv_df.copy()
    if "expiry_date" not in inv.columns:
        return pd.DataFrame(columns=["dish_name","risk_ingredient","days_to_exp","expiry_date"])
    ing_exp = (inv[inv["expiry_date"].notna()]
                 .sort_values("expiry_date")
                 .groupby(inv["ingredient"].astype(str).str.lower().str.strip())
                 .agg({"expiry_date":"min"})
                 .reset_index())
    ing_exp["days_to_exp"] = (ing_exp["expiry_date"] - pd.Timestamp.now()).dt.days
    exp_map = {row["ingredient"]: row for _, row in ing_exp.iterrows()}

    menu = clean_menu_df(menu_df).copy()
    rows = []
    for _, r in menu.iterrows():
        for ing in r["ingredients_list"]:
            if ing in exp_map:
                days = exp_map[ing]["days_to_exp"]
                if days <= horizon_days:
                    rows.append({
                        "dish_name": r["dish_name"],
                        "risk_ingredient": ing,
                        "days_to_exp": float(days),
                        "expiry_date": exp_map[ing]["expiry_date"]
                    })
                    break
    return pd.DataFrame(rows).sort_values("days_to_exp")

def high_margin_specials(ranked_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    df = ranked_df.copy()
    if "can_be_made" in df.columns:
        df = df[df["can_be_made"] == True]
    return df.sort_values(["profit_margin_pct","demo_score"], ascending=False).head(n)[
        ["dish_name","profit_margin_pct","price","cost","coverage_pct"]
    ]

def low_stock_watchlist(inv_df: pd.DataFrame, threshold_pct: float = 0.3) -> pd.DataFrame:
    inv = inv_df.copy()
    if "par_level" not in inv.columns or "quantity" not in inv.columns:
        return pd.DataFrame(columns=["ingredient","quantity","par_level","pct_of_par"])
    inv["pct_of_par"] = inv.apply(lambda r: (r["quantity"] / r["par_level"]) if r["par_level"] else np.nan, axis=1)
    inv = inv[inv["pct_of_par"].notna() & (inv["pct_of_par"] <= threshold_pct)]
    return inv.sort_values("pct_of_par")[["ingredient","quantity","par_level","pct_of_par"]]

def overstock_clearance(inv_df: pd.DataFrame, threshold_pct: float = 0.3) -> pd.DataFrame:
    inv = inv_df.copy()
    if "par_level" not in inv.columns:
        return pd.DataFrame(columns=["ingredient","quantity","par_level","incoming_qty","over_par_pct"])
    inv["over_par_pct"] = inv.apply(
        lambda r: ((r.get("quantity",0)+r.get("incoming_qty",0)-r["par_level"]) / r["par_level"]) if r["par_level"] else np.nan, axis=1
    )
    inv = inv[inv["over_par_pct"].notna() & (inv["over_par_pct"] >= threshold_pct)]
    return inv.sort_values("over_par_pct", ascending=False)[["ingredient","quantity","par_level","incoming_qty","over_par_pct"]]

def overstock_dish_matches(over_df: pd.DataFrame, menu_df: pd.DataFrame) -> pd.DataFrame:
    if over_df.empty:
        return pd.DataFrame(columns=["ingredient","dish_name"])
    menu = clean_menu_df(menu_df).copy()
    rows = []
    overs = set(over_df["ingredient"].astype(str).str.lower().str.strip().tolist())
    for _, r in menu.iterrows():
        ing_list = r["ingredients_list"]
        uses = [i for i in ing_list if i in overs]
        if uses:
            for u in uses:
                rows.append({"ingredient": u, "dish_name": r["dish_name"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["ingredient","dish_name"]).reset_index(drop=True)

def waste_forecast_summary(inv_df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    inv = inv_df.copy()
    if "expiry_date" not in inv.columns:
        return pd.DataFrame(columns=["ingredient","expiry_date","days_to_exp"])
    inv = inv[inv["expiry_date"].notna()].copy()
    inv["days_to_exp"] = (inv["expiry_date"] - pd.Timestamp.now()).dt.days
    inv = inv[inv["days_to_exp"] <= days]
    return inv.sort_values("days_to_exp")[["ingredient","expiry_date","days_to_exp"]]

def weekly_specials_plan(menu_df: pd.DataFrame, inv_df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    # Simple heuristic: pick highest urgency + can_be_made dishes for next N days (diverse ingredients)
    menu = clean_menu_df(menu_df)
    menu = availability_features(menu, coerce_inventory_schema(inv_df.copy()))
    pool = menu[menu["can_be_made"] == True].copy()
    if pool.empty:
        return pd.DataFrame(columns=["day","suggested_dish","why"])
    pool["score"] = pool["dyn_urgency"]*0.6 + pool["dyn_surplus"]*0.2 + (pool["profit_margin_pct"]/100)*0.2
    pool = pool.sort_values("score", ascending=False)
    picks = []
    used_ings = set()
    for d in range(days):
        for _, r in pool.iterrows():
            if not set(r["present_ingredients"]).intersection(used_ings):
                picks.append({
                    "day": f"Day {d+1}",
                    "suggested_dish": r["dish_name"],
                    "why": f"Urgency {r['dyn_urgency']:.2f}, Margin {r['profit_margin_pct']:.0f}%"
                })
                used_ings.update(r["present_ingredients"][:2])
                break
    return pd.DataFrame(picks)

def batch_prep_suggestions(menu_df: pd.DataFrame) -> pd.DataFrame:
    menu = clean_menu_df(menu_df).copy()
    bases = {
        "tomato_sauce": ["tomato","passata","pomodoro","san marzano"],
        "soffritto": ["carrot","celery","onion"],
        "pesto": ["basil","pine nut","parmigiano","olive oil"],
        "ragù_base": ["beef","pork","celery","carrot","onion","tomato"],
    }
    rows = []
    for _, r in menu.iterrows():
        ings = " ".join(r["ingredients_list"])
        for base, keys in bases.items():
            if all(k in ings for k in keys):
                rows.append({"base_prep": base, "dish_name": r["dish_name"]})
                break
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["base_prep","dish_name"])

def social_caption_text(picks: List[str]) -> str:
    if not picks: return "Nonna's specials coming soon… 🍝"
    names = ", ".join(picks[:3])
    return f"Nonna’s specials tonight: {names}! 🍝🍕🍷\nFresh, seasonal, and zero‑waste friendly. #RomaKitchen #MangiaMangia"

def server_script_text(picks: List[str]) -> str:
    if not picks: return "Tonight we’re cooking with what’s fresh and ready — ask me for the daily special!"
    lines = ["Buonasera! Nonna suggests:"]
    for p in picks[:3]:
        lines.append(f"• {p} — fresh, high‑margin, and zero‑waste friendly.")
    lines.append("Would you like a light antipasto to start?")
    return "\n".join(lines)

# -------------------- Tasty API --------------------
def tasty_for_top(ranked_df: pd.DataFrame, max_items: int = 3):
    names = ranked_df["dish_name"].head(max_items).tolist()
    out: Dict[str, Any] = {}
    for n in names:
        out[n] = tasty_search(n, num=2)
        time.sleep(0.1)
    return out

def tasty_search(query: str, num: int = 3) -> List[Dict[str, Any]]:
    if not USE_TASTY or not TASTY_API_KEY:
        return tasty_mock(query, num)
    try:
        url = "https://tasty.p.rapidapi.com/recipes/list"
        headers = {"x-rapidapi-host": "tasty.p.rapidapi.com", "x-rapidapi-key": TASTY_API_KEY}
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

# -------------------- OpenAI (Notes) --------------------
def chat_complete(system_prompt: str, user_prompt: str, model: str = None) -> str:
    if not OPENAI_API_KEY:
        return ("(Offline demo)\n"
                "• Uses today's inventory (availability + earliest expiry + surplus).\n"
                "• Picks prioritize can-be-made dishes and urgency.\n"
                "• Tips: rotate near-expiry herbs; use bread ends for panzanella.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=(model or MODEL),
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=0.6,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error fallback) {e}\nReturning concise template: urgent + available dishes; 2 waste tips."
