# logic.py
from __future__ import annotations
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

PREFERRED_MENU_FILES = [
    "menu_with_viability.csv",
    "data/menu_with_viability.csv",
    "authentic_menu.csv",
    "data/authentic_menu.csv",
]

PREFERRED_INV_FILES = [
    "authentic_inventory.csv",
    "data/authentic_inventory.csv",
]

def load_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_data(menu_path: Optional[str] = None, inv_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    menu_p = menu_path or load_first_existing(PREFERRED_MENU_FILES)
    inv_p = inv_path or load_first_existing(PREFERRED_INV_FILES)
    menu = pd.read_csv(menu_p) if menu_p else pd.DataFrame()
    inv = pd.read_csv(inv_p) if inv_p else pd.DataFrame()
    return normalize_menu(menu), normalize_inventory(inv)

def normalize_menu(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    rename = {}
    low = {c.lower(): c for c in out.columns}
    if "dish" in low and "dish_name" not in out.columns:
        rename[low["dish"]] = "dish_name"
    if "name" in low and "dish_name" not in out.columns:
        rename[low["name"]] = "dish_name"
    if "ingredients_list" in low and "ingredients" not in out.columns:
        rename[low["ingredients_list"]] = "ingredients"
    if "ingredient_list" in low and "ingredients" not in out.columns:
        rename[low["ingredient_list"]] = "ingredients"
    if rename:
        out.rename(columns=rename, inplace=True)
    if "dish_name" not in out.columns:
        out["dish_name"] = out.get("name", "")
    if "ingredients" not in out.columns:
        out["ingredients"] = ""
    if "priority_score" not in out.columns:
        out["priority_score"] = 0.5
    if "margin_pct" not in out.columns:
        price = out.get("price")
        cost = out.get("cost")
        if price is not None and cost is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                out["margin_pct"] = np.where(
                    (price.astype(float) > 0) & cost.notna(),
                    (price.astype(float) - cost.astype(float)) / price.astype(float),
                    0.0,
                )
        else:
            out["margin_pct"] = 0.0
    return out

def normalize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    rename = {}
    for c in list(out.columns):
        lc = c.lower().strip()
        if lc in {"item", "name"}: rename[c] = "ingredient"
        elif lc in {"qty", "stock", "on_hand"}: rename[c] = "quantity"
        elif lc in {"par", "parlevel"}: rename[c] = "par_level"
        elif lc in {"expiry_date", "best_before", "expires"}: rename[c] = "expiry"
        elif lc in {"surplus", "overstock", "excess_qty"}: rename[c] = "surplus_qty"
    if rename:
        out.rename(columns=rename, inplace=True)
    for col, default in [
        ("ingredient", ""), ("quantity", 0.0), ("unit", ""), ("par_level", 0.0),
        ("expiry", None), ("surplus_qty", 0.0)
    ]:
        if col not in out.columns:
            out[col] = default
    out["expiry_dt"] = _parse_date_series(out["expiry"])
    today = datetime.today().date()
    out["days_to_expiry"] = out["expiry_dt"].apply(lambda d: (d.date() - today).days if pd.notna(d) else 9999)
    out["below_par"] = out["quantity"].astype(float) < out["par_level"].astype(float).clip(lower=0)
    out["overstock_flag"] = out["surplus_qty"].astype(float).fillna(0) > 0
    out["over_par2x"] = out["quantity"].astype(float) > (out["par_level"].astype(float).fillna(0) * 2)
    out["is_surplus"] = out[["overstock_flag", "over_par2x"]].any(axis=1)
    return out

def _parse_date_series(s: pd.Series) -> pd.Series:
    def _p(x):
        if pd.isna(x): return pd.NaT
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
            try: return datetime.strptime(str(x), fmt)
            except Exception: continue
        try: return pd.to_datetime(x, errors="coerce")
        except Exception: return pd.NaT
    return s.apply(_p)

def tokenize_ingredients(s: str) -> List[str]:
    return [t.strip().lower() for t in str(s or "").replace(";", ",").split(",") if t.strip()]

def availability_ratio(ingredients: List[str], inv: pd.DataFrame) -> float:
    if not ingredients or inv.empty: return 0.0
    names = set(inv["ingredient"].astype(str).str.lower().str.strip())
    have = sum(1 for i in ingredients if i in names)
    return have / max(1, len(ingredients))

def min_days_to_expiry_for(ingredients: List[str], inv: pd.DataFrame) -> int:
    if not ingredients or inv.empty: return 9999
    df = inv.copy(); df["_norm"] = df["ingredient"].astype(str).str.lower().str.strip()
    days = []
    for i in ingredients:
        row = df.loc[df["_norm"] == i.lower().strip()]
        if not row.empty: days.append(int(row["days_to_expiry"].iloc[0]))
    return min(days) if days else 9999

def expiry_urgency(ingredients: List[str], inv: pd.DataFrame, horizon: int = 10) -> float:
    md = min_days_to_expiry_for(ingredients, inv)
    if md >= horizon: return 0.0
    return max(0.0, (horizon - md) / float(horizon))

def surplus_signal(ingredients: List[str], inv: pd.DataFrame) -> float:
    if not ingredients or inv.empty: return 0.0
    df = inv.copy(); df["_norm"] = df["ingredient"].astype(str).str.lower().str.strip()
    flags = []
    for i in ingredients:
        row = df.loc[df["_norm"] == i.lower().strip()]
        flags.append(bool(row["is_surplus"].iloc[0]) if not row.empty else False)
    return float(sum(flags)) / max(1, len(flags))

def query_match(query: str, dish_name: str, ingredients: str) -> float:
    if not query: return 0.0
    target = f"{dish_name} {ingredients}".lower()
    q = query.lower().strip()
    if fuzz: return fuzz.token_set_ratio(q, target) / 100.0
    tq = set(q.split()); tt = set(target.split())
    return len(tq & tt) / max(1, len(tq))

def margin_norm(margin_pct: float) -> float:
    if pd.isna(margin_pct): return 0.0
    return float(np.clip(float(margin_pct), 0.0, 1.0))

def compute_scores(menu: pd.DataFrame, inv: pd.DataFrame, query_text: str = "",
                   w_query: float = 1.5, w_expiry: float = 1.2, w_surplus: float = 1.0,
                   w_margin: float = 1.0, w_avail_penalty: float = 3.0, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for _, r in menu.iterrows():
        ing = tokenize_ingredients(r.get("ingredients", ""))
        avail = availability_ratio(ing, inv)
        urg = expiry_urgency(ing, inv, horizon=10)
        sur = surplus_signal(ing, inv)
        mrg = margin_norm(r.get("margin_pct", 0.0))
        qry = query_match(query_text, r.get("dish_name", ""), r.get("ingredients", ""))
        base = float(r.get("priority_score", 0.5))
        score = (base + w_query * qry + w_expiry * urg + w_surplus * sur + w_margin * mrg) - w_avail_penalty * (1.0 - avail)
        rows.append({
            "dish_name": r.get("dish_name", ""),
            "ingredients": r.get("ingredients", ""),
            "score": float(score),
            "base": base, "query": float(qry), "urgency": float(urg),
            "surplus": float(sur), "margin": float(mrg), "available_ratio": float(avail),
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df.head(top_n)

def insight_top_expiring(inv: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if inv.empty or "days_to_expiry" not in inv.columns:
        return pd.DataFrame(columns=["ingredient", "days_to_expiry"])
    cols = [c for c in ["ingredient", "days_to_expiry", "quantity", "unit"] if c in inv.columns]
    return inv.sort_values("days_to_expiry").head(n)[cols]

def insight_dishes_86_soon(menu: pd.DataFrame, inv: pd.DataFrame, window_max_days: int = 5) -> pd.DataFrame:
    if menu.empty or inv.empty:
        return pd.DataFrame(columns=["dish", "min_days"])
    rows = []
    for _, r in menu.iterrows():
        ing = tokenize_ingredients(r.get("ingredients", ""))
        md = min_days_to_expiry_for(ing, inv)
        if md <= window_max_days:
            rows.append({"dish": r.get("dish_name", ""), "min_days": int(md)})
    return pd.DataFrame(rows).sort_values("min_days")

def insight_high_margin(menu: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    if menu.empty or "margin_pct" not in menu.columns:
        return pd.DataFrame(columns=["dish_name", "margin_pct"])
    out = menu.loc[menu["margin_pct"] >= float(threshold)][["dish_name", "margin_pct"]]
    return out.sort_values("margin_pct", ascending=False)

def insight_low_stock(inv: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
    if inv.empty or "quantity" not in inv.columns:
        return pd.DataFrame(columns=["ingredient", "quantity", "par_level"])
    cols = [c for c in ["ingredient", "quantity", "par_level"] if c in inv.columns]
    return inv.loc[inv["quantity"].astype(float) <= float(threshold)][cols].sort_values("quantity")

def insight_overstock(inv: pd.DataFrame, qty_threshold: float = 20) -> pd.DataFrame:
    if inv.empty or "quantity" not in inv.columns:
        return pd.DataFrame(columns=["ingredient", "quantity", "par_level"])
    cols = [c for c in ["ingredient", "quantity", "par_level"] if c in inv.columns]
    return inv.loc[inv["quantity"].astype(float) >= float(qty_threshold)][cols].sort_values("quantity", ascending=False)
