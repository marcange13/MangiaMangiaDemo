# prompts.py
import os, json
from typing import Tuple, List, Dict

def _get_secret(name: str):
    try:
        import streamlit as st
        v = st.secrets.get(name, None)
        if v: return v
    except Exception:
        pass
    return os.getenv(name)

def init_openai():
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key: return None, None
    model = _get_secret("OPENAI_MODEL") or "gpt-4o-mini"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        return client, model
    except Exception:
        try:
            import openai
            openai.api_key = api_key
            return openai, model
        except Exception:
            return None, None

def caption_and_script(dish_name: str, default_wine: str = "Chianti") -> Tuple[str, str]:
    client, model = init_openai()
    if client is None:
        return (f"Today's special: {dish_name}! 🍝 Cozy, fresh, and delicious. #DailySpecial",
                f"Offer a pairing: a glass of {default_wine} or a premium add-on.")
    prompt = (
        "You are a restaurant marketer. Write a fun, punchy Instagram caption (≤25 words) "
        f"for the dish: {dish_name}. Include 1-2 food emojis and a gentle CTA.\n\n"
        "Also provide a 1-2 sentence server upsell script highlighting a pairing and a premium add-on. "
        "Return JSON with keys 'caption' and 'script'."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"You are helpful and concise."},
                      {"role":"user","content":prompt}], temperature=0.7)
        text = resp.choices[0].message.content.strip()
    except Exception:
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role":"system","content":"You are helpful and concise."},
                       {"role":"user","content":prompt}])
            text = resp.output_text.strip()
        except Exception:
            text = ""
    try:
        data = json.loads(text)
        return data.get("caption",""), data.get("script","")
    except Exception:
        parts = text.split("\n",1)
        cap = parts[0][:200] if parts and parts[0] else f"Today's special: {dish_name}! 🍝"
        scr = parts[1] if len(parts)>1 else "Suggest a wine pairing and a premium add-on."
        return cap, scr

def tasty_headers() -> Dict[str,str] | None:
    key = _get_secret("TASTY_API_KEY") or _get_secret("RAPIDAPI_KEY")
    if not key: return None
    return {"x-rapidapi-host":"tasty.p.rapidapi.com","x-rapidapi-key":key}

def tasty_search(query: str, size: int = 3) -> List[Dict[str,str]]:
    headers = tasty_headers()
    if not headers: return []
    try:
        import requests
        url = "https://tasty.p.rapidapi.com/recipes/list"
        params = {"q": query, "from": 0, "size": size}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            out = []
            for it in data.get("results", []):
                out.append({
                    "name": it.get("name"),
                    "description": it.get("description"),
                    "thumbnail_url": it.get("thumbnail_url"),
                    "video_url": it.get("original_video_url"),
                    "total_time_tier": (it.get("total_time_tier") or {}).get("display_tier"),
                })
            return out
    except Exception:
        pass
    return []
