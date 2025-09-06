# handler.py
import base64
from server-gpu import (
    initialize_for_worker, process_bytes_to_dict,
    DEFAULT_MODEL, TARGET_FILL_MIN, TARGET_FILL_MAX,
    AM_FG_T, AM_BG_T, AM_ERODE_PX
)
import urllib.request
from urllib.parse import urlparse

# Warmup au démarrage du worker
initialize_for_worker()

def _fetch_url(url: str, timeout: float = 15.0) -> bytes:
    pr = urlparse(url)
    if pr.scheme not in ("http", "https"):
        raise ValueError("image_url: schéma non supporté (http/https requis)")
    req = urllib.request.Request(url, headers={"User-Agent": "ecom2000/worker/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

def handler(event):
    """
    Runpod Serverless contract:
    event = {"input": {...}}
    Champs reconnus:
      - image_b64 (prioritaire) ou image_url
      - fmt, fill, model, no_matting, am_fg, am_bg, am_erode,
        refine_erode, refine_feather, no_decontaminate, no_yolo,
        fill_min, fill_max
    """
    inp = (event or {}).get("input", {}) or {}

    # Charger l'image
    if "image_b64" in inp and inp["image_b64"]:
        data = base64.b64decode(inp["image_b64"])
    elif "image_url" in inp and inp["image_url"]:
        data = _fetch_url(inp["image_url"])
    else:
        return {"error": "Fournir 'image_b64' ou 'image_url' dans input."}

    # Paramètres avec valeurs par défaut alignées sur ton API
    params = {
        "fmt":       (inp.get("fmt") or "jpeg").lower(),
        "fill":      float(inp.get("fill") or 0.82),
        "model":     inp.get("model") or DEFAULT_MODEL,
        "no_matting": bool(inp.get("no_matting") or False),
        "am_fg":     int(inp.get("am_fg") or AM_FG_T),
        "am_bg":     int(inp.get("am_bg") or AM_BG_T),
        "am_erode":  int(inp.get("am_erode") or AM_ERODE_PX),
        "refine_erode":   int(inp.get("refine_erode") or 1),
        "refine_feather": int(inp.get("refine_feather") or 1),
        "no_decontaminate": bool(inp.get("no_decontaminate") or False),
        "no_yolo":   bool(inp.get("no_yolo") or False),
        "fill_min":  float(inp.get("fill_min") or TARGET_FILL_MIN),
        "fill_max":  float(inp.get("fill_max") or TARGET_FILL_MAX),
    }

    try:
        out = process_bytes_to_dict(data, **params)
        return out
    except Exception as e:
        # Log côté Runpod et retourne un message simple
        return {"error": str(e)}

# Entrée serverless Runpod
import runpod
runpod.serverless.start({"handler": handler})
