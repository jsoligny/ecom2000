import base64
from server import (
    initialize_for_worker, process_bytes_to_dict,
    DEFAULT_MODEL, TARGET_FILL_MIN, TARGET_FILL_MAX,
    AM_FG_T, AM_BG_T, AM_ERODE_PX
)
import runpod

initialize_for_worker()

def handler(event):
    inp = (event or {}).get("input", {}) or {}
    b64 = inp.get("image_b64")
    if not b64:
        return {"error": "image_b64 manquant"}  # upload obligatoire

    try:
        data = base64.b64decode(b64)
    except Exception:
        return {"error": "image_b64 invalide (base64)"}

    params = {
        "fmt": (inp.get("fmt") or "jpeg").lower(),
        "fill": float(inp.get("fill") or 0.82),
        "model": inp.get("model") or DEFAULT_MODEL,
        "no_matting": bool(inp.get("no_matting") or False),
        "am_fg": int(inp.get("am_fg") or AM_FG_T),
        "am_bg": int(inp.get("am_bg") or AM_BG_T),
        "am_erode": int(inp.get("am_erode") or AM_ERODE_PX),
        "refine_erode": int(inp.get("refine_erode") or 1),
        "refine_feather": int(inp.get("refine_feather") or 1),
        "no_decontaminate": bool(inp.get("no_decontaminate") or False),
        "no_yolo": bool(inp.get("no_yolo") or False),
        "fill_min": float(inp.get("fill_min") or TARGET_FILL_MIN),
        "fill_max": float(inp.get("fill_max") or TARGET_FILL_MAX),
    }

    try:
        return process_bytes_to_dict(data, **params)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
