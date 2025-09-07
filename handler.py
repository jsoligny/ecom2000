# handler.py — Runpod Serverless Worker (image_b64 only)
# Usage (synchrone):
#   POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
#   Body JSON:
#     {
#       "input": {
#         "image_b64": "<BASE64 SANS 'data:...;base64,'>",
#         "fmt": "jpeg" | "png",
#         "fill": 0.82,
#         "no_yolo": true,            # par défaut true (cold start plus court)
#         "no_matting": false,
#         "am_fg": 240,
#         "am_bg": 10,
#         "am_erode": 2,
#         "refine_erode": 1,
#         "refine_feather": 1,
#         "no_decontaminate": false,
#         "fill_min": 0.80,
#         "fill_max": 0.85,
#         "model": "birefnet-massive"
#       }
#     }

import os
import base64
import runpod

# On importe le module local 'server.py' comme une librairie
import server as srv


# -------- Warm-up minimal (charge la session rembg et lit le modèle pré-téléchargé) --------
# Conseil: fixe U2NET_HOME dans l'image (ENV U2NET_HOME=/opt/models/u2net)
try:
    # Tu peux forcer ici un providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    # si ton server.ensure_rembg_session l'accepte; sinon il lira depuis USE_CUDA.
    srv.ensure_rembg_session(srv.DEFAULT_MODEL)
    print("[handler] rembg session initialisée")
except Exception as e:
    print("[handler] Warmup rembg error:", e)


# -------- Helpers robustes --------
def _to_float(v, default):
    try:
        return float(v)
    except Exception:
        return default

def _to_int(v, default):
    try:
        return int(v)
    except Exception:
        return default

def _to_bool(v, default):
    # Accepte booleans, "true"/"false", 0/1
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default

def _clean_b64(b64_str: str) -> str:
    # Supporte un data URL "data:image/jpeg;base64,...."
    if not b64_str:
        return ""
    if "," in b64_str and b64_str.lstrip().startswith("data:"):
        return b64_str.split(",", 1)[1]
    return b64_str

def _build_params(inp: dict) -> dict:
    # Valeurs par défaut alignées sur ton server.py
    fmt = (inp.get("fmt") or "jpeg").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in ("jpeg", "png"):
        fmt = "jpeg"

    # Par défaut on désactive YOLO pour limiter le cold start (tu peux passer no_yolo=false pour forcer)
    no_yolo_default = True

    return {
        "fmt":               fmt,
        "fill":              _to_float(inp.get("fill"), 0.82),
        "model":             inp.get("model") or srv.DEFAULT_MODEL,
        "no_matting":        _to_bool(inp.get("no_matting"), False),
        "am_fg":             _to_int(inp.get("am_fg"), srv.AM_FG_T),
        "am_bg":             _to_int(inp.get("am_bg"), srv.AM_BG_T),
        "am_erode":          _to_int(inp.get("am_erode"), srv.AM_ERODE_PX),
        "refine_erode":      _to_int(inp.get("refine_erode"), 1),
        "refine_feather":    _to_int(inp.get("refine_feather"), 1),
        "no_decontaminate":  _to_bool(inp.get("no_decontaminate"), False),
        "no_yolo":           _to_bool(inp.get("no_yolo"), no_yolo_default),
        "fill_min":          _to_float(inp.get("fill_min"), srv.TARGET_FILL_MIN),
        "fill_max":          _to_float(inp.get("fill_max"), srv.TARGET_FILL_MAX),
    }


# -------- Handler Runpod --------
def handler(event):
    """
    Contrat Runpod:
      event = {"input": {...}}
      Retour: dict sérialisable JSON, ex:
        {"report": {...}, "image_mime": "image/jpeg", "image_b64": "<...>"}
    """
    try:
        inp = (event or {}).get("input", {}) or {}

        b64 = _clean_b64(inp.get("image_b64", ""))
        if not b64:
            return {"error": "image_b64 manquant"}

        try:
            data = base64.b64decode(b64)
        except Exception:
            return {"error": "image_b64 invalide (base64)"}

        params = _build_params(inp)

        # Utilise process_bytes_to_dict si présent (plus simple)
        if hasattr(srv, "process_bytes_to_dict"):
            return srv.process_bytes_to_dict(data, **params)

        # Sinon, fallback via process_image_core
        if hasattr(srv, "process_image_core"):
            report, mime, out_b64 = srv.process_image_core(data=data, **params)
            report_dict = report.model_dump() if hasattr(report, "model_dump") else report
            return {"report": report_dict, "image_mime": mime, "image_b64": out_b64}

        # Dernier recours: message clair si l’API n’existe pas
        return {"error": "Aucune API de traitement trouvée (process_bytes_to_dict ni process_image_core)."}

    except Exception as e:
        # Évite de faire planter le worker : renvoie l’erreur dans output
        return {"error": str(e)}


# Démarrage Runpod Worker
runpod.serverless.start({"handler": handler})
