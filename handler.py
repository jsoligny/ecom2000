# handler.py
import base64
from server import process_image_core, DEFAULT_MODEL, TARGET_FILL_MIN, TARGET_FILL_MAX

def handler(event):
    inp = event.get("input", {})
    img_b64 = inp.get("image_b64")
    if not img_b64:
        return {"error": "image_b64 manquant"}
    data = base64.b64decode(img_b64)

    report, mime, b64 = process_image_core(
        data=data,
        fmt=inp.get("fmt","jpeg"),
        fill=float(inp.get("fill",0.82)),
        model=inp.get("model", DEFAULT_MODEL),
        no_matting=bool(inp.get("no_matting", False)),
        am_fg=int(inp.get("am_fg", 240)),
        am_bg=int(inp.get("am_bg", 10)),
        am_erode=int(inp.get("am_erode", 2)),
        refine_erode=int(inp.get("refine_erode", 1)),
        refine_feather=int(inp.get("refine_feather", 1)),
        no_decontaminate=bool(inp.get("no_decontaminate", False)),
        no_yolo=bool(inp.get("no_yolo", False)),
        fill_min=float(inp.get("fill_min", TARGET_FILL_MIN)),
        fill_max=float(inp.get("fill_max", TARGET_FILL_MAX)),
    )
    return {"report": report.model_dump(), "image_mime": mime, "image_b64": b64}

# Entrée serverless
import runpod
runpod.serverless.start({"handler": handler})
