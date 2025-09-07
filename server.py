# =============================
# server.py — FastAPI worker (logs + timing)
# =============================

from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware

import base64
import io
import os
import torch
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image, ImageFilter, ImageCms, ImageOps

# --- rembg
from rembg import remove, new_session

# ... garde tes imports actuels SAUF ceci :
# --- YOLO (optionnel)
# supprime l'import au top et remplace par une importation tardive dans ensure_yolo()
_YOLO_IMPORT_OK = True  # flag

os.environ.setdefault("U2NET_HOME", "/opt/models/u2net")

# =========================
#  CONFIG LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
#      PARAMÈTRES GLOBAUX
# =========================
TARGET_SIZE = 2000
TARGET_FILL_MIN = 0.80
TARGET_FILL_MAX = 0.85
CANVAS_COLOR_RGB = (255, 255, 255)

AM_ENABLED = True
AM_FG_T = 240
AM_BG_T = 10
AM_ERODE_PX = 2

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0  # 4:4:4

DEFAULT_MODEL = os.environ.get("REMBG_MODEL", "birefnet-massive")

# =========================
#      ÉTAT GLOBAL
# =========================
class GlobalState:
    rembg_session = None
    yolo_model = None

gstate = GlobalState()


app = FastAPI(title="ecom-2000 worker", version="1.0.0")

# Ajoute ce bloc
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://51.255.71.149:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,           # True si tu utilises des cookies/headers d'auth
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)



# Ajoute ce flag global
USE_CUDA = bool(int(os.environ.get("USE_CUDA", "1")))

def _onnx_providers():
    if USE_CUDA:
        # CUDA puis fallback CPU
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
# =========================
#      OUTILS GÉNÉRAUX
# =========================

def _to_srgb(img: Image.Image) -> Image.Image:
    try:
        if "icc_profile" in img.info:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(img.info["icc_profile"]))
            dst_profile = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(img, src_profile, dst_profile, outputMode=img.mode)
    except Exception as e:
        logger.debug(f"Conversion sRGB ignorée (profil ICC) : {e}")
    return img


def load_image_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    img = _to_srgb(img)
    return img


def np_bbox_from_alpha(alpha: Image.Image, thr: int = 10) -> Tuple[int, int, int, int]:
    a = np.array(alpha, dtype=np.uint8)
    mask = a > thr
    if not mask.any():
        return (0, 0, alpha.width, alpha.height)
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

# =========================
#     DÉTECTION OBJETS
# =========================



def ensure_yolo(model_name: str = "yolov8n.pt"):
    global _YOLO_IMPORT_OK
    if not _YOLO_IMPORT_OK:
        return None
    if gstate.yolo_model is None:
        try:
            # import tardif -> ne coûte rien si no_yolo=True
            from ultralytics import YOLO
        except Exception as e:
            logger.warning(f"Ultralytics indisponible: {e}")
            _YOLO_IMPORT_OK = False
            return None
        # ... suite de ta fonction inchangée ...



def detect_bbox_yolo(rgb: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    mdl = ensure_yolo()
    if mdl is None:
        return None
    try:
        t0 = time.perf_counter()
        res = mdl.predict(source=np.asarray(rgb), verbose=False)
        dt = time.perf_counter() - t0
        if not res:
            logger.info(f"YOLO: aucune sortie (en {dt:.3f}s)")
            return None
        boxes = getattr(res[0], "boxes", None)
        if boxes is None or boxes.shape[0] == 0:
            logger.info(f"YOLO: aucune bbox (en {dt:.3f}s)")
            return None
        max_area, best = 0, None
        for b in boxes.xyxy.cpu().numpy():
            x0, y0, x1, y1 = map(float, b[:4])
            area = (x1 - x0) * (y1 - y0)
            if area > max_area:
                max_area, best = area, (int(x0), int(y0), int(x1), int(y1))
        logger.info(f"YOLO: bbox={best} (en {dt:.3f}s)")
        return best
    except Exception as e:
        logger.warning(f"YOLO: exception lors de la détection : {e}")
        return None

# =========================
#  SEGMENTATION / MATTING
# =========================

def ensure_rembg_session(model_name: str = DEFAULT_MODEL):
    if gstate.rembg_session is None:
        t0 = time.perf_counter()
        logger.info(f"Initialisation rembg: modèle='{model_name}'…")
        # >>> GPU via onnxruntime-gpu
        gstate.rembg_session = new_session(model_name, providers=_onnx_providers())
        logger.info(f"Session rembg prête en {time.perf_counter() - t0:.3f}s")
    return gstate.rembg_session


def cutout_with_rembg(
    img: Image.Image,
    model_name: str = DEFAULT_MODEL,
    alpha_matting: bool = AM_ENABLED,
    am_fg: int = AM_FG_T,
    am_bg: int = AM_BG_T,
    am_erode: int = AM_ERODE_PX,
) -> Image.Image:
    session = ensure_rembg_session(model_name)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    t0 = time.perf_counter()
    out = remove(
        buf.getvalue(),
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=am_fg,
        alpha_matting_background_threshold=am_bg,
        alpha_matting_erode_size=am_erode,
        post_process_mask=True,
    )
    logger.info(f"rembg: segmentation effectuée en {time.perf_counter() - t0:.3f}s")
    return Image.open(io.BytesIO(out)).convert("RGBA")


def refine_alpha_soft(img_rgba: Image.Image, erode: int = 1, feather: int = 1) -> Image.Image:
    t0 = time.perf_counter()
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")
    r, g, b, a = img_rgba.split()
    if erode > 0:
        a = a.filter(ImageFilter.MinFilter(2 * erode + 1))
    if feather > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feather))
    out = Image.merge("RGBA", (r, g, b, a))
    logger.info(f"Affinage alpha: erode={erode}, feather={feather} en {time.perf_counter() - t0:.3f}s")
    return out


def decontaminate_edge_to_bg(img_rgba: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    t0 = time.perf_counter()
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")
    premult = Image.new("RGBA", img_rgba.size, bg + (255,))
    premult.alpha_composite(img_rgba)
    final = Image.new("RGBA", img_rgba.size, bg + (255,))
    final.alpha_composite(premult)
    logger.info(f"Décontamination bords sur bg={bg} en {time.perf_counter() - t0:.3f}s")
    return final

# =========================
#     COMPOSITION / QA
# =========================
@dataclass
class Placement:
    scale: float
    new_w: int
    new_h: int
    off_x: int
    off_y: int


def compose_on_white_square(
    obj_rgba: Image.Image,
    target_px: int = TARGET_SIZE,
    fill_ratio: float = 0.82,
    bg_color: Tuple[int, int, int] = CANVAS_COLOR_RGB,
    prefer_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> tuple[Image.Image, Placement]:
    t0 = time.perf_counter()
    if obj_rgba.mode != "RGBA":
        obj_rgba = obj_rgba.convert("RGBA")

    alpha = obj_rgba.split()[3]
    x0, y0, x1, y1 = prefer_bbox if prefer_bbox else np_bbox_from_alpha(alpha)

    obj_cropped = obj_rgba.crop((x0, y0, x1, y1))
    ow, oh = obj_cropped.size

    target_content = max(1, int(round(target_px * fill_ratio)))
    longest = max(1, max(ow, oh))
    scale = target_content / float(longest)
    new_w = max(1, int(round(ow * scale)))
    new_h = max(1, int(round(oh * scale)))
    obj_resized = obj_cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_px, target_px), bg_color)

    off_x = (target_px - new_w) // 2
    off_y = (target_px - new_h) // 2

    premult = Image.new("RGBA", obj_resized.size, bg_color + (255,))
    premult.alpha_composite(obj_resized)
    canvas.paste(premult.convert("RGB"), (off_x, off_y))

    logger.info(
        f"Composition: fill={fill_ratio:.3f}, new=({new_w}x{new_h}), "
        f"offset=({off_x},{off_y}) en {time.perf_counter() - t0:.3f}s"
    )
    return canvas, Placement(scale, new_w, new_h, off_x, off_y)


class QAReport(BaseModel):
    ok_size: bool
    ok_background_pure_white: bool
    ok_fill_ratio: bool
    fill_ratio_effective: float
    dimensions: Tuple[int, int]
    object_box_pixels: Tuple[int, int]
    offset_xy: Tuple[int, int]


def qa_check(
    final_rgb: Image.Image,
    placement: Placement,
    bg_color: Tuple[int, int, int] = CANVAS_COLOR_RGB,
    target_px: int = TARGET_SIZE,
    fill_min: float = TARGET_FILL_MIN,
    fill_max: float = TARGET_FILL_MAX,
    bg_tol: int = 1,  # 254 pour JPEG
) -> QAReport:
    t0 = time.perf_counter()
    w, h = final_rgb.size
    ok_size = (w == target_px and h == target_px)

    arr = np.asarray(final_rgb)
    mask_canvas = np.ones((h, w), dtype=bool)
    mask_canvas[placement.off_y:placement.off_y+placement.new_h,
                placement.off_x:placement.off_x+placement.new_w] = False
    bg = arr[mask_canvas]

    lo = 255 - int(bg_tol)
    ok_bg = bool((bg[:, 0] >= lo).all() and (bg[:, 1] >= lo).all() and (bg[:, 2] >= lo).all())

    fill_ratio_eff = max(placement.new_w, placement.new_h) / float(target_px)
    ok_fill = (fill_ratio_eff >= fill_min) and (fill_ratio_eff <= fill_max)

    logger.info(
        f"QA: size_ok={ok_size}, bg_white={ok_bg}, fill_ok={ok_fill}, "
        f"fill_eff={fill_ratio_eff:.4f} en {time.perf_counter() - t0:.3f}s"
    )

    return QAReport(
        ok_size=ok_size,
        ok_background_pure_white=ok_bg,
        ok_fill_ratio=ok_fill,
        fill_ratio_effective=round(float(fill_ratio_eff), 4),
        dimensions=(w, h),
        object_box_pixels=(placement.new_w, placement.new_h),
        offset_xy=(placement.off_x, placement.off_y),
    )

# =========================
#          I/O
# =========================

def encode_image(img_rgb: Image.Image, fmt: str) -> tuple[str, bytes]:
    fmt = fmt.lower()
    buf = io.BytesIO()
    t0 = time.perf_counter()
    if fmt in ("jpeg", "jpg"):
        img_rgb.convert("RGB").save(
            buf,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
            optimize=True,
            progressive=True
        )
        mime = "image/jpeg"
    elif fmt == "png":
        img_rgb.convert("RGB").save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        raise ValueError("format must be jpeg|jpg|png")
    logger.info(f"Encodage {fmt} en {time.perf_counter() - t0:.3f}s (taille={buf.tell()/1024:.1f} KiB)")
    return mime, buf.getvalue()

# =========================
#        ENDPOINTS
# =========================

@app.on_event("startup")
def _startup():
    logger.info(f"Démarrage serveur FastAPI — init rembg '{DEFAULT_MODEL}', USE_CUDA={USE_CUDA}")
    # Moins de threads Python quand on a un GPU
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    ensure_rembg_session(DEFAULT_MODEL)



@app.get("/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL, "yolo": bool(_YOLO_IMPORT_OK)}


class ProcessResponse(BaseModel):
    report: QAReport
    image_mime: str
    image_b64: str  # image encodée base64


@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(
    file: UploadFile = File(...),
    fmt: str = Form("jpeg"),
    fill: float = Form(0.82),
    model: str = Form(DEFAULT_MODEL),
    no_matting: bool = Form(False),
    am_fg: int = Form(AM_FG_T),
    am_bg: int = Form(AM_BG_T),
    am_erode: int = Form(AM_ERODE_PX),
    refine_erode: int = Form(1),
    refine_feather: int = Form(1),
    no_decontaminate: bool = Form(False),
    no_yolo: bool = Form(False),
    fill_min: float = Form(TARGET_FILL_MIN),
    fill_max: float = Form(TARGET_FILL_MAX),
):
    t_all = time.perf_counter()
    logger.info(
        f"Requête /process: file='{file.filename}', fmt={fmt}, fill={fill}, "
        f"model='{model}', matting={not no_matting}, yolo={not no_yolo}"
    )

    # Chargement image
    t0 = time.perf_counter()
    data = await file.read()
    img = load_image_bytes(data)
    rgb = img.convert("RGB")
    logger.info(f"Chargement + préparation image en {time.perf_counter() - t0:.3f}s")

    # Détection
    prefer_bbox = None
    if not no_yolo:
        logger.info("Détection YOLO…")
        prefer_bbox = detect_bbox_yolo(rgb)

    # Segmentation
    cut = cutout_with_rembg(
        rgb,
        model_name=model,
        alpha_matting=(not no_matting),
        am_fg=am_fg,
        am_bg=am_bg,
        am_erode=am_erode,
    )

    # Affinage / Décontamination
    cut = refine_alpha_soft(cut, erode=refine_erode, feather=refine_feather)
    if not no_decontaminate:
        cut = decontaminate_edge_to_bg(cut, bg=CANVAS_COLOR_RGB)

    # Composition
    final_rgb, placement = compose_on_white_square(
        cut,
        target_px=TARGET_SIZE,
        fill_ratio=fill,
        bg_color=CANVAS_COLOR_RGB,
        prefer_bbox=prefer_bbox,
    )

    # QA
    report = qa_check(
        final_rgb,
        placement,
        bg_color=CANVAS_COLOR_RGB,
        target_px=TARGET_SIZE,
        fill_min=fill_min,
        fill_max=fill_max,
        bg_tol=1 if fmt.lower() in ("jpg", "jpeg") else 0,
    )

    # Encodage
    mime, raw = encode_image(final_rgb, fmt)
    b64 = base64.b64encode(raw).decode("ascii")

    logger.info(f"Traitement total /process en {time.perf_counter() - t_all:.3f}s")
    return ProcessResponse(report=report, image_mime=mime, image_b64=b64)

# Pour lancer :
# uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

# --- AJOUT EN BAS DE server.py ---

def initialize_for_worker():
    """Init lazy pour l'exécution serverless (warmup CUDA/ORT, optionnel YOLO)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    ensure_rembg_session(DEFAULT_MODEL)
    # Si YOLO te sert en serverless, décommente :
    # ensure_yolo()

def process_bytes_to_dict(data: bytes, **kwargs) -> dict:
    """Enrobe process_image_core pour retourner un dict JSON-sérialisable."""
    report, mime, b64 = process_image_core(data=data, **kwargs)
    return {"report": report.model_dump(), "image_mime": mime, "image_b64": b64}



