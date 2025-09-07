# ============================
# Dockerfile — GPU FastAPI worker (fix OpenCV/GLib)
# ============================

FROM nvidia/cuda:13.0.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache des modèles rembg
    U2NET_HOME=/opt/models/u2net \
    # Petites optimisations d'exécution
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=0 \
    PYTHONOPTIMIZE=1

# --- Dépendances système minimales (OpenCV + SSL + Git) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# --- Virtualenv ---
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Dépendances Python minimales (GPU, sans Torch/YOLO) ---
# IMPORTANT : NumPy 1.x avant ORT pour l’ABI
RUN python -m pip install --upgrade pip \
 && python -m pip install "numpy==1.26.4" \
 && python -m pip install onnxruntime-gpu==1.18.1 \
 && python -m pip install "rembg[birefnet]==2.*" \
 && python -m pip install "opencv-python-headless==4.10.*" \
 && python -m pip install pillow==10.* pydantic==2.* fastapi==0.115.* python-multipart \
 && python -m pip install "runpod==1.*"

# --- Pré-télécharger BiRefNet massive AU BUILD (pas de réseau au runtime) ---
RUN mkdir -p "$U2NET_HOME"
RUN python - <<'PY'
import os, sys, traceback
print("=== Preload BiRefNet (build-time) ===")
try:
    import numpy as np, onnxruntime as ort
    print("numpy:", np.__version__)
    print("onnxruntime:", ort.__version__)
    print("U2NET_HOME:", os.environ.get("U2NET_HOME"))
    from rembg import new_session
    # Au build on force CPU provider (pas de GPU dans le contexte build)
    s = new_session("birefnet-massive", providers=["CPUExecutionProvider"])
    p = os.environ["U2NET_HOME"]
    print("Files in U2NET_HOME:", os.listdir(p))
    print("Preload OK.")
except Exception as e:
    print("Preload FAILED:", e)
    traceback.print_exc()
    sys.exit(1)
PY

# Vérification explicite (échoue si le .onnx n'est pas présent)
RUN test -s "$U2NET_HOME/birefnet-massive.onnx" || (echo "birefnet-massive.onnx introuvable" && ls -la "$U2NET_HOME" && exit 1)

# --- Code de l'app ---
WORKDIR /app

# Cache-bust pour re-cloner quand tu veux invalider le cache
ARG REPO_URL="https://github.com/jsoligny/ecom2000.git"
ARG REPO_REF="main"
ARG CACHE_BUST=dev
RUN echo "CACHE_BUST=$CACHE_BUST"

# Clone repo
RUN git clone --depth 1 -b ${REPO_REF} ${REPO_URL} .

# (Optionnel mais conseillé) Vérifier que server.py et handler.py existent
RUN test -f server.py || (echo "server.py manquant à la racine" && ls -la && exit 1)
RUN test -f handler.py || (echo "handler.py manquant à la racine" && ls -la && exit 1)

# Installer d'éventuelles deps du projet (en forçant numpy<2 via un constraints)
# Si ton requirements.txt est vide/inexistant, cette étape passe en no-op.
RUN echo "numpy==1.26.4" > /tmp/constraints.txt \
 && (test -f requirements.txt && python -m pip install -r requirements.txt -c /tmp/constraints.txt || true)

# (Optionnel) Compiler en .pyc pour accélérer les imports
# NB: Python système 3.10 sur Ubuntu 22.04 → adapte le chemin si besoin.
RUN python -m compileall -q /app /opt/venv/lib/python3.10/site-packages || true

# --- Démarrage du Worker Runpod ---
# PAS d'uvicorn ici, on lance le handler (serverless)
CMD ["python", "-u", "handler.py"]
