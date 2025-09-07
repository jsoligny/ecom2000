# ============================
# Dockerfile — GPU FastAPI worker (fix OpenCV/GLib)
# ============================

FROM nvidia/cuda:13.0.0-base-ubuntu24.04


ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/opt/models/u2net \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=0 \ 
    PYTHONOPTIMIZE=1

# libs système minimales + runtime OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dépendances Python GPU minimales (NUMPY 1.x + ORT GPU)
RUN python -m pip install --upgrade pip \
 && python -m pip install "numpy==1.26.4" \
 && python -m pip install onnxruntime-gpu==1.18.1 \
 && python -m pip install pillow==10.* pydantic==2.* fastapi==0.115.* \
    python-multipart

# Pré-télécharger BiRefNet dans l’image (pas de download à l’exécution)
RUN mkdir -p "$U2NET_HOME" \
 && python - <<'PY'
from rembg import new_session
new_session("birefnet-massive", providers=["CPUExecutionProvider"])
print("Preloaded birefnet-massive into U2NET_HOME")
PY

# ---- Code app ----
WORKDIR /app
# Si repo public :
#   on bust le cache clone à chaque build si besoin
ARG CACHE_BUST=$(Get-Date -UFormat %s)
RUN echo "CACHE_BUST=$CACHE_BUST"
RUN git clone --depth 1 -b main https://github.com/jsoligny/ecom2000.git .

# Sinon, copie locale :
# COPY . .

# Installer les deps du projet (sans YOLO/torch)
# IMPORTANT: si ton requirements.txt contient opencv-python, remplace par opencv-python-headless
# et veille à ne pas tirer numpy>=2
RUN echo "numpy==1.26.4" > /tmp/constraints.txt \
 && python -m pip install -r requirements.txt -c /tmp/constraints.txt || true

# Installer la lib Runpod (worker)
RUN python -m pip install "runpod==1.*"

# (optionnel) Compiler les .py en .pyc pour accélérer les imports
RUN python -m compileall -q /app /opt/venv/lib/python3.10/site-packages || true

# Entrée serverless
CMD ["python", "-u", "handler.py"]
