# ============================
# Dockerfile — GPU FastAPI worker (fix OpenCV/GLib)
# ============================

FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Outils système + libs requises par OpenCV (cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-venv python3-dev build-essential \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Virtualenv ----
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Important : NumPy 1.x avant ORT (compat ABI), puis Torch GPU cu124 et ORT GPU
RUN python -m pip install --upgrade pip \
 && python -m pip install "numpy==1.26.4" \
 && python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
 && python -m pip install onnxruntime-gpu==1.18.1 \
 && python -m pip install python-multipart

# ---- Clone du dépôt ----
WORKDIR /app
# change cette valeur à chaque build (ex: timestamp, commit, etc.)
ARG CACHE_BUST=v1
RUN echo "CACHE_BUST=$CACHE_BUST" \
 && git clone --depth 1 -b main https://github.com/jsoligny/ecom2000.git .

# Sanity check
RUN ls -la && test -f requirements.txt || (echo "requirements.txt manquant" && exit 1)

# Installer les deps du projet en forçant numpy<2 si besoin
RUN echo "numpy==1.26.4" > /tmp/constraints.txt \
 && python -m pip install -r requirements.txt -c /tmp/constraints.txt

# Vars d’exécution
ENV USE_CUDA=1 \
    REMBG_MODEL=birefnet-massive \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENCV_LOG_LEVEL=ERROR

# Réseau
EXPOSE 80
# ... (toutes tes étapes précédentes inchangées : apt, venv, numpy==1.26.4, torch cu124, onnxruntime-gpu, git clone, pip -r) ...

# Installer la lib Runpod côté worker
RUN python -m pip install "runpod==1.*"

# Si handler.py est dans ton repo, nothing to do.
# Si tu le crées localement à côté du Dockerfile, ajoute:
# COPY handler.py /app/handler.py

# Environnement (GPU)
ENV USE_CUDA=1 REMBG_MODEL=birefnet-massive OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Pas besoin d'EXPOSE ni d'uvicorn ici
CMD ["python", "-u", "handler.py"]

