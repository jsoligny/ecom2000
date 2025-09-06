# ============================
# Dockerfile — GPU FastAPI worker
# ============================

# Runtime CUDA + cuDNN (Ubuntu 22.04)
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Outils système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-venv python3-dev \
    build-essential \
    libjpeg-turbo-progs libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Virtualenv pour contourner PEP 668 ----
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Versions CUDA/Torch : cu121 (CUDA 12.1+, OK pour base 12.2 runtime)
# Installe d’abord la bonne roue GPU pour Torch, sinon Ultralytics risque d’installer la version CPU.
RUN python -m pip install --upgrade pip \
 && python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# (Optionnel) ORT GPU avant le reste pour fail-fast en cas de mismatch CUDA
RUN python -m pip install onnxruntime-gpu==1.18.1

# ---- Clone ton dépôt ----
WORKDIR /app
# Remplace l’URL/branche par les tiennes. Pour une branche précise :
# RUN git clone -b main https://github.com/ton-organisation/ton-repo.git .
RUN git clone https://github.com/jsoligny/ecom2000.git .

# ---- Dépendances Python du projet ----
# Assure-toi que requirements.txt est à la racine du repo.
RUN test -f requirements.txt || (echo "requirements.txt manquant" && exit 1) \
 && python -m pip install -r requirements.txt

# Vars d’exécution
ENV USE_CUDA=1 \
    REMBG_MODEL=birefnet-massive \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Réseau
EXPOSE 8000

# Démarrage FastAPI
# Ajuste le module si server.py n’est pas à la racine ou si l’app s’appelle autrement
CMD ["uvicorn", "server-gpu:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
