# ============================
# Dockerfile — GPU FastAPI worker (fix minimal)
# ============================

FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-venv python3-dev \
    build-essential libjpeg-turbo-progs libgl1 \
 && rm -rf /var/lib/apt/lists/*

# venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pin numpy<2 puis GPU libs
RUN python -m pip install --upgrade pip \
 && python -m pip install "numpy==1.26.4" \
 && python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
 && python -m pip install onnxruntime-gpu==1.18.1

# ---- Clone du dépôt ----
WORKDIR /app
RUN git clone https://github.com/jsoligny/ecom2000.git .

# Sanity check (facultatif mais utile en CI)
RUN ls -la && test -f requirements.txt || (echo "requirements.txt manquant" && exit 1)

# ---- Installer les deps du projet (après clone !) ----
# Si ton requirements.txt essaye de ré-installer numpy>=2, crée un constraints temporaire :
RUN echo "numpy==1.26.4" > /tmp/constraints.txt \
 && python -m pip install -r requirements.txt -c /tmp/constraints.txt

ENV USE_CUDA=1 \
    REMBG_MODEL=birefnet-massive \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 80
CMD ["uvicorn", "server-gpu:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]

