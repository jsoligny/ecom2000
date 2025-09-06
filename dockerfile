# ============================
# Dockerfile — GPU FastAPI worker
# ============================

# Runtime CUDA + cuDNN (Ubuntu 22.04)
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dépendances système minimales
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    libjpeg-turbo-progs \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Clone du repo (remplace par ton URL)
WORKDIR /app
RUN git clone https://github.com/jsoligny/ecom2000.git .
# ou si tu veux une branche spécifique :
# RUN git clone -b main https://github.com/ton-organisation/ton-repo.git .

# Installation requirements
# (assure-toi que ton repo a un requirements.txt)
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Variables d’environnement (CUDA activé par défaut)
ENV USE_CUDA=1 \
    REMBG_MODEL=birefnet-massive

# Expose port FastAPI
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "server-gpu:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
