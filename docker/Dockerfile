FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

# OS deps needed by opencv/pillow/scipy stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# keep requirements copy separate for layer caching
COPY requirements-intel.txt requirements.txt
COPY setup.py setup.py

# install base dependencies first (without editable package)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# copy source and install package
COPY src src
COPY config config
COPY notebooks notebooks
COPY README.md README.md
RUN pip install -e .

# non-root runtime user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token="]
