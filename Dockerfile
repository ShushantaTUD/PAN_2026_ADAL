# FROM python:3

# ADD script.py /script.py
# ADD requirements.txt /requirements.txt

# RUN pip3 install -r /requirements.txt

# ENTRYPOINT [ "python3", "/script.py", "-i", "$inputDataset", "-o", "$outputDir" ]x

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    MODEL_DIR=/opt/model

WORKDIR /app

# System deps (git is occasionally needed by HF tools; keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# --- Bake the model into the image at build time (no network at runtime) ---
# We temporarily allow online HF during build, then disable it.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Shushant/ADAL-Detector-PANCLEF",
    local_dir="/opt/model",
    local_dir_use_symlinks=False,
)
print("Model snapshot complete.")
PY

# Copy the inference script
COPY script.py /app/script.py

# Default command — TIRA passes -i and -o at run time.
ENTRYPOINT ["python3", "/app/script.py"]