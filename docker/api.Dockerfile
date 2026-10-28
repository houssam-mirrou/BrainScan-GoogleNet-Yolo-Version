FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 YOLO_CONFIG_DIR=/tmp/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
# Install CPU-only PyTorch before Ultralytics to avoid CUDA dependencies.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir '.[api]'
COPY api ./api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
