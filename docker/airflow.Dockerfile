FROM apache/airflow:2.11.0-python3.12
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 python3-venv && rm -rf /var/lib/apt/lists/*
# Keep the ML dependency resolver separate from Airflow's constrained environment.
RUN python -m venv /opt/ml && /opt/ml/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml /tmp/brainscan/pyproject.toml
RUN /opt/ml/bin/pip install --no-cache-dir '/tmp/brainscan[training,tracking]'
COPY --chown=airflow:root src /app/src
COPY --chown=airflow:root airflow/dags /opt/airflow/dags
ENV PYTHONPATH=/app CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 YOLO_CONFIG_DIR=/tmp/ultralytics
RUN mkdir -p /app/models/candidates /opt/airflow/state && chown -R airflow:root /app/models /opt/airflow/state
USER airflow
CMD ["standalone"]
