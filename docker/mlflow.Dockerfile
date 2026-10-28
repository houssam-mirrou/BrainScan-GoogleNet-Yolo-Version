FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml .
RUN pip install --no-cache-dir '.[tracking]'
COPY src/mlflow_tracking.py ./src/mlflow_tracking.py
