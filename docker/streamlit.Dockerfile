FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml .
RUN pip install --no-cache-dir '.[ui]'
COPY streamlit/app.py ./streamlit/app.py
CMD ["streamlit", "run", "streamlit/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.maxUploadSize=10"]
