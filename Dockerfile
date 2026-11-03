# Lightweight inference image for Google Cloud Run (NumPy + Flask only)
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY reviewintel ./reviewintel
COPY webapp ./webapp
COPY models ./models
COPY outputs/results.json ./outputs/results.json
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 webapp.app:app
