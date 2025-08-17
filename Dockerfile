# Lightweight Python base
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs needed by OpenCV at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY templates ./templates
COPY static ./static
COPY models ./models

# App port (hosts like Render/HF will override with $PORT)
ENV PORT=7860
EXPOSE 7860

# Run with Gunicorn in production
CMD ["sh","-c","gunicorn -w 1 -b 0.0.0.0:${PORT:-7860} app:app"]
