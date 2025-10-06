# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for numpy/scipy and additional dependencies for Gradio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for both FastAPI (8000) and Gradio (7860)
EXPOSE 8000 7860

# Create a startup script to run both services
RUN echo '#!/bin/bash\n\
# Start FastAPI in background\n\
uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
\n\
# Start Gradio in foreground\n\
python gradio_app.py\n\
' > start_services.sh && chmod +x start_services.sh

CMD ["./start_services.sh"]

