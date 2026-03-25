FROM python:3.12-slim

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# set working dir
WORKDIR /app

# Copy project
COPY . .

# Install Python deps
RUN pip install -no-cache-dir -r requirements.txt

RUN mkdir -p vectorstore logs

# Expose FASTAPI port
EXPOSE 8000