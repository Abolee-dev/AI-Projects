# Dockerfile for the IT service analytics chatbot

# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for building PyHive and SASL)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       default-libmysqlclient-dev \
       libsasl2-dev \
       libssl-dev \
       libffi-dev \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Entrypoint
CMD ["python", "main.py"]