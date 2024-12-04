# Use Python 3.11-slim image
FROM python:3.11-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Disable output buffering
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ make libffi-dev libssl-dev zlib1g-dev libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt ./

# Install Python dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Add non-root user and set permissions
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Copy application code
COPY . /app

# Set the default command to run the application
CMD ["python", "AutoCut_v3_beta.py"]
