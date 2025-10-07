FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create data directory
RUN mkdir -p data/logs

# Expose port
EXPOSE 3001

# Run the application
CMD ["uvicorn", "src.gatewayx.server:app", "--host", "0.0.0.0", "--port", "3001"]
