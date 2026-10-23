FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and starter threat knowledge base
COPY . .

# Expose ports (8501 for Streamlit UI, 8000 for FastAPI)
EXPOSE 8501
EXPOSE 8000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: launch modern Streamlit interface
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
