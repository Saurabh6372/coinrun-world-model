FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to save space (Hugging Face Free Tier is CPU only)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the project files
COPY . .

# Install the project and its dependencies
RUN pip install --no-cache-dir -e .

# Hugging Face exposes port 7860 by default
EXPOSE 7860

# We use uvicorn directly instead of 'wm demo' to guarantee we bind to 0.0.0.0 and port 7860
CMD ["uvicorn", "coinrun_world_model.demo.server:app", "--host", "0.0.0.0", "--port", "7860"]
