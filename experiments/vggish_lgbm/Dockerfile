FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing (if needed)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run the pipeline
CMD ["python", "experiments/vggish_lgbm/train_model.py"]

