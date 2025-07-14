# === Base Image ===
FROM python:3.11-slim

# === System dependencies (for scientific libs, TensorFlow, etc.) ===
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# === Working Directory ===
WORKDIR /app

# === Install Python Dependencies ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Copy All Project Files ===
COPY . .

# === Environment Variables (optional) ===
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# === Expose Port ===
EXPOSE 10000

# === Run the Flask App with Gunicorn ===
# NOTE: "app:app" means app.py contains 'app = Flask(_name_)'
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000","--reload","--timeout","180","--workers","2","--threads","2"]