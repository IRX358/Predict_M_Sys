# === Use Python 3.11 Official Slim Image ===
FROM python:3.11-slim

# === Set Working Directory ===
WORKDIR /app

# === Install Dependencies ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Copy All Files ===
COPY . .

# === Expose the default port for Railway ===
EXPOSE 8080

# === Start with Gunicorn (Timeout 180s) ===
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "180"]
