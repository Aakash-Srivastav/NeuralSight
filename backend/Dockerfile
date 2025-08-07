FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 5000

# Set environment variable to enable Flask
ENV FLASK_APP=app_new.py
ENV FLASK_RUN_HOST=0.0.0.0

# Start Flask server
CMD ["flask", "run"]

# # Run with Gunicorn in production mode
# CMD ["gunicorn", "-b", "0.0.0.0:5000", "app_new:app"]
