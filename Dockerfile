# # Use a lightweight Python base image
# FROM python:3.9-slim-buster

# # Set the working directory inside the container
# WORKDIR /app

# # Copy requirements first (so Docker can cache installs)
# COPY requirements.txt .

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy only necessary project files
# COPY app.py whitelisting.py config.json ./
# COPY model/ ./model/
# COPY data/ ./data/

# # Download NLTK data during build
# RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# # Expose FastAPI port
# EXPOSE 8000

# # Run FastAPI app with Uvicorn
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Use an official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]