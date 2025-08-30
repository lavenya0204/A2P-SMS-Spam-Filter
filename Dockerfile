# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements.txt first for caching Docker layer
COPY requirements.txt .

# Upgrade pip before installing dependencies to avoid install issues
RUN python -m pip install --upgrade pip

# Install dependencies from requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and folders, excluding unnecessary files via .dockerignore
COPY . .

# Download NLTK data during image build (if needed)
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Expose port for FastAPI
EXPOSE 8000

# Use the correct entry point script (update if your main app file is not 'main.py')
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
