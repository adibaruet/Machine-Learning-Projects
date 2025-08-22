FROM python:3.11-slim

# Prevent .pyc files and ensure unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and training script
COPY app.py train.py ./

# Build the model at image build time (bakes titanic_model.joblib into the image)
RUN python train.py

EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app:app", "--host","0.0.0.0", "--port","8000"]
