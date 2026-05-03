FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Download DistilBERT weights at build time so container starts fast
RUN python -c "from transformers import DistilBertModel, DistilBertTokenizer; \
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased'); \
    DistilBertModel.from_pretrained('distilbert-base-uncased')"

EXPOSE 8000

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]