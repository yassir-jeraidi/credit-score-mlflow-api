FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    pip install 'mlflow<2.10' psycopg2-binary && \
    rm -rf /var/lib/apt/lists/*

# Default command (can be overridden)
CMD ["mlflow", "server", "--host", "0.0.0.0"]
