# Credit Score MLflow API

A production-ready credit scoring API with MLflow experiment tracking, Docker containerization, and Prometheus monitoring.

## 🎯 Project Overview

This project implements a complete MLOps pipeline for a credit scoring system:

- **ML Model**: Random Forest classifier for credit approval prediction
- **API**: FastAPI-based REST API with automatic documentation
- **MLflow**: Experiment tracking and model registry
- **Monitoring**: Prometheus metrics for observability
- **Docker**: Multi-stage builds for optimized containerization

## 📋 Features

- ✅ Synthetic data generation (RGPD compliant)
- ✅ MLflow experiment tracking and model versioning
- ✅ FastAPI REST endpoints with Pydantic validation
- ✅ Single and batch prediction endpoints
- ✅ Prometheus metrics integration
- ✅ Health check and readiness endpoints
- ✅ Docker multi-stage builds
- ✅ Docker Compose orchestration
- ✅ Automated tests with pytest

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### 1. Clone and Setup

```bash
cd credit-score-mlflow-api

# Copy environment configuration
cp .env.example .env
```

### 2. Start Services with Docker Compose

```bash
# Start all services (PostgreSQL, MLflow, Prometheus)
docker-compose up -d postgres mlflow prometheus

# Wait for MLflow to be ready (about 30 seconds)
docker-compose logs -f mlflow
```

### 3. Train the Model

```bash
# Run the training job
docker-compose run --rm train
```

This will:
- Generate synthetic credit data
- Train a Random Forest classifier
- Log metrics and model to MLflow
- Register and promote the model to Production

### 4. Start the API

```bash
# Start the API service
docker-compose up -d api

# Check API health
curl http://localhost:8000/api/v1/health
```

### 5. Access the Services

| Service | URL | Description |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | Swagger UI documentation |
| API ReDoc | http://localhost:8000/redoc | Alternative API documentation |
| MLflow UI | http://localhost:5001 | Experiment tracking UI |
| Prometheus | http://localhost:9090 | Metrics dashboard |
| API Metrics | http://localhost:8000/metrics | Prometheus metrics endpoint |

## 📖 API Documentation

### Endpoints

#### Health Checks

```bash
# Basic health check
GET /api/v1/health

# Readiness check (model loaded?)
GET /api/v1/health/ready
```

#### Predictions

```bash
# Single prediction
POST /api/v1/predict

# Batch prediction
POST /api/v1/predict/batch
```

#### Model Information

```bash
# Get model info
GET /api/v1/model/info
```

### Example: Single Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 65000.0,
    "employment_length": 8,
    "loan_amount": 15000.0,
    "loan_intent": "PERSONAL",
    "home_ownership": "MORTGAGE",
    "credit_history_length": 10,
    "num_credit_lines": 5,
    "derogatory_marks": 0,
    "total_debt": 20000.0
  }'
```

**Response:**

```json
{
  "application_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction": "APPROVED",
  "confidence": 0.85,
  "risk_score": 0.15,
  "approval_probability": 0.85,
  "rejection_probability": 0.15,
  "model_version": "1",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Example: Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "applications": [
      {
        "age": 35,
        "income": 65000.0,
        "employment_length": 8,
        "loan_amount": 15000.0,
        "loan_intent": "PERSONAL",
        "home_ownership": "MORTGAGE",
        "credit_history_length": 10,
        "num_credit_lines": 5,
        "derogatory_marks": 0,
        "total_debt": 20000.0
      },
      {
        "age": 22,
        "income": 25000.0,
        "employment_length": 1,
        "loan_amount": 50000.0,
        "loan_intent": "VENTURE",
        "home_ownership": "RENT",
        "credit_history_length": 1,
        "num_credit_lines": 1,
        "derogatory_marks": 3,
        "total_debt": 15000.0
      }
    ]
  }'
```

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   FastAPI   │    │   MLflow    │    │  PostgreSQL │      │
│  │    :8000    │───▶│    :5001    │───▶│    :5432    │      │
│  └──────┬──────┘    └─────────────┘    └─────────────┘      │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                             │
│  │ Prometheus  │                                             │
│  │    :9090    │                                             │
│  └─────────────┘                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
credit-score-mlflow-api/
├── app/                          # FastAPI application
│   ├── __init__.py
│   ├── api.py                    # API endpoints
│   ├── config.py                 # Configuration settings
│   ├── main.py                   # Application entry point
│   ├── monitoring.py             # Prometheus metrics
│   └── schemas.py                # Pydantic models
├── ml/                           # Machine learning module
│   ├── __init__.py
│   ├── config.py                 # ML configuration
│   ├── data_generator.py         # Synthetic data generation
│   ├── predict.py                # Prediction utilities
│   └── train.py                  # Model training with MLflow
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_api.py               # API integration tests
│   └── test_model.py             # ML model tests
├── .env.example                  # Environment template
├── .gitignore
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Multi-stage Dockerfile
├── prometheus.yml                # Prometheus configuration
├── pytest.ini                    # Pytest configuration
├── README.md
└── requirements.txt              # Python dependencies
```

## 🧪 Testing

### Run Tests Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov=ml --cov-report=html
```

### Run Tests in Docker

```bash
docker-compose run --rm api pytest tests/ -v
```

## 📊 Monitoring

### Prometheus Metrics

The API exposes the following Prometheus metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by method, endpoint, status |
| `http_request_duration_seconds` | Histogram | Request latency |
| `predictions_total` | Counter | Predictions by result and model version |
| `prediction_duration_seconds` | Histogram | Prediction latency |
| `batch_prediction_size` | Histogram | Batch prediction sizes |
| `model_loaded` | Gauge | Model load status (1=loaded, 0=not) |

### Example Prometheus Queries

```promql
# Request rate (per second)
rate(http_requests_total[5m])

# Average prediction latency
rate(prediction_duration_seconds_sum[5m]) / rate(prediction_duration_seconds_count[5m])

# Approval rate
sum(predictions_total{result="APPROVED"}) / sum(predictions_total)
```

## 🔧 Development

### Local Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start MLflow locally (optional, for testing)
mlflow server --host 0.0.0.0 --port 5000

# Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | Credit Score API | Application name |
| `APP_VERSION` | 1.0.0 | API version |
| `DEBUG` | false | Enable debug mode |
| `HOST` | 0.0.0.0 | API host |
| `PORT` | 8000 | API port |
| `MLFLOW_TRACKING_URI` | http://mlflow:5001 | MLflow server URL |
| `MODEL_NAME` | credit-score-model | Registered model name |
| `MODEL_STAGE` | Production | Model stage to load |

## 📦 Deployment

### Docker Build

```bash
# Build the image
docker build -t credit-score-api .

# Run the container
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5001 \
  credit-score-api
```

### Production Considerations

1. **Security**: Implement authentication (OAuth2, API keys)
2. **SSL/TLS**: Use HTTPS in production
3. **Rate Limiting**: Add rate limiting middleware
4. **Logging**: Configure centralized logging
5. **Secrets**: Use secret management (Vault, AWS Secrets Manager)
6. **Scaling**: Use container orchestration (Kubernetes)

## 📄 License

This project is for educational purposes as part of the DevOps & MLOps module.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
