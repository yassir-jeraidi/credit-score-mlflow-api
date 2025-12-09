"""
Prometheus Monitoring and Metrics.

Provides metrics collection for the Credit Score API.
"""
import time
from typing import Callable
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Request metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["result", "model_version"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Prediction latency in seconds",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

BATCH_SIZE = Histogram(
    "batch_prediction_size",
    "Size of batch prediction requests",
    buckets=[1, 5, 10, 25, 50, 75, 100]
)

# Model metrics
MODEL_INFO = Info(
    "model",
    "Information about the loaded model"
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is currently loaded (1=loaded, 0=not loaded)"
)

# Application metrics
APP_INFO = Info(
    "app",
    "Application information"
)


def set_app_info(name: str, version: str) -> None:
    """Set application info metrics."""
    APP_INFO.info({
        "name": name,
        "version": version,
    })


def set_model_info(model_name: str, model_version: str, model_stage: str) -> None:
    """Set model info metrics."""
    MODEL_INFO.info({
        "name": model_name,
        "version": model_version,
        "stage": model_stage,
    })


def set_model_loaded(loaded: bool) -> None:
    """Set model loaded gauge."""
    MODEL_LOADED.set(1 if loaded else 0)


def record_prediction(result: str, model_version: str, duration: float) -> None:
    """
    Record a prediction metric.

    Args:
        result: Prediction result (APPROVED/REJECTED)
        model_version: Model version used
        duration: Prediction duration in seconds
    """
    PREDICTION_COUNT.labels(result=result, model_version=model_version).inc()
    PREDICTION_LATENCY.labels(model_version=model_version).observe(duration)


def record_batch_size(size: int) -> None:
    """Record batch prediction size."""
    BATCH_SIZE.observe(size)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        path = request.url.path

        # Normalize path for metrics (remove IDs, etc.)
        endpoint = self._normalize_path(path)

        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

        return response

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for consistent metrics.

        Replaces dynamic path segments with placeholders.
        """
        # Add path normalization rules as needed
        # For now, return as-is
        return path


async def metrics_endpoint(request) -> Response:
    """
    Prometheus metrics endpoint.

    Returns:
        Response with Prometheus metrics
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
