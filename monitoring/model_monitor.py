from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator, metrics
import time
from functools import wraps

# Create a custom registry
CUSTOM_REGISTRY = CollectorRegistry()

# Metrics for inference speed
inference_duration = Histogram(
    "model_inference_duration_seconds",
    "Time spent processing inference",
    ["model_name", "device"],
    registry=CUSTOM_REGISTRY
)

# Metrics for confidence scores
confidence_score = Gauge(
    "model_confidence_score",
    "Confidence score of model predictions",
    ["model_name"],
    registry=CUSTOM_REGISTRY
)

# Counter for total predictions
total_predictions = Counter(
    "model_total_predictions",
    "Total number of predictions made",
    ["model_name"],
    registry=CUSTOM_REGISTRY
)

def track_inference(model_name: str, device: str = "cpu"):
    """
    Decorator to track inference time and confidence scores
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Execute the model inference
            result = await func(*args, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            inference_duration.labels(model_name=model_name, device=device).observe(duration)
            confidence_score.labels(model_name=model_name).set(result.get("confidence", 0.0))
            total_predictions.labels(model_name=model_name).inc()
            
            return result
        return wrapper
    return decorator

def setup_monitoring(app):
    """
    Setup monitoring for FastAPI application
    """
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[],
        env_var_name="ENABLE_METRICS",
        inprogress_name="fastapi_inprogress",
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app).expose(app)
    
    # Expose custom model metrics
    from prometheus_client import make_asgi_app
    metrics_app = make_asgi_app(registry=CUSTOM_REGISTRY)
    app.mount("/model-metrics", metrics_app) 