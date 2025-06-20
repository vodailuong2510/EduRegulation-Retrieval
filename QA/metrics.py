from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics for model inference
model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Time spent processing inference requests',
    ['model_name']
)

model_confidence_score = Histogram(
    'model_confidence_score',
    'Model confidence scores',
    ['model_name']
)

model_inference_total = Counter(
    'model_inference_total',
    'Total number of inference requests',
    ['model_name']
)

model_inference_errors = Counter(
    'model_inference_errors_total',
    'Total number of inference errors',
    ['model_name']
)

# Context for timing inference
class InferenceTimer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        model_inference_total.labels(model_name=self.model_name).inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            model_inference_errors.labels(model_name=self.model_name).inc()
        else:
            duration = time.time() - self.start_time
            model_inference_duration.labels(model_name=self.model_name).observe(duration) 