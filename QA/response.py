from transformers import pipeline
import time
from prometheus_client import Histogram, Gauge

# Define Prometheus metrics
model_inference_time = Histogram(
    'model_inference_cpu_time_seconds',
    'Time spent on model inference',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
)

model_confidence = Gauge(
    'model_confidence_score',
    'Confidence score of model predictions'
)

def infer(question, context, model_name_or_path= "vodailuong2510/MLops"):
    qa_pipeline = pipeline("question-answering", model=model_name_or_path, tokenizer=model_name_or_path)
    
    # Measure inference time
    start_time = time.time()
    result = qa_pipeline(question=question, context=context)
    inference_time = time.time() - start_time
    
    # Record metrics
    model_inference_time.observe(inference_time)
    model_confidence.set(result['score'])
    
    return result

async def reply(question, context, model_path="vodailuong2510/MLops"):
    result = infer(question=question, context=context, model_name_or_path=model_path)

    for char in result['answer']:
        yield char