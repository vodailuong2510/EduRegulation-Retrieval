from transformers import pipeline
from .metrics import InferenceTimer, model_confidence_score

def infer(question, context, model_name_or_path= "vodailuong2510/MLops"):
    with InferenceTimer(model_name_or_path) as timer:
        qa_pipeline = pipeline("question-answering", model=model_name_or_path, tokenizer=model_name_or_path)
        result = qa_pipeline(question=question, context=context)
        
        # Record confidence score
        model_confidence_score.labels(model_name=model_name_or_path).observe(result['score'])
        
        return result

async def reply(question, context, model_path="vodailuong2510/MLops"):
    result = infer(question=question, context=context, model_name_or_path=model_path)

    for char in result['answer']:
        yield char