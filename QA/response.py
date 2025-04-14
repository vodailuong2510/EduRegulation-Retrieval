from transformers import pipeline

def infer(question, context, model_name_or_path= "vodailuong2510/MLops"):
    qa_pipeline = pipeline("question-answering", model=model_name_or_path, tokenizer=model_name_or_path)
    result = qa_pipeline(question=question, context=context)

    return result

def reply(question, context, model_path="vodailuong2510/MLops"):
    result = infer(question=question, context=context, model_name_or_path=model_path)

    return result