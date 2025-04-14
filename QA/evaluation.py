import evaluate
from .response import infer

def compute_em(predictions, references):
    metric = evaluate.load("squad")
    em_score = metric.compute(predictions=predictions, references=references)
    return em_score

def evaluate_model(test_dataset, model_name_or_path= "vodailuong2510/MLops"):
    predictions = []
    references = []

    for sample in test_dataset:
        context = sample["context"]
        question = sample["question"]
        answer = sample["extractive answer"]["text"][0]

        result = infer(question=question, context=context, model_name_or_path=model_name_or_path)

        predictions.append({
            "id": str(sample["index"]),
            "prediction_text": result["answer"].strip()
        })

        references.append({
            "id": str(sample["index"]), 
            "answers": [{
                "text": answer.strip(),
                "answer_start": sample["extractive answer"]["answer_start"][0]
            }]
        })

    em_score = compute_em(predictions, references)
    return em_score
