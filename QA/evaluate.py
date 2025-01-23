import evaluate
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoTokenizer

def infer(question, context, model_name_or_path= "vodailuong2510/saved_model"):
    qa_pipeline = pipeline("question-answering", model=model_name_or_path, tokenizer=model_name_or_path)
    result = qa_pipeline(question=question, context=context)

    return result

def compute_em(predictions, references):
    metric = evaluate.load("squad")
    em_score = metric.compute(predictions=predictions, references=references)
    return em_score

def evaluate_model(test_dataset, model_name_or_path= "vodailuong2510/saved_model"):
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

def rank_contexts(question, contexts, tokenizer, batch_size=32):

    tokenized_contexts = [tokenizer.tokenize(context.lower()) for context in contexts]
    tokenized_question = tokenizer.tokenize(question.lower())

    bm25 = BM25Okapi(tokenized_contexts)
    scores = bm25.get_scores(tokenized_question)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return ranked_indices, scores

def reply(question, contexts, model_path="vodailuong2510/saved_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    candidate_contexts = contexts['context'].tolist()

    ranked_context_indices, _ = rank_contexts(question, candidate_contexts, tokenizer)

    best_context_index = ranked_context_indices[0]
    best_context = candidate_contexts[best_context_index]


    best_context_index = ranked_context_indices[0]
    best_context = candidate_contexts[best_context_index]

    result = infer(question=question, context=best_context, model_name_or_path=model_path)

    return result

if __name__ == "__main__":
    contexts = pd.read_csv(r"../EducationRegulation-QA/app/contexts.csv")
    question = "Sinh viên chưa hết thời gian tối đa hoàn thành khóa học quy định tại Điều 6 của Quy chế này, đã hoàn thành các học phần trong chương trình đào tạo có nguyện vọng xin thôi học theo diện này thì phải làm gì?"

    answer = reply(question, contexts, model_path="vodailuong2510/saved_model")
    print("Answer:", answer)
