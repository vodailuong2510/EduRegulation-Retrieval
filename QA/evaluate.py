from rank_bm25 import BM25Okapi
import pandas as pd
from transformers import pipeline, AutoTokenizer
import evaluate

def compute_em(predictions, references):
    metric = evaluate.load("exact_match")
    em_score = metric.compute(predictions=predictions, references=references)["exact_match"]
    return em_score

def rank_contexts(question, contexts, tokenizer, batch_size=32):

    tokenized_contexts = [tokenizer.tokenize(context.lower()) for context in contexts]
    tokenized_question = tokenizer.tokenize(question.lower())

    bm25 = BM25Okapi(tokenized_contexts)
    scores = bm25.get_scores(tokenized_question)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return ranked_indices, scores

def infer(question, contexts, model_path="./results/saved_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    documents = contexts['document'].unique().tolist()  
    ranked_document_indices, _ = rank_contexts(question, documents, tokenizer)

    top_documents = [documents[i] for i in ranked_document_indices[:3]]

    top_articles = []
    for doc in top_documents:
        articles_in_document = contexts[contexts['document'] == doc]['article'].unique().tolist()
        ranked_article_indices, _ = rank_contexts(question, articles_in_document, tokenizer)
        top_articles.append([articles_in_document[i] for i in ranked_article_indices[:3]])

    candidate_contexts = []
    for doc, articles in zip(top_documents, top_articles):
        for article in articles:
            candidate_contexts.extend(contexts[(contexts['document'] == doc) & 
                                               (contexts['article'] == article)]['context'].tolist())

    ranked_context_indices, _ = rank_contexts(question, candidate_contexts, tokenizer)

    best_context_index = ranked_context_indices[0]
    best_context = candidate_contexts[best_context_index]

    print("Top Documents:", top_documents)
    print("Top Articles:", top_articles)

    qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)
    result = qa_pipeline(question=question, context=best_context)

    return result

if __name__ == "__main__":
    contexts = pd.read_csv(r"../EducationRegulation-QA/app/context.csv")
    question = "Sau khi giáo trình được in, đơn vị nào phân phối giáo trình?"

    answer = infer(question, contexts, model_path="./results/saved_model")
    print("Answer:", answer)
