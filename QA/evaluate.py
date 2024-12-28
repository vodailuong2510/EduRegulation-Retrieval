from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

def rank_contexts(question, contexts, model, batch_size=32):
    question_embedding = model.encode(question, convert_to_tensor=True)
    context_embeddings = model.encode(contexts, convert_to_tensor=True, batch_size=32)

    cosine_scores = util.pytorch_cos_sim(question_embedding, context_embeddings)

    ranked_contexts = cosine_scores[0].cpu().numpy()
    ranked_indices = ranked_contexts.argsort()[::-1] 

    return ranked_indices, ranked_contexts

def infer(question, contexts, model_path:str=r"./results/saved_model"):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 1. Rank by Document
    documents = contexts['document'].unique().tolist()  # Get unique documents
    ranked_document_indices, _ = rank_contexts(question, documents, model)

    # Get top 5 documents
    top_documents = [documents[i] for i in ranked_document_indices[:3]]

    # 2. Rank by Article within the best document
    top_articles = []
    for doc in top_documents:
        articles_in_best_document = contexts[contexts['document'] == doc]['article'].unique().tolist()
        ranked_article_indices, _ = rank_contexts(question, articles_in_best_document, model)

        # Get top 5 articles for each document
        top_articles.append([articles_in_best_document[i] for i in ranked_article_indices[:3]])

    # 3. Rank by Context within the best article
    top_contexts = []
    for doc, articles in zip(top_documents, top_articles):
        for article in articles:
            top_contexts.extend(contexts[(contexts['document'] == doc) & (contexts['article'] == article)]['context'].tolist())

    ranked_context_indices, _ = rank_contexts(question, top_contexts, model)

    best_context_index = ranked_context_indices[0]
    best_context = top_contexts[best_context_index]
    print(top_articles)
    print(top_documents)

    qa_pipeline = pipeline("question-answering", model = r"./results/saved_model")
    return qa_pipeline(question=question, context=best_context)

if __name__ == "__main__":
    import pandas as pd
    contexts = pd.read_csv(r"..\EducationRegulation-QA\app\context.csv")
    question = "Sau khi giáo trình được in, đơn vị nào phân phối giáo trình?"
    print(infer(question, contexts, model_path=r"./results/saved_model"))