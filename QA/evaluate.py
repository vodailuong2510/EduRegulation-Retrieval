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
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1. Rank by Document
    documents = contexts['document'].unique().tolist()  # Get unique documents
    ranked_document_indices, _ = rank_contexts(question, documents, model)

    best_document_index = ranked_document_indices[0]  # Get best document
    best_document = documents[best_document_index]

    # 2. Rank by Article within the best document
    articles_in_best_document = contexts[contexts['document'] == best_document]['article'].unique().tolist()  # Get unique articles in the best document
    ranked_article_indices, _ = rank_contexts(question, articles_in_best_document, model)

    best_article_index = ranked_article_indices[0]  # Get best article within the best document
    best_article = articles_in_best_document[best_article_index]

    # 3. Rank by Context within the best article
    contexts_in_best_article = contexts[(contexts['document'] == best_document) & (contexts['article'] == best_article)]['context'].tolist()
    ranked_context_indices, _ = rank_contexts(question, contexts_in_best_article, model)

    best_context_index = ranked_context_indices[0]  # Get best context
    best_context = contexts_in_best_article[best_context_index]

    qa_pipeline = pipeline("question-answering", model = r"./results/saved_model")
    return qa_pipeline(question=question, context=best_context)

if __name__ == "__main__":
    import pandas as pd
    contexts = pd.read_csv(r"..\EducationRegulation-QA\app\context.csv")
    question = "Đơn vị nào chịu trách nhiệm phân công chấm bài thi?"
    print(infer(question, contexts, model_path=r"./results/saved_model"))