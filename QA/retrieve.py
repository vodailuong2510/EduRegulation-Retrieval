from .vector_database import start_weaviate
from sentence_transformers import SentenceTransformer

def retrieve_document(query: str=""):
    client = start_weaviate()
    collection = client.collections.get("chatbot")

    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    query_vector = embedding_model.encode(query).tolist()

    result = collection.query.hybrid(
        query=query,
        vector=query_vector,
        limit=10,
        alpha=0.7,
    )
    
    context= []
    for o in result.objects:
        context.append(f"filename: {o.properties['filename']}\nlink: {o.properties['link']}\ncontent:\n{o.properties['content']}")

    client.close()
    
    return "\n\n".join(context)

if __name__ == "__main__":
    print(retrieve_document("Quyết định Thành lập Ban điều hành Galaxy Holdings")) 