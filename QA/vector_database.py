from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import weaviate
import weaviate.classes.config as wc
from sentence_transformers import SentenceTransformer
from clearml import Task
task = Task.init(project_name='EduRegulation-Retrieval', task_name='Vector Database Build')

def start_weaviate():
    client = weaviate.connect_to_local()

    print("Client is ready:", client.is_ready())

    return client

def create_new_collection(client):
    try:
        print(client.collections.get("chatbot"))
        print("Collection already exists")
    except weaviate.exceptions.UnexpectedStatusCodeException:
        print("Creating new collection")
        client.collections.create(
            name="chatbot",
            description="A collection of documents",
            properties=[
                wc.Property(name="filname", data_type=wc.DataType.TEXT),
                wc.Property(name="content", data_type=wc.DataType.TEXT),
            ],
        )
        print("Collection created")

def insert_data(client, splits, metas):
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

    with client.batch.dynamic() as batch:
        for md_text, meta in tqdm(zip(splits, metas), desc="Inserting into Weaviate", unit="doc"):
            vector = embedding_model.encode(md_text).tolist() 

            batch.add_object(
                collection= "chatbot",
                properties= {
                    "filename": meta["filename"],
                    "content": md_text,  
                },
                vector=vector, 
            )

if __name__ == "__main__":
    client= start_weaviate()
    print(client.collections.delete("chatbot"))
    client.close()