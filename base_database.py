import pandas as pd
from tqdm import tqdm

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from QA.vector_database import start_weaviate, create_new_collection, insert_data

def create_txt_splitter():
    splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=256)
    return splitter

def split_document(contexts):
    splitter = create_txt_splitter()
    
    splits = {
        f"context_{i}": {
            "splits": [node.text for node in splitter([Document(text=context)])]
        }
        for i, context in enumerate(tqdm(contexts, desc="Splitting contexts...", unit="context"))
    }
    return splits

def formating_chunk(splitted_chunks):
    md_splits = []
    metas = []

    for filename, splitted_doc in splitted_chunks.items():
        for chunk in splitted_doc["splits"]:
            md_splits.append(chunk)
            metas.append({
                "filename": filename,
            })

    return md_splits, metas

def process_csv_to_vectordb(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Remove duplicates from context column
    df = df.drop_duplicates(subset=['context'])
    print(f"Removed {len(df) - len(df.drop_duplicates(subset=['context']))} duplicate contexts")
    
    # Get contexts from the 'context' column
    contexts = df['context'].tolist()
    
    # Split documents
    splits = split_document(contexts)
    
    # Format chunks
    md_splits, metas = formating_chunk(splitted_chunks=splits)
    
    # Initialize and connect to Weaviate
    client = start_weaviate()
    
    # Check if collection exists
    print("Creating new collection...")
    create_new_collection(client=client)
    
    # Insert data into vector database
    insert_data(
        client=client,
        splits=md_splits,
        metas=metas
    )
    
    client.close()

if __name__ == "__main__":
    # Specify your CSV file path here
    csv_path = "/home/mlops/EduRegulation-Retrieval/data/finetune/train.csv"
    process_csv_to_vectordb(csv_path)
