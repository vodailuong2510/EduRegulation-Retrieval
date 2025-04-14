import os
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from QA.vector_database import start_weaviate, create_new_collection, insert_data
from reader.reader import pdf2text, load_data

PARSER = os.getenv("PARSER")
GEMINI = os.getenv("GEMINI")

pdf_folder= "./data/pdf"
text_path= "./data/documents"
tesseract_path= r'C:\Program Files\Tesseract-OCR\tesseract.exe'
lang="vie"

def create_txt_splitter():
    splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=256)
    return splitter

def split_document(file_contents):
    splitter = create_txt_splitter()

    splits = {
        doc['file_name']: {
            "splits": [node.text for node in splitter([Document(text=doc['content'])])]
        }
        for doc in tqdm(file_contents, desc="Splitting documents...", unit="doc")
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

if __name__ == "__main__":
    pdf2text(
        parser_api_key=PARSER,
        folder_path=pdf_folder,
        output_folder=text_path,
        tesseract_cmd=tesseract_path,
        lang=lang
    )

    data= load_data(
        folder_path=text_path,
    )

    splits = split_document(data)

    md_splits, metas = formating_chunk(
        splitted_chunks=splits)

    client = start_weaviate()
    create_new_collection(client=client)

    insert_data(
        client=client,
        splits=md_splits, 
        metas=metas
    ) 
    
    client.close()