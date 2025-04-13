import os
from collections import defaultdict
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from .utils import is_scan_pdf, read_scanPDF

def pdf2text(parser_api_key:str, folder_path:str, 
             output_folder:str, tesseract_cmd:str, lang:str):
    parser = LlamaParse(
        result_type="text", 
        api_key=parser_api_key,
        language="vi",
        split_by_page=False,
    )

    all_files = os.listdir(folder_path)
    pdf_files = [os.path.join(folder_path, f) for f in all_files if f.lower().endswith('.pdf')]

    file_extractor = {".pdf": parser}

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
        txt_path = os.path.join(output_folder, pdf_name)

        if os.path.exists(txt_path):
            continue
        
        if is_scan_pdf(pdf_path):
            try:
                print(f"Processing scanned PDF: {pdf_path}")

                texts = "\n".join(read_scanPDF(pdf_path=pdf_path, lang=lang, tesseract_cmd=tesseract_cmd))

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(texts)

                print(f"Saved: {txt_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
        else:
            print(f"Processing normal PDF: {pdf_path}")

            reader = SimpleDirectoryReader(
                input_files=[pdf_path],
                file_extractor=file_extractor
            )
            documents = reader.load_data()

            for doc in documents:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(doc.text)
                print(f"Saved: {txt_path}")

def load_data(folder_path:str):
    reader = SimpleDirectoryReader(input_dir= folder_path)
    
    documents = reader.load_data(show_progress=True)

    merged_documents = defaultdict(str)

    for doc in documents:
        file_name = doc.metadata["file_name"]
        merged_documents[file_name] += doc.text + "\n"

    final_documents = []
    for file_name, content in merged_documents.items():
        final_documents.append({"file_name": file_name, "content": content}) 

    return final_documents
