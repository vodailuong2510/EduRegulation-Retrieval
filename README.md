# 📌 UIT Policies Retriever

## 🧠 Introduction

This project is designed to retrieve and provide information about academic policies and regulations at the University of Information Technology (UIT). It aims to support students and faculty members by delivering accurate and up-to-date information quickly and efficiently.

The data collected by the team is from PDF files regarding the regulations and training policies of the University of Information Technology (UIT), available at the following link: https://student.uit.edu.vn/qui-che-qui-dinh-qui-trinh

By combining retrieval techniques with large language models (LLMs), the system ensures that users receive relevant and context-aware responses to their queries related to UIT's academic rules, procedures, and general policies.

<p align="center">
  <img src="https://github.com/vodailuong2510/EduRegulation-Retrieval/blob/main/images/retriver.png?raw=true" alt="Retriever" />
</p>
---

## ⚙️ Technologies Used

This system leverages cutting-edge technologies in the fields of Natural Language Processing (NLP) and Artificial Intelligence (AI) to build an effective, responsive information assistant. The key components include:

- **Large Language Models (LLM)**: BERT is used to extract relevant answers from retrieved context, ensuring accurate and meaningful responses.
- **Hybrid Search (Semantic + Keyword)**: Combines dense vector search with traditional keyword-based retrieval to enhance the relevance of context retrieved based on user queries.
- **Weaviate & MongoDB**:
  - *Weaviate*: Acts as a vector database to store embedded context documents and power semantic search.
  - *MongoDB*: Stores user query history, raw documents, and system metadata for better user tracking and management.
- **LlamaIndex & Tesseract OCR**:
  - *LlamaIndex*: Parses and indexes documents (PDF, text, etc.) to enable efficient retrieval.
  - *Tesseract OCR*: Extracts text from scanned PDF files, enabling the system to process non-digital documents and include them in the searchable knowledge base.
- **FastAPI, HTML, CSS, JavaScript**: 
  - *FastAPI*: Provides the backend API with high performance and async support.
  - *HTML/CSS/JS*: Used for building a simple demo frontend for user interaction.
- **Docker, MLflow, DVC, Optuna, ClearML**:
  - *Docker*: Containerizes the application for easy deployment and consistency across environments.
  - *MLflow*: Tracks experiments, parameters, metrics, and model versions.
  - *DVC (Data Version Control)*: Manages datasets and model versions efficiently.
  - *Optuna*: Performs automated hyperparameter tuning to optimize model performance.
  - *ClearML*: Provides experiment tracking, orchestration, and visualization.

These components work together to create a scalable, maintainable, and high-performance system capable of delivering trustworthy policy-related information to students and faculty at UIT.

<p align="center">
  <img src="https://github.com/vodailuong2510/EduRegulation-Retrieval/blob/main/images/training.png?raw=true" alt="Training" />
</p>
---

## ✨ Features

- ✅ Automatically answers questions related to academic policies and regulations
- ✅ Real-time interaction through a user-friendly interface
- ✅ Easy to deploy locally or on a remote server

---

## Getting Started

- **Python version**: 3.10.16
- Remember to create a .env file with the following keys:
   > HUGGING_FACE=your_token_here  
   > PARSER=your_llama_cloud_url_here  
   > MONGO_URI_URI=your_mongodb_uri_here (you'll get it when you start its docker compose file below)
- Next, start the MongoDB and Weaviate services using Docker Compose:
```bash
# Start MongoDB
docker compose -f docker-compose.mongodb.yml up -d

# Start Weaviate
docker compose -f docker-compose.weaviate.yml up -d
```
- You can either run the application using Docker (see the Docker section below), or follow the steps below to set up the environment manually:

```bash
# Clone the project
git clone git@github.com:vodailuong2510/EduRegulation-Retrieval.git
cd EduRegulation-Retrieval

# Install required Python packages
pip install -r requirements.txt

# Install Tesseract OCR and Vietnamese language data
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-vie

# Set up DVC (Data Version Control)
dvc init
unzip data.zip  # Make sure to unzip your dataset
dvc add data
dvc remote add -d myremote /.dvc/dvcstore -f
dvc push
```
- Then, extract text from PDFs and scanned PDFs, and store the processed data into Weaviate:

```bash
python setup_vectordb.py  # Make sure to set the correct path of tesseract in your device
```
- Now, you can fine-tune BERT using MLFlow and Optuna for information extraction, and then evaluate and test the model:
```bash
python train.py
python test.py
```

- Or you can test the application directly by starting it:
```bash
cd web
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
- For ClearML first you have to connect to the ClearML server, the default server URL is: https://app.clear.ml
```bash
clearml-init
```
Then the terminal ask you to provide the API Credentials
- On the website, create a workspace then it will generate the Access Key and Secret Key, just copy it and paste to the terminal
- After that you can manage and view the pipeline:
```bash
python pipeline.py
```
## License
Distributed under the Unlicense License. See LICENSE.txt for more information.

## Contact
Luong Vo Dai - vodailuong2510@gmail.com