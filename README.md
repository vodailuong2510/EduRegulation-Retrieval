# 📌 UIT Policies Retriever
## LINK VIDEO
[Video Demo](https://drive.google.com/drive/folders/1wB7OugWjR4IahRpYnIeV5ECZLKMWDd8B?usp=drive_link)
## 🧠 Introduction

This project is designed to retrieve and provide information about academic policies and regulations at the University of Information Technology (UIT). It aims to support students and faculty members by providing accurate and up-to-date information in a timely and efficient manner.

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
- **Monitoring Stack**:
  - *Prometheus*: Collects and stores metrics from the application and model.
  - *Grafana*: Visualizes metrics through interactive dashboards.
  - *Loki*: Aggregates and stores application logs.
  - *Promtail*: Ships logs to Loki for centralized logging.

These components work together to create a scalable, maintainable, and high-performance system that delivers trustworthy policy-related information to students and faculty at UIT.

<p align="center">
  <img src="https://github.com/vodailuong2510/EduRegulation-Retrieval/blob/main/images/training.png?raw=true" alt="Training" />
</p>
---

## ✨ Features

- ✅ Automatically answers questions related to academic policies and regulations
- ✅ Real-time interaction through a user-friendly interface
- ✅ Easy to deploy locally or on a remote server
- ✅ Comprehensive monitoring of API and Model performance
- ✅ Centralized logging and metrics visualization
- ✅ Automated model training and evaluation pipeline

---

## 📊 Monitoring System

The project includes a comprehensive monitoring system using Prometheus and Grafana to track both API and Model performance.

### API Monitoring
- Request rates by endpoint and method
- Response time distributions
- Error rates and status codes
- API endpoint usage patterns

### Model Monitoring
- Model inference time (CPU Time)
- Model confidence scores
- Performance metrics (99th percentile)
- Model behavior over time

### System Metrics
- System resource usage
- Container metrics
- Network metrics

---

## 📁 Project Structure

```
EduRegulation-Retrieval/
├── web/                  # Web application (FastAPI + Frontend)
├── QA/                   # Question Answering system
├── reader/              # Document reader and processor
├── monitoring/          # Monitoring system
│   ├── prometheus/     # Prometheus configuration
│   ├── grafana/        # Grafana dashboards
│   ├── loki/          # Log aggregation
│   └── promtail/      # Log shipping
├── docker-compose.*.yml # Docker compose files
└── [other config files]
```

## 🚀 Getting Started

### Prerequisites
- **Python version**: 3.10.16
- Remember to create a .env file with the following keys:
   > HUGGING_FACE=your_token_here  
   > PARSER=your_llama_cloud_url_here  
   > MONGO_URI=your_mongo_uri_here

### Docker Setup
```bash
# Create a monitoring network
docker network create monitoring_network

# Start MongoDB
docker compose -f docker-compose.mongodb.yml up -d

# Start Weaviate
docker compose -f docker-compose.weaviate.yml up -d

# Start app 
docker compose -f docker-compose.app.yml up -d

# Start monitoring the stack
docker compose -f docker-compose.monitor.yml up -d
```

### Manual Setup
```bash
# Clone the project
git clone git@github.com:vodailuong2510/EduRegulation-Retrieval.git
cd EduRegulation-Retrieval

# Install required Python packages
pip install -r requirements.txt  # Main project dependencies
pip install -r app_requirements.txt  # Web application dependencies

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

### Data Version Control (DVC)

The project uses DVC for data versioning and management. Key files:
- `data.dvc`: Tracks the data directory
- `dvc.yaml`: Defines data processing pipeline
- `.dvcignore`: Specifies files to ignore in DVC

### DVC Commands
```bash
# Initialize DVC
dvc init

# Add data to DVC
dvc add data

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull

# Run data pipeline
dvc repro
```

### Setup ClearML
```bash
clearml-init
```
When setting up ClearML for the first time, you need to create new ClearML credentials through the settings page in your `clearml-server` web app (e.g., http://localhost:8080//settings/workspace-configuration) or create a free account at https://app.clear.ml/settings/workspace-configuration. Press "Create new credentials," then copy the configuration to the clipboard and paste it. Wait until the credentials are verified, and then your ClearML setup is successful.

### Data Processing and Model Training
```bash
# Extract text from PDFs and store in Weaviate
python setup_vectordb.py

# Train and evaluate the model:
python train.py
python test.py
```

### Accessing the Application
- Web Application: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Grafana Dashboard: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### Monitoring Dashboards
The following dashboards are available in Grafana:
- API Monitoring: `monitoring/grafana/dashboards/api-monitoring.json`
- Model Monitoring: `monitoring/grafana/dashboards/model-monitoring.json`
- System Metrics: `monitoring/grafana/dashboards/system-metrics.json`

## 💻 Development

### Testing

The project includes automated tests for both the model and API:

```bash
# Run model tests
python test.py

# Run API tests (from web directory)
cd web
pytest
```

## License
Distributed under the Unlicense License. See LICENSE.txt for more information.

## Contact
Luong Vo Dai - vodailuong2510@gmail.com
