FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-vie \
    git \
    && apt-get clean

RUN apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN dvc init && \
    unzip data.zip -d ./data && \
    dvc add data && \
    dvc remote add -d myremote /.dvc/dvcstore -f && \
    dvc push
    
RUN python setup_vectordb.py

EXPOSE 8000

WORKDIR /app/web

CMD ["uvicorn", "web.backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]