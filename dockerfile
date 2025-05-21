FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip==24.3.1
RUN pip install --no-cache-dir sentence-transformers==4.0.2 --no-deps
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r app_requirements.txt
    
EXPOSE 8000

#cd web
CMD ["sh", "-c", "cd web && python backend.py"]


