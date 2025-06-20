import os
import sys
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.path.abspath(".."))

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

from QA.response import reply
from QA.retrieve import retrieve_document
from QA.database import get_last_messages, save_message, get_mongo_collection, delete_all_messages

load_dotenv(find_dotenv())

MONGO_URI= os.getenv("MONGO_URI")

mongo_collections = get_mongo_collection(MONGO_URI)

app = FastAPI()

# Initialize Prometheus instrumentation with additional metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True,
)

app.mount("/static", StaticFiles(directory="./static"), name="static")
HTML_PATH = r"./frontend.html"

class MessageRequest(BaseModel):
    message: str

@app.get("/")
async def serve_html():
    return FileResponse(HTML_PATH)

@app.get("/history")
async def get_history():
    chat_id = 1
    chat_history = get_last_messages(mongo_collections, chat_id=chat_id, limit=20)
    
    messages = [
        {"sender": msg["role"], "content": msg["content"]}
        for msg in chat_history
    ]
    
    return JSONResponse(content=messages)

@app.get("/clear-history")
async def clear_history():
    chat_id = 1
    delete_all_messages(mongo_collections, chat_id=chat_id)
    return JSONResponse(content={"message": "History cleared."})

@app.post("/chat")
async def chat(request: MessageRequest, background_tasks: BackgroundTasks):
    prompt = request.message
    chat_id= 1
    
    context = retrieve_document(query=prompt)
    response = []

    async def response_generator():
        async for chunk in reply(prompt, context, model_path="./results/saved_model"):
            response.append(chunk)
            yield chunk 

    async def save_response():
        full_response = "".join(response)
        save_message(mongo_collections, chat_id=chat_id, sender="user", content=prompt)
        save_message(mongo_collections, chat_id=chat_id, sender="assistant", content=full_response)

    background_tasks.add_task(save_response)

    return StreamingResponse(response_generator(), media_type="text/plain")

def run():
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run()