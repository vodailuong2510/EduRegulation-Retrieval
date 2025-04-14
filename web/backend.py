import sys
import os
sys.path.append(os.path.abspath(".."))

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from dotenv import load_dotenv, find_dotenv
from source.utils import system_instruction 
from source.chatbot_streaming import gpt_chat
from source.database import get_last_messages, save_message, get_mongo_collection

load_dotenv(find_dotenv())

MONGO_URI= os.getenv("MONGO_URI")

instruction = system_instruction()
mongo_collections = get_mongo_collection(MONGO_URI)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True,
)

app.mount("/static", StaticFiles(directory="./"), name="static")
HTML_PATH = r"/home/user/projects/lark-chatbot/chatbot/web/frontend.html"

class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: MessageRequest, background_tasks: BackgroundTasks):
    prompt = request.message

    chat_id= 1
    chat_history= get_last_messages(mongo_collections, chat_id=chat_id, limit=10)

    response = []

    async def response_generator():
        async for chunk in gpt_chat(prompt, instruction, chat_history):
            response.append(chunk)
            yield chunk 

    save_message(mongo_collections, chat_id=chat_id, sender="user", content=prompt)

    async def save_response():
        full_response = "".join(response)
        save_message(mongo_collections, chat_id=chat_id, sender="assistant", content=full_response)

    background_tasks.add_task(save_response)

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/")
async def serve_html():
    return FileResponse(HTML_PATH)

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run()
    