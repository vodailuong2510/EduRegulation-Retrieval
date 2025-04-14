from pymongo import MongoClient
from clearml import Task
task = Task.init(project_name='EduRegulation-Retrieval', task_name='Data Preparation')

def get_mongo_collection(Mongo_Key):
    client = MongoClient(Mongo_Key)
    db = client["chat"] 
    messages_collection = db["messages"] 

    return messages_collection


def save_message(messages_collection, chat_id: str, sender: str, content: str):
    message = {
        "chat_id": chat_id,
        "sender": sender,
        "content": content,
    }
    messages_collection.insert_one(message)

def get_last_messages(messages_collection, chat_id: str, limit: int = 10):
    messages = messages_collection.find({"chat_id": chat_id}) \
        .sort("timestamp", -1) \
        .limit(limit)

    return [{"role": "user" if msg["sender"] == "user" else "assistant", "content": msg["content"]} for msg in messages]

def delete_all_messages(messages_collection, chat_id):
    result = messages_collection.delete_many({"chat_id": chat_id})
    return result.deleted_count

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    print("Deleted:", delete_all_messages(get_mongo_collection(os.getenv("MONGO_URI")), chat_id= 1))