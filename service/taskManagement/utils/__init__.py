import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "chatbot")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "conversation_states")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

def save_conversation_to_db(convo_id: str, state: dict):
    state_to_save = {
        "fields": state.get("fields", {}),
        "conversation": state.get("conversation", [])
    }

    collection.update_one(
        {"_id": convo_id},
        {"$set": {"state": state_to_save}},
        upsert=True
    )


def load_conversation_from_db(convo_id: str):
    doc = collection.find_one({"_id": convo_id})
    if doc:
        return doc["state"]
    return None

def delete_conversation_from_db(convo_id: str):
    collection.delete_one({"_id": convo_id})
