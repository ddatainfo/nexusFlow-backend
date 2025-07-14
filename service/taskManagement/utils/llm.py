import requests
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def call_mistral(prompt):
    try:
        res = requests.post(OLLAMA_BASE_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "top_p": 0.8,
            "repeat_penalty": 1.2
        })
        res.raise_for_status()
        return res.json()["response"].strip()
    except Exception as e:
        print("❌ Mistral call failed:", e)
        return "I'm having trouble processing that. Could you please try again?"