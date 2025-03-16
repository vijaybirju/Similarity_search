import os
import time
import requests
import uvicorn
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.models.predict_model import preprocess_text_dataframe, normalize_text_dataframe
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
api_key = os.getenv("HF_API_KEY")

# Initializie the app

app = FastAPI()

headers = {"Authorization": f"Bearer {api_key}"}

# define the request model
class PredictionRequest(BaseModel):
    text1:str
    text2:str

@app.get('/')
def home():
    return 'Welocme to the text Similarity prediction API'


@app.post('/prediction')
def get_similarity_score(request:PredictionRequest):
    try:
        # convert the text to tf-idf vectors
        data = {
            'text1':request.text1,
            'text2':request.text2
        }
        df = preprocess_text_dataframe(data)
        df = normalize_text_dataframe(df)
        # Compute sentence embeddings
        similarity_score = query({
            "inputs": {
            "source_sentence": df["text1"].iloc[0],
            "sentences": [df["text2"].iloc[0]]
        },
        })

        similarity_score = similarity_score[0]

        return {
            'text1':request.text1,
            'text2':request.text2,
            "similarity score": round(similarity_score, 4)
        }
    except Exception as e:
        return {"error": str(e)}


def query(payload, retries=3, delay=5):
    for i in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        time.sleep(delay)  # Wait before retrying
    raise ValueError(f"API Error: {response.status_code}, {response.text}")
	

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)