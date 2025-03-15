import joblib
import uvicorn
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.models.predict_model import preprocess_text_dataframe, normalize_text_dataframe

# Initializie the app

app = FastAPI()

# load the model
current_path = Path(__file__)
root_path = current_path.parent
mode_name  = 'sentence_transformer'
model_path = root_path / 'models' /'models' / mode_name
model = SentenceTransformer(str(model_path))


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
        embeddings = model.encode([df.text1[0], df.text2[0]])

        # Compute cosine similarity
        similarity_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

        return {
            'text1':request.text1,
            'text2':request.text2,
            "similarity score": round(similarity_score, 4)
        }
    except Exception as e:
        return {"error": str(e)}


# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)
