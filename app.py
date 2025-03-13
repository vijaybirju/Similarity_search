import joblib
import uvicorn
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity

# Initializie the app

app = FastAPI()

# load the model
current_path = Path(__file__)
root_path = current_path.parent
mode_name  = 'tf-idf.joblib'
model_path = root_path / 'models' /'models' / mode_name
tfidf_vectorizer_model = joblib.load(model_path)


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
        tfidf_matrix = tfidf_vectorizer_model.transform([request.text1,request.text2])

        # compute similarity
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

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
