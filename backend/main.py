from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware
from newspaper import Article as NewspaperArticle

app = FastAPI()

# CORS setup
origins = [
    "http://localhost:5173", # Default Vite port
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

model = None
vectorizer = None

# open model & vectorizer
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")


class ArticleRequest(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"message": "Fake News Detector API is running"}


@app.post("/predict")
def predict(request: ArticleRequest):
    """
    Handles prediction:
    Finds and parses article from posted link to run through model.
    """
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Download and parse article
        article = NewspaperArticle(request.url)
        article.download()
        article.parse()
        text = article.text

        # Vectorize text directly (model trained with TfidfVectorizer's own preprocessing)
        tfidf_text = vectorizer.transform([text])
        
        # Predict probability
        # classes_ are usually [0, 1] where 0 is REAL, 1 is FAKE (based on training script)
        probabilities = model.predict_proba(tfidf_text)[0]
        fake_prob = probabilities[1]
        
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob * 100
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
