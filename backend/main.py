from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
ps = PorterStemmer()
nltk.download("stopwords")

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


def pre_process(data: str):
    """
    Process article text to correct format for model analysis.
    """
    result = data.lower()
    result = result.split()

    # stem words
    result = [ps.stem(word) for word in result if not word in stopwords.words("english")]
    result = " ".join(result)
    return result


def get_sentiment(label: int):
    """Convert the binary labels into appropriate sentiment strings."""
    return "REAL" if label == 0 else "FAKE"



class Article(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Fake News Detector API is running"}


@app.post("/predict")
def predict(article: Article):
    """
    Handles prediction:
    Finds and parses article from posted link to run through model.
    """
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess
        text = pre_process(article.text)
        # Vectorize text
        tfidf_text = vectorizer.transform([text])
        # Predict
        prediction = model.predict(tfidf_text)
        result = get_sentiment(prediction[0])
        print(result)
        
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
