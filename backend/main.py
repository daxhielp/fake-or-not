from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from newspaper import Article as NewspaperArticle
import numpy as np
import onnxruntime as rt

app = FastAPI()

# CORS setup
origins = [
    "http://localhost:5173", # Default prod port
    "http://localhost:3000",
    "https://fake-or-not-6rcp.vercel.app",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load pipeline
try:
    sess = rt.InferenceSession("model_pipeline.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
except Exception as e:
    print(f"Error loading model-vectorizer pipeline: {e}")


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
    if not sess:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Download and parse article
        article = NewspaperArticle(request.url)
        article.download()
        article.parse()
        text = article.text

        # Vectorize text directly (model trained with TfidfVectorizer's own preprocessing)
        input_data = np.array([[text]], dtype=object)

        prediction = sess.run([label_name], {input_name: input_data})[0]
        prediction = prediction.tolist()
        print(prediction)
        fake_prob = prediction[0]
        
        prediction = "FAKE" if fake_prob == 1 else "REAL"
        
        return {
            "prediction": prediction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
