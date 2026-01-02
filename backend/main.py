from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from newspaper import article
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
sess = rt.InferenceSession("model_pipeline.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


def get_article_text(url):
    """
    Extracts all text from an given link, expectedly a news article.
    
    :param url: news link url
    """
    a = article(url=url)
    a.download()
    a.parse()
    return a.text

def get_prediction(text):
    """
    Analyzes given text to determine a fake/real analysis
    
    :param text: Text to be analyzed by the model
    """
    text = text.replace('\n', ' ').strip() # clean text

    input_data = np.array([[text]], dtype=object)
    prediction = sess.run([label_name], {input_name: input_data})[0].tolist()
    label = prediction[0]
    prediction = "FAKE" if label == 1 else "REAL"
    return prediction


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
        text = get_article_text(request.url)

        # Run text thru model
        prediction = get_prediction(text)
        
        return {
            "prediction": prediction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
