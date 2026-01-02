# fake-or-not

This project contains a machine learning model to detect fake news, served via a FastAPI backend and accessible through a React frontend.

## Project Structure

- `backend/`: Contains the Python/FastAPI server and model training logic.
- `frontend/`: Contains the React application.
- `model.ipynb`: Contains model training/analysis.

## Project Overview

### 1. Backend

The backend serves the trained model.

- `model.ipynb` contains the training logic for the model.
- The backend is built with python and FastAPI.
- `main.py` contains the API endpoints, including the logic for preprocessing data for model predictions
- The backend will expect a URL to a news article. It will then extract the text from the article
  and preprocess it for a prediction.

  When run, the server will default to `http://localhost:8000`.

### 2. Frontend

The frontend uses React and Vite to host the model. It will call the API and retrieve the model
predictions to display to the user.

## Usage

1.  Copy the link of a news article you want to verify.
2.  Paste it into the text area on the website.
3.  Click "Verify News".
4.  The system will analyze the text and predict if it is **REAL** or **FAKE**.
5.  If fake, the system will give an extent of how fake the article is.

### 3. Model

The model uses NLP techniques and Logistic Regression to extract sentiment from text. The text is
first filtered with a stemmer. Then, the text is quantized by Term Frequency-Inverse Document
Frequency vectorization. The model uses logistic regression for a fake/real classification.

The model is saved using onnx for less memory-intensive deployment.

For more detail, see [model notebook.](/backend/model.ipynb).

[Access dataset used for training](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)
