# Fake News Detector Website

This project contains a machine learning model to detect fake news, served via a FastAPI backend and accessible through a React frontend.

## Project Structure

- `backend/`: Contains the Python/FastAPI server and model training logic.
- `frontend/`: Contains the React application.
- `news.csv`: Dataset used for training.
- `fake_news_detector.ipynb`: Original analysis notebook.

## Setup & Running

### 1. Backend

The backend serves the trained model.

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  (Optional) Create a virtual environment and activate it.
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  (Optional) Retrain the model if needed:
    ```bash
    python train_model.py
    ```
5.  Start the server:
    ```bash
    python main.py
    ```
    The server will run on `http://localhost:8000`.

### 2. Frontend

The frontend is a modern React application.

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    Open the link provided (usually `http://localhost:5173`) in your browser.

## Usage

1.  Copy the text of a news article you want to verify.
2.  Paste it into the text area on the website.
3.  Click "Verify News".
4.  The system will analyze the text and predict if it is **REAL** or **FAKE**.
