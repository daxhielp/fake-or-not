import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'news.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

def train():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Data cleaning: drop rows with missing text
    print(f"Initial data shape: {df.shape}")
    df.dropna(subset=['text'], inplace=True)
    print(f"Data shape after dropping NaNs: {df.shape}")

    # Get labels
    labels = df.label
    
    # Split data
    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    
    # Initialize vectorizer
    print("Vectorizing...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    
    # Initialize and train classifier
    print("Training model...")
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    # Evaluate
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100, 2)}%')
    
    # Save model and vectorizer
    print("Saving artifacts...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pac, f)
        
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
        
    print("Done!")

if __name__ == "__main__":
    train()
