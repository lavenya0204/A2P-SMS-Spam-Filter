from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from whitelisting import check_whitelist # Updated import path
import logging

# Set up logging and saving logging info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log', # Saves logs to 'app.log' file
    filemode='a'        # Appends new logs to the file
)

# --- Load Model and Vectorizer ---

try:

    classifier = joblib.load('model/sms_classifier_model.joblib')
    vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
    logging.info("Model and Vectorizer loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Failed to load model files: {e}. Ensure they are in the 'model' directory.")
    classifier = None
    vectorizer = None

# --- NLTK Data Check ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:  
    nltk.download('stopwords')
    nltk.download('wordnet')
    logging.info("Downloaded necessary NLTK data.")


# --- Pydantic Model for API Input ---
class SMSMessage(BaseModel):
    message: str

# --- API Instance ---
app = FastAPI()

# --- Utility Function for Text Cleaning ---
# This function remains the same as it's not path-dependent.
def clean_message_for_inference(message: str) -> str:
    """Cleans a raw message for the AI model."""
    text = message.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    return " ".join(lemmatized_words)

# --- The Main API Endpoint ---
@app.post("/check_sms")
def check_sms_endpoint(sms: SMSMessage):
    raw_message = sms.message
    
    # Whitelisting Layer
    if check_whitelist(raw_message):
        logging.info(f"VERDICT: ALLOWED - REASON: whitelisted | MESSAGE: '{raw_message}'")
        return {"verdict": "allowed", "reason": "whitelisted"}

    # AI Classifier Layer
    if not classifier or not vectorizer:
        raise HTTPException(status_code=500, detail="Classifier service is not available.")
        
    cleaned_message = clean_message_for_inference(raw_message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = classifier.predict(vectorized_message)[0]

    # Determine the final verdict
    if prediction == 'Spam':
        verdict = 'blocked'
        reason = 'ai'
        logging.info(f"VERDICT: BLOCKED - REASON: ai | PREDICTION: Spam | MESSAGE: '{raw_message}'")
    else:
        verdict = 'allowed'
        reason = 'ai'
        logging.info(f"VERDICT: ALLOWED - REASON: ai | PREDICTION: {prediction} | MESSAGE: '{raw_message}'")
        
    return {"verdict": verdict, "reason": reason}