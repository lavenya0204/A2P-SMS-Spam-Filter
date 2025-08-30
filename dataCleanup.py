import pandas as pd
#read the dataset
df = pd.read_csv("data/message_dataset_50k.csv")
df.head(10)

#print the no. of rows,columns
print(df.shape)
#print the sum of null values for each column
print(df.isnull().sum())
#print the no. of rows for each class
print(df['Category'].value_counts())

#check for duplicates
df["Message"].duplicated().sum()

#print the no. of duplicated messages
message_counts = df["Message"].value_counts()
duplicates_with_counts = message_counts[message_counts > 1]
print(duplicates_with_counts)

#remove duplicates
df = df.drop_duplicates(subset=["Message"])
df.shape

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib




nltk.download('stopwords')
nltk.download('wordnet')

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_sms_message(message: str) -> str:
    
    if not isinstance(message, str):
        return ""

    # 1. Convert to lowercase
    text = message.lower()
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 3. Remove punctuation and digits.
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    # 6. Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    return " ".join(lemmatized_words)


# Create Cleaned_message column
df['Cleaned_message'] = df['Message'].apply(clean_sms_message)
print(df.head(15))

df.to_csv("data/message_dataset_cleaned.csv", index=False)


# Define features (X) and labels (y)
X = df['Cleaned_message']
y = df['Category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("-" * 50)

# TF-IDF Vectorizer will convert the text data into a numerical matrix.
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize the Classifier 
classifier = MultinomialNB()

# Train the model
classifier.fit(X_train_vec, y_train)

print("Model training complete.")
print("-" * 50)

# Make predictions on the test set
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test_vec)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Get the unique classes from the labels for confusion matrix
class_labels = np.unique(y_test)

# Create a heatmap for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("Confusion Matrix:")
print(cm)
print("-" * 50)

import os
import joblib

# Define folder and file paths relative to the project root
model_folder = 'model'
model_filename = os.path.join(model_folder, 'sms_classifier_model.joblib')
vectorizer_filename = os.path.join(model_folder, 'tfidf_vectorizer.joblib')

# Create 'model' folder if it doesn't exist
os.makedirs(model_folder, exist_ok=True)

# Save the trained classifier
joblib.dump(classifier, model_filename)

# Save the fitted vectorizer
joblib.dump(vectorizer, vectorizer_filename)

print(f"Model saved to: {os.path.abspath(model_filename)}")
print(f"Vectorizer saved to: {os.path.abspath(vectorizer_filename)}")
