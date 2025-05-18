import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

nltk_data_paths = ['C://nltk_data', 'E://nltk_data', 'D://nltk_data']
for path in nltk_data_paths:
    if os.path.exists(path):
        nltk.data.path.append(path)

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.word_tokenize("example")
except LookupError:
    nltk.download('punkt')


data = pd.DataFrame({
    'review': [
        "I loved this movie, it was fantastic and exciting!",
        "Absolutely terrible. I hated every moment of it.",
        "What a great film! Will watch it again.",
        "Worst movie ever. Waste of time.",
        "Awesome storyline and brilliant acting.",
        "Not good, very boring and dull.",
        "A masterpiece, beautifully directed.",
        "Bad acting and weak plot.",
        "Highly recommend this movie!",
        "I do not recommend this movie."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})


def preprocess_text(text):

    text = re.sub(r'[^a-zA-Z\s]', '', text)
   
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

data['cleaned_review'] = data['review'].apply(preprocess_text)

X = data['cleaned_review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = SVC(kernel='linear')  # Using a linear kernel for simplicity
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)

print("Classification Report (Support Vector Machine):")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Example Prediction
sample_reviews = [
    "I really enjoyed this film, it was amazing!",
    "It was a horrible movie, very disappointing."
]
sample_cleaned = [preprocess_text(r) for r in sample_reviews]
sample_features = vectorizer.transform(sample_cleaned)
sample_preds = model.predict(sample_features)

for review, pred in zip(sample_reviews, sample_preds):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Review: \"{review}\" => Sentiment Prediction (SVM): {sentiment}")