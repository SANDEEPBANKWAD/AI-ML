# Install Required Libraries
# pip install pandas scikit-learn nltk
 # Import Libraries
import pandas as pd
import numpy as np
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK Data
nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
df = pd.read_csv("drugsComTrain_raw.csv")

print(df.head())
print(df.columns)

# Convert Ratings to Sentiment Labels
def rating_to_sentiment(rating):
    if rating >= 7:
        return 'positive'
    elif rating <= 4:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['rating'].apply(rating_to_sentiment)

# Keep only positive & negative (optional)
df = df[df['sentiment'] != 'neutral']

# Text Cleaning Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return ' '.join(words)

df['clean_review'] = df['review'].apply(clean_text)

# Train-Test Split
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test Custom Input
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    return model.predict(vector)[0]

print(predict_sentiment("This medicine worked perfectly for me"))
print(predict_sentiment("Terrible side effects, not recommended"))

