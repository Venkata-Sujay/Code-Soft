import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
exit() # Type this to exit the Python interpreter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# --- 0. Ensure NLTK Data is Downloaded (Run this in your local environment) ---
# You might need to run these lines once in your Python environment:
# nltk.download('stopwords')
# nltk.download('wordnet')
# If you get a ModuleNotFoundError for nltk, run: pip install nltk

# --- 1. Load the datasets ---
train_df = pd.read_csv('train_data.txt', sep=' ::: ', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')
test_df = pd.read_csv('test_data.txt', sep=' ::: ', header=None, names=['ID', 'TITLE', 'DESCRIPTION'], engine='python')

print("Train Data Info:")
print(train_df.info())
print("\nTrain Data Head:")
print(train_df.head())

print("\nTest Data Info:")
print(test_df.info())
print("\nTest Data Head:")
print(test_df.head())

# --- 2. Text Preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

train_df['cleaned_description'] = train_df['DESCRIPTION'].apply(preprocess_text)
test_df['cleaned_description'] = test_df['DESCRIPTION'].apply(preprocess_text)

# --- 3. Multi-label Binarization of Genres ---
mlb = MultiLabelBinarizer()
train_df['GENRE_LIST'] = train_df['GENRE'].apply(lambda x: x.split('|'))
y_train = mlb.fit_transform(train_df['GENRE_LIST'])

# --- 4. Feature Engineering (TF-IDF Vectorization) ---
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['cleaned_description'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['cleaned_description'])

# --- 5. Model Selection and Training ---
print("\nTraining MultiOutputClassifier with LinearSVC...")
# Set max_iter for convergence, and dual=False for n_samples > n_features in newer sklearn versions
svm_classifier = MultiOutputClassifier(LinearSVC(random_state=42, dual=False, max_iter=2000))
svm_classifier.fit(X_train_tfidf, y_train)

# --- 6. Make Predictions on the test data ---
y_pred_test = svm_classifier.predict(X_test_tfidf)

# Convert predictions back to genre labels
predicted_genres = mlb.inverse_transform(y_pred_test)

# Format the predictions for output
predicted_genres_str = ['|'.join(genres) if genres else '' for genres in predicted_genres]

# Create a DataFrame for the output
output_df = pd.DataFrame({
    'ID': test_df['ID'],
    'TITLE': test_df['TITLE'],
    'PREDICTED_GENRE': predicted_genres_str,
    'DESCRIPTION': test_df['DESCRIPTION']
})

# Save the predictions to a CSV file
output_file_name = 'movie_genre_predictions.csv'
output_df.to_csv(output_file_name, index=False)

print(f"\nPredictions saved to {output_file_name}")
print("\nSample of predicted genres:")
print(output_df.head())