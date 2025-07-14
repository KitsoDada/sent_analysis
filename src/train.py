import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample dataset
data = {
    'text': [
        'I love this product!',
        'This is the worst service ever.',
        'Amazing experience.',
        'I will never buy this again.',
        'Highly recommend it!',
        'Terrible quality.'
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)

# Text preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'app/model.pkl')
joblib.dump(vectorizer, 'app/vectorizer.pkl')

print("Model trained and saved.")
