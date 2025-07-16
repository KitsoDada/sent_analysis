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
        'Terrible quality.',
        'Bad customer support.',
        'Good value for money.',
        'The product broke after one use.',
        'Fast delivery and great packaging.',
        'This is disappointing.',
        'Excellent service.',
        'Awful, just awful.',
        'Very bad.',
        'Not worth the price.',
        'Best purchase ever!',
        'Great performance!',
        'Poorly built and weak.',
        'Extremely satisfied.',
        'It does not work',
        'It is just okay',
        'need to improve.',
    ],
    'label': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
              0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]
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
