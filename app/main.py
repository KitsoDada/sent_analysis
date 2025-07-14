from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('app/model.pkl')
vectorizer = joblib.load('app/vectorizer.pkl')

@app.route('/')
def home():
    return 'Sentiment Analysis API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No input text provided'}), 400

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    sentiment = 'positive' if prediction == 1 else 'negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
