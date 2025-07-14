import os
import joblib
import pytest
from src.train import model, vectorizer

def test_model_file_exists():
    assert os.path.exists("model.pkl"), "model.pkl not found"
    assert os.path.exists("vectorizer.pkl"), "vectorizer.pkl not found"

def test_model_prediction():
    clf = joblib.load("model.pkl")
    vec = joblib.load("vectorizer.pkl")

    test_text = ["I hated the experience", "Absolutely wonderful!"]
    X_test = vec.transform(test_text)
    predictions = clf.predict(X_test)

    assert all(p in [0, 1] for p in predictions), "Predictions must be 0 or 1"
