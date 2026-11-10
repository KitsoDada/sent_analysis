import re
from functools import lru_cache
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
THEME_PATTERNS = {
    "shipping": r"\b(ship(?:ping)?|deliver(?:y|ed)?|arriv(?:ed|al)|tracking|late|delay(?:ed)?)\b",
    "quality": r"\b(quality|broken|broke|damaged|faulty|defect(?:ive)?|durab(?:le|ility)|material|fell apart)\b",
    "customer_service": r"\b(customer service|support|staff|seller|agent|response|ignored|rude|helpful)\b",
    "sizing_fit": r"\b(size|sizing|fit|fits|small|large|tight|loose|length)\b",
    "returns_refunds": r"\b(return|refund|exchange|money back|reimburse)\b",
    "packaging": r"\b(packag(?:e|ing)|box|wrapped|scratched)\b",
    "price_value": r"\b(price|cost|expensive|cheap|value|worth|overpriced)\b",
    "functionality": r"\b(work(?:s|ed|ing)?|function|battery|performance|use|usable|stopped)\b",
}


@lru_cache(maxsize=1)
def load_artifacts():
    model_path = ROOT / "app" / "model.pkl"
    vectorizer_path = ROOT / "app" / "vectorizer.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        raise RuntimeError("Model artifacts missing. Run `python -m src.train` first.")
    return joblib.load(model_path), joblib.load(vectorizer_path)


def analyze_review(text, rating=None):
    text = str(text).strip()
    if not text:
        raise ValueError("Review text is required.")
    model, vectorizer = load_artifacts()
    features = vectorizer.transform([text])
    predicted = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
    confidence = float(max(probability)) if probability is not None else None

    # Ratings provide useful context; the trained text model remains the primary signal.
    if rating not in (None, ""):
        rating = int(rating)
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5.")
        if rating <= 2:
            sentiment = "negative"
        elif rating == 3:
            sentiment = "neutral"
        else:
            sentiment = "positive"
    elif confidence is not None and confidence < 0.58:
        sentiment = "neutral"
    else:
        sentiment = "positive" if predicted == 1 else "negative"

    lowered = text.lower()
    themes = [name for name, pattern in THEME_PATTERNS.items() if re.search(pattern, lowered)]
    return {"sentiment": sentiment, "confidence": round(confidence, 4) if confidence is not None else None, "themes": themes or ["general"]}


def create_reply_draft(analysis, review_text):
    sentiment = analysis["sentiment"]
    themes = set(analysis["themes"])
    if sentiment == "positive":
        return "Thank you for sharing your experience! We're glad you're happy with your purchase and appreciate your support."
    if "shipping" in themes:
        return "We're sorry your delivery experience did not meet expectations. Please contact our support team with your order details so we can look into this."
    if "quality" in themes or "functionality" in themes:
        return "We're sorry the product did not meet expectations. Please contact our support team with your order details so we can help find a solution."
    if "returns_refunds" in themes:
        return "We're sorry this has been frustrating. Please contact our support team with your order details so we can review the available return or refund options."
    if "customer_service" in themes:
        return "We're sorry you did not receive the support you expected. Please contact our support team so we can understand what happened and help."
    return "Thank you for the feedback. We're sorry your experience fell short. Please contact our support team with your order details so we can help."
