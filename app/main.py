import streamlit as st
import joblib

model = joblib.load('app/model.pkl')
vectorizer = joblib.load('app/vectorizer.pkl')

st.title("Sentiment Analysis")

text = st.text_input("Enter your text")
if st.button("Predict"):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    st.success(f"Sentiment: {sentiment}")
