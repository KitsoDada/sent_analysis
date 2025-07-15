import streamlit as st
import requests

# Streamlit page setup
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title("💬 Sentiment Analysis App")

# Text input from user
user_input = st.text_area("Enter your review:", height=150)

# API endpoint
API_URL = "https://sentanalysisapi-b7g0fwbccfcuexay.eastus-01.azurewebsites.net/predict"

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        try:
            # Make POST request to the Flask API
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                json={"text": user_input}
            )

            if response.status_code == 200:
                result = response.json()
                sentiment = result.get("sentiment", "unknown")
                
                if sentiment == "positive":
                    st.success("✅ Sentiment: Positive")
                elif sentiment == "negative":
                    st.error("❌ Sentiment: Negative")
                elif sentiment == "neutral":
                    st.info("ℹ️ Sentiment: Neutral")
                else:
                    st.warning(f"Unexpected sentiment: {sentiment}")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Failed to reach API: {e}")
