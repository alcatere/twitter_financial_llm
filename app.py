
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load your trained model and tokenizer
model_path = "./models/sentiment_twitter_financial_model"  # Path to your trained model
tokenizer_path = "./models/sentiment_twitter_financial_tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define label mapping
labels = {
    0: "Bearish", 
    1: "Bullish", 
    2: "Neutral"
}  

# Streamlit app
st.title("Text Sentiment Classifier from Twitter Financial News")
st.write('''Enter a text below to classify its sentiment. \n
         The classifier can predict one of the following sentiments: Bearish, Bullish, or Neutral.''')

# Input text
user_input = st.text_area("Input Text", placeholder="Type your financial text here...")

# Predict button
if st.button("Classify"):
    if user_input.strip():
        # Tokenize input text
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Get predictions
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = F.softmax(logits, dim=-1).squeeze()
            predicted_class_id = torch.argmax(probabilities).item()
            predicted_label = labels[predicted_class_id]

        # Display results
        st.subheader("Prediction")
        st.write(f"**Predicted Sentiment:** {predicted_label}")
        st.write("**Probabilities:**")
        for label_id, prob in enumerate(probabilities):
            st.write(f"- {labels[label_id]}: {prob:.2f}")
    else:
        st.warning("Please enter some text to classify!")