# Twitter Financial Analysis 🚀

This project uses Natural Language Processing (NLP) to analyze financial-related tweets and classify their sentiment into categories (e.g., Bearish, Neutral, Bullish) to gain insights into public sentiment around financial topics.

### Streamlit application
https://twitter-financial-sentiment.streamlit.app/

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model and Training](#model-and-training)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
Financial sentiment analysis is crucial for understanding how markets respond to news and trends. By analyzing tweets, this project aims to:
- Provide insights into market sentiment.
- Predict how public sentiment correlates with market movements.
- Assist in decision-making for investors and analysts.

---

## Features
- **Sentiment Analysis**: Classifies tweets into Positive, Neutral, or Negative categories.
- **Customizable Models**: Train or fine-tune on financial-specific data for better accuracy.
- **Interactive Predictions**: Includes a Streamlit app for real-time sentiment analysis.
- **Evaluation Metrics**: Reports accuracy, precision, recall, and F1-score for performance tracking.

---

## Dataset
The dataset consists of financial tweets, preprocessed and labeled into three sentiment categories:
- **Bullish**: Indicates optimism. Investors believe prices will increase in the future.
- **Neutral**: Tweets expressing neutral or factual information.
- **Bearish**:  Indicates pessimism. Investors believe prices will decrease in the future.

### Sample Data Format
| Tweet                                 | Sentiment   |
|---------------------------------------|-------------|
| "$SAN: Deutsche Bank cuts to Hold"    | Bearish    |
| "Market seems stable."                | Neutral    |
| "AMD +1.3% after Cowen's target lift" | Bullish    |

---

## Model and Training
- **Model Architecture**: The model is based on a transformer architecture (e.g., `DistilBERT`).
- **Pretrained Checkpoint**: Fine-tuned using `distilbert-base-uncased-finetuned-sst-2-english` as the base model.
- **Training Pipeline**:
  - Tokenization with Hugging Face’s `AutoTokenizer`.
  - Fine-tuning using `AutoModelForSequenceClassification`.
  - Optimized with techniques like learning rate scheduling and early stopping.

---

## Installation
### Prerequisites
Ensure you have Python 3.8 or later installed.

### Clone the Repository
```bash
git clone https://github.com/alcatere/twitter_financial_llm.git
cd twitter_financial_llm

