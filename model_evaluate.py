from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from datasets import load_dataset

import numpy as np

import evaluate

import pandas as pd

dataset_validation = load_dataset('zeroshot/twitter-financial-news-sentiment', split='validation')

# Replace 'saved_model_path' with the path to your saved model and tokenizer
model_path = "./models/sentiment_twitter_financial_model"
tokenizer_path = "./models/sentiment_twitter_financial_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define label mapping
labels = {
    "LABEL_0": "Bearish", 
    "LABEL_1": "Bullish", 
    "LABEL_2": "Neutral"
}  

def tokenize_data(batch):
    return tokenizer(batch['text'], truncation=True)

tokenized_dataset = dataset_validation.map(tokenize_data, batched=True)
# Drop features that are not needed
tokenized_dataset = tokenized_dataset.remove_columns(['text'])

# Define a function to compute the accuracy of the model
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
)


trainer = Trainer(
    model,
    training_args,
    tokenizer=tokenizer
)

# Evaluate the model on the validation set
predictions = trainer.predict(test_dataset=tokenized_dataset)


preds = F.softmax(torch.tensor(predictions.predictions), dim=-1)
preds = np.argmax(preds, axis=1).tolist()

# Compute metrics
def compute_metrics(preds, predictions):

    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    precision = precision_metric.compute(predictions=preds, references=predictions, average='weighted')['precision']
    recall = recall_metric.compute(predictions=preds, references=predictions, average='weighted')['recall']
    f1 = f1_metric.compute(predictions=preds, references=predictions, average='weighted')['f1']
    accuracy = accuracy_metric.compute(predictions=preds, references=predictions)['accuracy']
    
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
    return precision, recall, f1, accuracy


precision, recall, f1, accuracy = compute_metrics(preds, predictions.label_ids)

# Save the metrics in a CSV file
pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}, index=[0]).to_csv("metrics.csv", index=False)
