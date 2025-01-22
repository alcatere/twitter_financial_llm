from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorWithPadding

import evaluate

import numpy as np

# Load dataset
dataset_name = 'zeroshot/twitter-financial-news-sentiment'
dataset = load_dataset(dataset_name)  

# Sentiments
sentiments = {
    "LABEL_0": "Bearish", 
    "LABEL_1": "Bullish", 
    "LABEL_2": "Neutral"
}  

# Checkpoint
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_data(batch):
    return tokenizer(batch['text'], truncation=True)

tokenized_dataset = dataset.map(tokenize_data, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3, ignore_mismatched_sizes=True)

# Compute metrics
def compute_metrics(eval_pred):
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")
    metric4 = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions, references=labels,
                                average="weighted")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels,
                             average="weighted")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels,
                         average="weighted")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)[
        "accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy}

# Training arguments
training_args = TrainingArguments(
    'twitter_financial_sentiment',
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch", 
    push_to_hub=True
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train model
trainer.train()

trainer.push_to_hub('End of training')

# Save model
model.save_pretrained("./models/sentiment_twitter_financial_model")
tokenizer.save_pretrained("./models/sentiment_twitter_financial_tokenizer")

# Save the tokenizer to Hugging Face Hub
tokenizer.push_to_hub("alcatere/twitter_financial_sentiment")