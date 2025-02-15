import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import re
import torch.optim as optim
import torch.nn.functional as F

# Load the dataset
df = pd.read_csv('tripadvisor_hotel_reviews.csv')
df.head()

def preprocess_text_before_tokenization(text):
  # Remove URLs
  text = re.sub(r'https?://\S+|www\.\S+', '', text)

  # Remove HTML tags
  text = re.sub(r'<.*?>', '', text)
  return text

df['Review'] = df['Review'].apply(preprocess_text_before_tokenization)

#Assigning Sentiments for the ratings(1 to 5)(0 : Neg, 1 : Neutral, 2 : Positive)
df['Sentiment'] = df['Rating'].apply(lambda x: 0 if x < 3 else (2 if x > 3 else 1))

df.head()

"""Load Tokenizer and Model"""

tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model_base = RobertaModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

train_texts, val_texts, train_labels, val_labels = train_test_split(df['Review'], df['Sentiment'], test_size=0.2)

"""Defining the dataset"""

# Define your dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )

        return {
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(label, dtype=torch.long)
        }

#To Try

import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.roberta = model_base
        # Freeze RoBERTa parameters to prevent them from being updated during training
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.bi_gru = nn.GRU(768, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.2)
        self.leaky_relu = nn.LeakyReLU()
        self.out = nn.Linear(128*2, n_classes)

        # Attention Layer
        self.attention = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        gru_out, _ = self.bi_gru(last_hidden_state)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(gru_out).squeeze(-1), dim=-1)
        weighted_gru_out = torch.sum(gru_out * attention_weights.unsqueeze(-1), dim=1)

        norm_out = self.batch_norm(weighted_gru_out)
        dropout_out = self.dropout(norm_out)
        leaky_relu_out = self.leaky_relu(dropout_out)

        return self.out(leaky_relu_out)

"""Create Dataset and DataLoaders"""

# Create datasets and dataloaders
train_dataset = SentimentDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len=256)
val_dataset = SentimentDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len=256)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

"""Model Training"""

model = SentimentClassifier(n_classes=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

"""Training loop"""

def train_model(model, data_loader, optimizer, criterion, n_epochs, device):
    model.train()  # Put the model in training mode
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch: {epoch+1}, Loss: {total_loss / len(data_loader)}")

"""Evaluation"""

def evaluate_model(model, data_loader, criterion, device):
    model.eval()  # Put the model in evaluation mode
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")

"""Integrating training and Evaluation"""

n_epochs = 3

for epoch in range(n_epochs):
  print(f"Epoch {epoch+1}/{n_epochs}")
  print('-' * 10)

  # Training
  train_model(model, train_loader, optimizer, criterion, n_epochs, device)
  # Evaluate on the validation set
  evaluate_model(model, val_loader, criterion, device)

from sklearn.metrics import accuracy_score, classification_report
import torch

def predict(model, data_loader, device):
    model.eval()  # Put the model in evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    return true_labels, predictions

true_labels, predictions = predict(model, val_loader, device)  # Predict
accuracy = accuracy_score(true_labels, predictions)  # Calculate accuracy
print(f'Final Accuracy: {accuracy * 100:.0f}%')  # Print accuracy

report = classification_report(true_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])  # Generate report
print(report)

"""After training for 3 epochs, the model predicts with an accuracy of 87%."""

def predict_sentiment(model, tokenizer, sentences, device):
    model.eval()  # Ensure the model is in evaluation mode.

    # Preprocess each sentence in the list
    preprocessed_sentences = [preprocess_text_before_tokenization(sentence) for sentence in sentences]

    # Tokenize sentences for batch processing
    encoded_inputs = tokenizer(
        preprocessed_sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True
    )

    # Move the tokenized inputs to the device
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    # Predict without assuming a .logits attribute
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # The outputs are assumed to be the logits directly
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted_classes = torch.max(probabilities, dim=1)

    # Map predicted class indices to labels
    class_names = ['Negative', 'Neutral', 'Positive']
    predicted_labels = [class_names[class_id] for class_id in predicted_classes.cpu().numpy()]

    return predicted_labels, probabilities.cpu().numpy()

test_sentences = ["The hotel had exceptional service, and the rooms were incredibly clean and spacious, offering breathtaking views of the city skyline. Definitely looking forward to staying here again on my next trip!", "The hotel is conveniently located near the train station, but the Wi-Fi connection was quite slow and unreliable.", "The room was exactly as described, with no surprises during our stay."]

predicted_labels, probabilities = predict_sentiment(model, tokenizer, test_sentences, device)

for sentence, label, prob in zip(test_sentences, predicted_labels, probabilities):
    print(f"Sentence: {sentence}\nPredicted sentiment: {label}\nProbabilities: {prob}\n")

torch.save(model.state_dict(), 'sentiment_analysis_model.pth')

