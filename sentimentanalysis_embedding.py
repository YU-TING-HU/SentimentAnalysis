# -*- coding: utf-8 -*-
"""
SentimentAnalysis_Embedding.ipynb
"""

import numpy as np
import pandas as pd
import pickle
from collections import Counter
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer

# Load the dataset
twitter_file = '/content/Tweets.csv'
df = pd.read_csv(twitter_file).dropna()
df

# Create Target Variable
cat_id = {'neutral': 1,
          'negative': 0,
          'positive': 2}
df['class'] = df['sentiment'].map(cat_id)

# Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 80
MAX_FEATURES = 10

# Load the embedding model
emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')

# Example to show the shape of embeddings
sentences = ["Each sentence is converted"]  
embeddings = emb_model.encode(sentences) # Encode the example sentence
print(embeddings.squeeze().shape) # Print the shape of the embedding

# prepare X and y
X = emb_model.encode(df['text'].values) # Encode the tweet texts
# Save 
with open("/content/tweets_X.pkl", "wb") as output_file:
    pickle.dump(X, output_file)

# Load
with open("/content/tweets_X.pkl", "rb") as input_file:
    X = pickle.load(input_file)
y = df['class'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

class SentimentData(Dataset):
    """Custom Dataset class for sentiment analysis."""
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.len = len(self.X) # Length of the dataset

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Return a single data point and label
        return self.X[index], self.y[index] 

# Create dataset objects
train_ds = SentimentData(X= X_train, y = y_train)
test_ds = SentimentData(X_test, y_test)

# Create data loaders
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=15000)

# Model
class SentimentModel(nn.Module):
    """Simple neural network model for sentiment analysis."""
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN = 10):
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x

# Model, Loss and Optimizer
model = SentimentModel(NUM_FEATURES = X_train.shape[1], NUM_CLASSES = 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

# Model Training
train_losses = []
for e in range(NUM_EPOCHS):
    curr_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad() # Zero the gradients
        y_pred_log = model(X_batch) # Forward pass
        loss = criterion(y_pred_log, y_batch.long())  # Compute the loss

        curr_loss += loss.item() # Accumulate the loss
        loss.backward() # Backward pass
        optimizer.step() # Update the weights
    train_losses.append(curr_loss) # Append the current loss
    print(f"Epoch {e}, Loss: {curr_loss}")

# Plot the training losses
sns.lineplot(x=list(range(len(train_losses))), y= train_losses)

# Model Evaluation
with torch.no_grad(): # Disable gradient computation
    for X_batch, y_batch in test_loader:
        y_test_pred_log = model(X_batch)
        y_test_pred = torch.argmax(y_test_pred_log, dim = 1)

# Convert predictions to numpy array
y_test_pred_np = y_test_pred.squeeze().cpu().numpy()
acc = accuracy_score(y_pred=y_test_pred_np, y_true = y_test)
print(f"The accuracy of the model is {np.round(acc, 3)*100}%.")

# Naive classifier accuracy
most_common_cnt = Counter(y_test).most_common()[0][1]
print(f"Naive Classifier: {np.round(most_common_cnt / len(y_test) * 100, 1)} %")