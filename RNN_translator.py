from spellchecker import SpellChecker
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch.optim as optim
from collections import Counter

# Load Data
data = pd.read_csv('concatenated_data/c2.csv')


X = data.drop(['sign'], axis='columns').values 
y = data['sign']


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode labels as integers



# Model parameters

#will be constant depending on what why decide and how many signs there are
input_size = X.shape[1]  # Number of features
sequence_length = 20
num_classes = 23 #number of signs


# parameters to play around with to get higher test accuracy
num_layers = 4
hidden_size = 256 #nodes in hidden layer
lr = .001

num_epochs = 100


num_sequences = X.shape[0] // sequence_length

X = X[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, input_size)
y = y[:num_sequences * sequence_length]
y = y[::sequence_length] #take every sequence_length element


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle= True,random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out



model = RNN(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr) #adam is best i found

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


#Test Loop
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")






