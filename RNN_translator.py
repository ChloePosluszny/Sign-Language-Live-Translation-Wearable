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
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Load Data
data = pd.read_csv('concatenated_data/c2.csv')

X = data.drop(['sign'], axis='columns').values 
y = data['sign']


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode labels as integers

#will be constant depending on what why decide and how many signs there are
input_size = X.shape[1]  # Number of features
num_classes = 23 #number of signs
sequence_length = 1

# parameters to play around with to get higher test accuracy
num_layers = 4
hidden_size = 256
lr = .00085

num_epochs = 300


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
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
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
train_epoch_list = []
train_loss_list = []
train_acc_list = []

test_loss_list = []
test_acc_list = []

best_test_acc = 0
best_test_acc_epoch = 0

model_best = None
for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    _, predicted = outputs.max(1)
    train_total += y_train_t.size(0)
    train_correct += predicted.eq(y_train_t).sum().item()
    
    train_epoch_list.append(epoch)
    train_loss_list.append(loss.item())
    train_acc_list.append(100. * train_correct / train_total)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_loss = criterion(test_outputs, y_test_t).item()
        _, test_predicted = torch.max(test_outputs, 1)
        test_accuracy = (test_predicted == y_test_t).sum().item() / y_test_t.size(0)
        
        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy * 100)
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_test_acc_epoch = epoch
            model_best = model
            torch.save(model.state_dict(), "RNN_model.pth")
            

    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Acc: {100 * train_correct / train_total:.4f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy * 100:.2f}%")



model_best = RNN(input_size, hidden_size, num_layers, num_classes)
model_best.load_state_dict(torch.load("RNN_model.pth"))

model_best.eval()
with torch.no_grad():
    outputs = model_best(X_test_t)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
    print(f"Best Test Accuracy: {accuracy * 100:.2f}% at epoch {best_test_acc_epoch +1}")



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_loss_list, label='Training Loss')
plt.plot(range(1, num_epochs+1), test_loss_list, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(visible=True)
plt.title('Loss Graph')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_acc_list, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), test_acc_list, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(visible=True)
plt.title('Accuracy Graph')
plt.legend()
plt.gca().yaxis.set_major_formatter(PercentFormatter()) 

plt.tight_layout()
plt.show()





