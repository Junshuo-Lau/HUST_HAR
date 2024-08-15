# Copyright (c) 2024.08.14, Junshuo Liu.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from thop import profile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import time
import pywt
from einops import rearrange
from MambaSimple import Mamba

# Load the dataset and labels
def load_hust_har_data(data_path, label_path):
    data = torch.load(data_path)
    labels = torch.load(label_path)
    return data, labels

# Normalize the data
def data_norm(data):
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    data_normalized = (data - mean) / (std + 1e-6)
    return data_normalized

# Load the data and labels
data_path = ".../HUST_HAR_dataset.pt"
label_path = ".../HUST_HAR_labels.pt"

data, labels = load_hust_har_data(data_path, label_path)

# Normalize the data
data_normed = data_norm(data).permute(0, 2, 1)

# Combine data and labels into a single dataset for splitting
dataset = TensorDataset(data_normed, labels)

# Split the dataset into training and testing (80:20) randomly
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class EmbeddingLayer(nn.Module):
    def __init__(self, num_features, embed_dim, max_len=500):
        super(EmbeddingLayer, self).__init__()
        self.positional_embedding = PositionalEmbedding(embed_dim, max_len)
        self.mlp_block = MLPBlock(num_features, embed_dim)

    def forward(self, x):
        pe = self.positional_embedding(x)
        x = self.mlp_block(x) + pe
        return x

class MambaStack(nn.Module):
    def __init__(self, num_features, d_model, d_state, d_conv, expand, num_layers, num_classes, use_layer_norm=True, max_len=500):
        super(MambaStack, self).__init__()
        self.use_layer_norm = use_layer_norm

        self.embedding_layer = EmbeddingLayer(num_features=num_features, embed_dim=d_model, max_len=max_len)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            if self.use_layer_norm:
                self.layers.append(nn.LayerNorm(d_model))
            mamba_layer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.layers.append(mamba_layer)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.embedding_layer(x)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            elif isinstance(layer, Mamba):
                x = layer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

num_layers = 3
num_classes = 6
model = MambaStack(num_features=135, d_model=64, d_state=16, d_conv=4, expand=2, num_layers=num_layers, num_classes=num_classes, use_layer_norm=True, max_len=500)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, train_loader, criterion, optimizer, num_epochs=40, early_stopping_threshold=0.01):
    model.train()
    losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0

        for data, labels in train_loader:
            # print(data.shape)
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        average_loss = epoch_loss / len(train_loader)
        losses.append(average_loss)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Time: {epoch_time:.4f} seconds')


        if average_loss < early_stopping_threshold:
            print(f"Stopping early at epoch {epoch+1} with loss {average_loss:.4f}")
            break
    
    # Plotting training loss
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return losses, epoch_times

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
losses, epoch_times = train_model(model, train_loader, criterion, optimizer)

average_training_time = sum(epoch_times) / len(epoch_times)
print(f'Average training time per epoch: {average_training_time:.4f} seconds')

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_probs.extend(outputs.softmax(dim=1).cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    specificity = np.diag(cm) / np.sum(cm, axis=0)
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")

    # Display confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, precision, f1, recall, specificity, auc

metrics = evaluate_model(model, test_loader)
print(f'Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, F1 Score: {metrics[2]:.4f}, Recall: {metrics[3]:.4f}, AUC: {metrics[5]:.4f}')

# num_layers = 3
# num_classes = 6
# model = MambaStack(num_features=270, d_model=64, d_state=16, d_conv=4, expand=2, num_layers=num_layers, num_classes=num_classes, use_layer_norm=True, max_len=1000).to(device)

# input = torch.randn(32, 1000, 270).to(device)
# macs, params = profile(model, inputs=(input, ), verbose=False)

# print(f"MACs: {macs}")
# print(f"Params: {params}")

# print('MACs = ' + str(macs/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')