import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Step 1: Load Pre-Extracted Features and Labels
def load_data(features_path, labels_path):
    features = torch.load(features_path)
    labels = torch.load(labels_path)
    return TensorDataset(features, labels)

# Assuming the paths to the training and validation data (replace with your paths)
train_features_path = 'path/to/train_features.pt'
train_labels_path = 'path/to/train_labels.pt'
val_features_path = 'path/to/val_features.pt'
val_labels_path = 'path/to/val_labels.pt'

train_data = load_data(train_features_path, train_labels_path)
val_data = load_data(val_features_path, val_labels_path)

# DataLoader
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Step 2: Define the Neural Network with a Fully Connected Layer
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        return x

# Initialize the model
# Adjust input_size and num_classes according to your dataset
input_size = 2048  # For ResNet50, the feature size before the final FC layer is 2048
num_classes = 100  # Example: 100 classes
model = Classifier(input_size, num_classes)

# Step 3: Train the Model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Step 4: Evaluate the Model
def evaluate_model(model, val_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on the validation set: {100 * correct / total:.2f}%')

# Run training and evaluation
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, val_loader)