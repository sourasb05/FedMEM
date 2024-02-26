import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import os

class FeatureDataset(Dataset):
    def __init__(self, features_folder, annotations_file):
        self.features_folder = features_folder
        self.annotations = pd.read_csv(annotations_file, header=None, names=['filename', 'label'])
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        filename, label = self.annotations.iloc[idx]
        feature_path = os.path.join(self.features_folder, filename)
        feature = torch.load(feature_path)
        return feature, label




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() 
        resnet = torchvision.models.resnet50(pretrained=True)
        self.avgpool = nn.Sequential(list(resnet.children())[-2])
        # Define linear layers and ReLU activation
        self.fc1 = nn.Linear(2048, 512)  # Assuming the output of avgpool is [batch_size, 2048, 1, 1]
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # print("2:",x.shape)

        features_1d = self.avgpool(x)
        # print("3:",features_1d.shape)

        features_1d = torch.flatten(features_1d, 1)  # Flatten the features
        # print("4:",features_1d.shape)
        # input("press")
        x = self.fc1(features_1d)
        # print("5:",x.shape)
        
        x = self.relu(x)
        # print("6:",x.shape)
        
        x = self.fc2(x)
        # print("7:",x.shape)
        
        x = self.relu(x)
        # print("8:",x.shape)
        
        return x
      
  
# Training function
def train(model, device, criterion, optimizer, splits, dataset,  num_epochs=20):

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"Fold: {fold}")
        model.train()
        # Split the dataset 

        # Sample elements randomly from a given list of ids, no replacement.
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
      
        # Define data loaders for training and testing data in this fold
        
        train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            # Validate the model
            val_accuracy, validation_loss, precision, recall, f1 = validate(model, val_loader, device)
            print(f"Epoch {epoch+1}, Validation Loss: {validation_loss}")
            print(f'Validation Accuracy: {val_accuracy:.2f}%')
            print(f'Precision: {precision:.2f}%')
            print(f'Recall: {recall:.2f}%')
            print(f'f1: {f1:.2f}%')


def validate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    total_loss = 0.0
    with torch.no_grad():  # Inference mode, gradients not needed
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())  # Collect true labels
            y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels


    # accuracy = 100 * correct / total
    
    # Convert collected labels to numpy arrays for metric calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    validation_loss = total_loss / len(dataloader)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
    
    return accuracy, validation_loss, precision, recall, f1





# Load dataset
features_folder = '/proj/sourasb-220503/FedMEM/dataset/r3_mem_ResNet50_features'
annotations_file = '/proj/sourasb-220503/FedMEM/dataset/tensor_train_val_1.csv'
dataset = FeatureDataset(features_folder, annotations_file)

# Dataset size
dataset_size = len(dataset)  # Replace with your dataset size

# Number of splits
k_folds = 5

# Create KFold object
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Split your dataset indices into k_folds
splits = kf.split(np.arange(dataset_size))
print(splits)

# Split dataset into training and validation
# train_dataset, val_dataset = train_test_split(dataset, test_size=0.25, random_state=42)

# DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=124, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

# Assuming features from ResNet50 (2048 features) and an example of 10 classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Model().to(device)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the second last layer
for param in model.fc1.parameters():
    param.requires_grad = True

# Unfreeze the last layer
for param in model.fc2.parameters():
    param.requires_grad = True


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([
    {'params': model.fc1.parameters()},
    {'params': model.fc2.parameters()}
], lr=0.05)  # Only optimize the last layer


# Train the model
train(model, device, criterion, optimizer, splits, dataset)
