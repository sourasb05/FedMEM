import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label



class FeaturesDataset(Dataset):
    def __init__(self, csv_file, features_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            features_dir (string): Directory with all the feature files.
        """
        self.features_labels = pd.read_csv(csv_file)
        self.features_dir = features_dir

    def __len__(self):
        return len(self.features_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature_name = os.path.join(self.features_dir, self.features_labels.iloc[idx, 0])
        feature = torch.load(feature_name)
        label = self.features_labels.iloc[idx, 1]
        feature = F.interpolate(feature.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
        feature = feature.repeat(1, 3, 1, 1)

        return feature, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def evaluate_model(model, dataloader, device):
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
    precision = precision_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
    recall = recall_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
    
    return accuracy, validation_loss, precision, recall, f1

# Evaluate on the validation set



labels_df = pd.read_csv('/proj/sourasb-220503/FedMEM/dataset/refined_training_file.csv')

# Assuming you have initialized your dataset
dataset = CustomDataset(dataframe=labels_df, img_dir='/proj/sourasb-220503/FedMEM/dataset/r3_refined/', transform=transform)

# features_dir = '/proj/sourasb-220503/FedMEM/dataset/r3_mem_ResNet50FC_features'  # Adjust this path
# csv_file = '/proj/sourasb-220503/FedMEM/dataset/tensor_training_file.csv'    # Adjust this path

# dataset = FeaturesDataset(csv_file=csv_file, features_dir=features_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# Splitting the dataset
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # Adjust `number_of_classes` as per your dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Example training loop
for epoch in range(10):  # Loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}")
    val_accuracy, validation_loss, precision, recall, f1 = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch+1}, Validation Loss: {validation_loss}")
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    print(f'Precision: {precision:.2f}%')
    print(f'Recall: {recall:.2f}%')
    print(f'f1: {f1:.2f}%')


print('Finished Training')
