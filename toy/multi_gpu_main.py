import torch.nn.functional as F
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel as DataParallel
from torch.utils.data import DataLoader
import torchvision
# Initialize the distributed environment
dist.init_process_group("nccl", rank=0, world_size=4)

# Define the neural network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a model on each GPU
models = [Model().to(f"cuda:{i}") for i in range(4)]

# Wrap the model in the DataParallel class
models = [DataParallel(model) for model in models]

# Load the MNIST data
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the federated averaging algorithm
def federated_averaging(models):
    # Aggregate the model weights
    model_weights = []
    for model in models:
        model_weights.append(model.state_dict())

    avg_weights = {}
    for key in model_weights[0]:
        avg_weights[key] = torch.mean(torch.stack([model_weights[i][key] for i in range(len(model_weights))]), dim=0)

    # Update the model weights on each GPU
    for i in range(len(models)):
        models[i].load_state_dict(avg_weights)

# Train the model on each GPU
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to("cuda:0")
        target = target.to("cuda:0")

        # Train the model
        for model in models:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss = F.cross_entropy(model(data), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Aggregate the model weights
    federated_averaging(models)

# Save the aggregated model
torch.save(models[0].state_dict(), "model.pt")