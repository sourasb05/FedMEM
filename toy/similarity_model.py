import torch
import torch.nn as nn
import numpy as np

def flatten_params(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

def cosine_similarity(model1, model2):
    params1 = flatten_params(model1)
    params2 = flatten_params(model1)
    similarity = torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))
    return similarity.item()

# Example models
model1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
model2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

# Compute cosine similarity
similarity = cosine_similarity(model1, model2)
print("Cosine Similarity:", similarity)