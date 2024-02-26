import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np
import pandas as pd
import random
from tqdm import trange
import json
from sklearn.model_selection import train_test_split
import random
import time

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

