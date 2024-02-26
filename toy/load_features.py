import torch

# Load the .pt file
file_path = '/proj/sourasb-220503/FedMEM/dataset/r3_mem_ResNet50FC_features/NEUR16_scan1_img01_day22_S.pt'
data = torch.load(file_path)

# Check the type of the loaded data
print(data)
print(type(data))
print(data.shape)
