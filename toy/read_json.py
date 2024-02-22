import json

# Replace 'your_file_path.json' with the actual path to your JSON file
file_path = '/proj/sourasb-220503/FedMEM/FedMEM_dataset/train/fedr3_train.json'

# Open the JSON file for reading
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data = json.load(file)
    
    # Print the data
    print(data)