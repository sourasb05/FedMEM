import json
my_file = "/proj/sourasb-220503/Predicting-Event-Memorability/FedMEM/dataset/train/fedr3_train.json"
# 1. Open the JSON file
clients = []
train_data = {}
test_data = {}
    
with open(my_file, 'r') as file:
    # 2. Load the JSON data
    cdata = json.load(file)
    clients.extend(cdata['users'])
    train_data.update(cdata['user_data'])
    print(clients)
