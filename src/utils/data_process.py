import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np
import random
from tqdm import trange
import json
from sklearn.model_selection import train_test_split
import random
import time


def read_data(args, current_directory):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    
    train_data_dir = os.path.join(current_directory,'FedMEM_dataset', 'train')
    test_data_dir = os.path.join(current_directory, 'FedMEM_dataset', 'test')
    
    print(train_data_dir)

    print(test_data_dir)

    clients = []
    train_data = {}
    test_data = {}
    
    
    train_file = train_data_dir +"/fedr3_train.json"
    print(train_file)
    
    with open(train_file, 'r') as f:
        cdata = json.load(f)
    clients.extend(cdata['users'])
    train_data.update(cdata['user_data'])

    # test_file = current_directory + "/FedMEM/dataset/train/fedr3_test.json"
    test_file = test_data_dir + "/fedr3_test.json"
    print(test_file)
    
    with open(test_file, 'r') as f:
        cdata = json.load(f)
    
    test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    print(len(clients))
    

    group = [[] for _ in range(args.num_teams)]
    grp_idx = 0
    total_clients = len(clients)
    print(total_clients)
    cl_per_group = total_clients // args.num_teams
    print(cl_per_group)
    try:
        if args.group_division == 0:
            for i in range(total_clients):
                group[grp_idx].append(i)
        elif args.group_division == 1:
            user_list = list(range(0,total_clients))
            user_list = random.shuffle(user_list)
            for i in range(total_clients):
                group[grp_idx].append(user_list[i])
                if i != 0 and (i+1) % cl_per_group == 0 :
                    grp_idx+=1
        elif args.group_division == 2:
            random.shuffle(clients)
            # print(clients)
            for i in range(total_clients):
                group[grp_idx].append(clients[i])

    except ValueError:
        raise ValueError("Wrong group division selectedd")
        
    return clients, train_data, test_data, group

def read_user_data(index,data):
    id = data[0][index]
    train_data = data[1][id]
    test_data = data[2][id]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    X_train = torch.Tensor(X_train).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data

