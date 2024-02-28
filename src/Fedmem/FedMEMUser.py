import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,  TensorDataset
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import copy
import pandas as pd
import numpy as np
from src.Optimizer.Optimizer import Fedmem
from src.utils.data_process import FeatureDataset
from sklearn.model_selection import train_test_split


class Fedmem_user():

    def __init__(self,device, model, args, id, exp_no, current_directory):

        self.device = device
        
        self.id = id  # integer
        self.batch_size = args.batch_size
        self.exp_no = exp_no
        self.current_directory = current_directory
        """
        Hyperparameters
        """
        self.learning_rate = args.alpha
        self.local_iters = args.local_iters
        self.eta = args.eta
        self.global_model_name = args.model_name
        self.algorithm = args.algorithm
        """
        DataLoaders
        """

        # Load dataset
        features_folder = '/proj/sourasb-220503/FedMEM/dataset/r3_mem_ResNet50_features'
        if args.target == 10:
            annotations_file = '/proj/sourasb-220503/FedMEM/dataset/clients/'+ 'Client_ID_' + str(self.id) +'.csv'
        else:
            annotations_file = '/proj/sourasb-220503/FedMEM/dataset/clients/A1/'+ 'Client_ID_' + str(self.id) +'.csv'
        print(annotations_file)
        

        dataset = FeatureDataset(features_folder, annotations_file)

        # Split dataset into training and validation
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.25, random_state=self.exp_no)
        self.train_samples = len(train_dataset)
        self.test_samples = len(val_dataset)

        # DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=124, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        
        self.trainloaderfull = DataLoader(train_dataset, self.train_samples)
        self.testloaderfull = DataLoader(val_dataset, self.test_samples)
        
        self.iter_trainloader = iter(self.train_loader)
        self.iter_testloader = iter(self.val_loader)


        # those parameters are for persionalized federated learing.
        
        self.local_model = copy.deepcopy(model)
        self.eval_model = copy.deepcopy(model)

        self.local_model.to(self.device)
        self.eval_model.to(self.device)
        # print(self.local_model)
        # print(self.eval_model)

        # Freeze all layers
        for param in self.local_model.parameters():
            param.requires_grad = False

        # Unfreeze the second last layer
        for param in self.local_model.fc1.parameters():
            param.requires_grad = True

        # Unfreeze the last layer
        for param in self.local_model.fc2.parameters():
            param.requires_grad = True

        """
        Loss
        """

        self.loss = nn.CrossEntropyLoss()
        self.minimum_loss = 0.0

        """
        Optimizer
        """
        # self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=bb_step)
        # self.optimizer = torch.optim.SGD([{'params': self.local_model.fc1.parameters()},
        #                                    {'params': self.local_model.fc2.parameters()}
        #                                ], lr=self.learning_rate, weight_decay=0.001)  # Only optimize the last layer

        self.optimizer = Fedmem(self.local_model.parameters(), self.learning_rate, self.eta)


    #def bb_step(optimizer, grad, step_size):

    def set_parameters(self, cluster_model):
        for param, glob_param in zip(self.local_model.parameters(), cluster_model):
            param.data = glob_param.data.clone()
            # print(f"user {self.id} parameters : {param.data}")
        # input("press")
            
    def get_parameters(self):
        return self.local_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.eval_model.parameters(), new_params):
            param.data = new_param.data.clone()


    def test(self, global_model):
        # Set the model to evaluation mode
        self.eval_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.testloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.eval_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.testloaderfull)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        cm = confusion_matrix(y_true,y_pred)         
        return accuracy, validation_loss, precision, recall, f1, cm

    def test_local(self):
        self.local_model.eval()
        loss = 0
    
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.testloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.testloaderfull)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        cm = confusion_matrix(y_true,y_pred)
                
        return accuracy, validation_loss, precision, recall, f1, cm


    def train_error_and_loss(self, global_model):
      
        # Set the model to evaluation mode
        self.eval_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.trainloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.eval_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.testloaderfull)
        # precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
                
        return accuracy, train_loss

    def train_error_and_loss_local(self):
        self.local_model.eval()
        loss = 0
    
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.trainloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.testloaderfull)
        # precision = precision_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
                
        return accuracy, train_loss

    
    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.local_model = torch.load(os.path.join(model_path, "server" + ".pt"))

    
#    @staticmethod
#    def model_exists():
#        return os.path.exists(os.path.join("models", "server" + ".pt"))
    def evaluate_model(self, epoch, t):
        self.local_model.eval()  # Set the model to evaluation mode
        y_true = []
        y_pred = []
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.testloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        # accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.testloaderfull)
        cm = confusion_matrix(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        
        if epoch == 0 and self.minimum_loss == 0.0:
            self.minimum_per_loss = validation_loss
        else:
            if validation_loss < self.minimum_per_loss:
                self.minimum_per_loss = validation_loss
                # print(f"new minimum loss of local model at client {self.id} found at global round {t} local epoch {epoch}")
                self.save_local_model(epoch, cm, t)
                
                
        
    
    def save_local_model(self, iter, cm, t):
        cm_df = pd.DataFrame(cm)
        
        # file = "per_model_user"+ str(self.id) +"_exp_no_" + str(self.exp_no) + "_LR_" + str(iter) + "_GR_" + str(t) 
        file = "per_model_user_" + str(self.id)
        file_cm = "cm_user_" + str(self.id)
        #print(file)
       
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" +"local_models"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/models/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/"+ directory_name)
        if not os.path.exists(self.current_directory + "/models/confusion_matrix/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/confusion_matrix/"+ directory_name)
        
        torch.save(self.local_model,self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")
        cm_df.to_csv(self.current_directory + "/models/confusion_matrix/"+ directory_name + "/" + file_cm + ".csv", index=False)

        # print(f"local model saved at global round :{t} local round :{iter}")

    def train(self, cluster_model, t):
        
        for epoch in range(self.local_iters):  # Loop over the dataset multiple times
            self.local_model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
        
                # Zero the parameter gradients
                self.optimizer.zero_grad()
        
                # Forward + backward + optimize
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                
                proximal_term = 0.0
                for param, c_param in zip(self.local_model.parameters(), cluster_model):
                    proximal_term += 0.5 * torch.norm(param - c_param) ** 2
                loss += proximal_term
                loss.backward()
                self.optimizer.step(cluster_model)
                #self.optimizer.step()
                running_loss += loss.item()
            # print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(self.train_loader)}")
            self.evaluate_model(epoch, t)
            
    