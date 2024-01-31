import torch
import os
import h5py
from src.FeSEM.FeSEM_user import User
from src.utils.data_process import read_data, read_user_data
import numpy as np
import copy
from datetime import date
from tqdm import trange
from tqdm import tqdm
import numpy as np
# from sklearn.cluster import SpectralClustering
import time
from sklearn.cluster import KMeans
from src.TrainModels.trainmodels import *
import torch.nn.init as init
import matplotlib.pyplot as plt
# Implementation for FeSEM Server
"""
Class FeSEM() is the server class.

"""

class FeSEM():
    def __init__(self,device, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.num_users = args.numusers   #selected users
        self.num_teams = args.num_teams
        self.group_division = args.group_division
        self.total_train_samples = 0
        self.exp_no = exp_no
        self.n_clusters = args.num_teams
        self.gamma = args.gamma # scale parameter for RBF kernel 
        self.current_directory = current_directory
        self.clusterhead_models = []
        self.cluster_dict = {}
        """
        Global model
        
        """

        
        self.global_model_name = args.model_name
  
        """
        Clusterhead models
        """

        for n_clust in range(self.n_clusters):
            model = ResNet50FT().to(device)
            self.clusterhead_models.append(model)
        self.users = []
        self.selected_users = []
        self.global_train_acc = []
        self.global_train_loss = [] 
        self.global_test_acc = [] 
        self.global_test_loss = []


        """
        Cluster head evaluation
        """

        self.cluster_train_acc = []
        self.cluster_test_acc = []
        self.cluster_train_loss = []
        self.cluster_test_loss = []

        """
        Local model evaluation
        """

        self.local_train_acc = []
        self.local_test_acc  = []
        self.local_train_loss  = []
        self.local_test_loss  = []


        data = read_data(args, current_directory)
        self.tot_users = len(data[0])
        print(self.tot_users)

        for i in trange(self.tot_users, desc="Data distribution to clients"):
            id, train, test = read_user_data(i, data)
            user = User(device, train, test, args, i)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Finished creating FedAvg server.")

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        
    def send_global_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model)
    
    def send_cluster_parameters(self):
        for clust_id in range(self.num_teams):
            users = np.array(self.cluster_dict[clust_id])
            for user in users:
                user.set_parameters(self.clusterhead_models[clust_id])

    def add_parameters(self, cluster_model, ratio):
        for server_param, cluster_param in zip(self.global_model.parameters(), cluster_model.parameters()):
            server_param.data = server_param.data + cluster_param.data.clone() * ratio


    def add_parameters_clusters(self, user, ratio, cluster_id):

        # model = self.clusterhead_models.parameters()

        for cluster_param, user_param in zip(self.clusterhead_models[cluster_id].parameters(), user.get_parameters()):
            cluster_param.data = cluster_param.data + user_param.data.clone() * ratio


    def aggregate_clusterhead(self, r_ik_dict):

        for clust_id in range(self.num_teams):
            users = np.array(self.cluster_dict[clust_id])
            for user in users:
                self.add_parameters_clusters(user, 1/len(users), clust_id)
            

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.global_model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, subset_users):
        np.random.seed(round)
        return np.random.choice(self.users, subset_users, replace=False)

    def flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.view(-1))
        return torch.cat(params)
    
    def find_similarity(self, params1, params2, ratio):
        similarity =  ratio * torch.norm(params1 - params2, p=2)                    

        print(similarity.item())
        return similarity


    def similarity_check(self):
        similarity_matrix = {}
        for k in trange(self.n_clusters):
            if k not in similarity_matrix:
                similarity_matrix[k] = []
            #print(similarity_matrix)
            params1 = self.flatten_params(self.clusterhead_models[k])
            for user in self.selected_users:
                # if user != comp_user:
                    
                params2 = self.flatten_params(user.local_model)

                print("user sample",user.train_samples)
                print("total sample",self.total_train_samples)
                ratio = float(user.train_samples/self.total_train_samples)
                print(ratio)
                similarity = self.find_similarity(params1, params2, ratio)


               #print("user_id["+ str(user.id)+"] and user_id["+str(comp_user.id)+"] = ",similarity.item())
                    

                similarity_matrix[k].extend([similarity.item()])
                
        
        return similarity_matrix



    def find_rik_dict(self, similarity_dict):


        # Step 1: Find the minimum in each column
        min_in_clusters = [min(similarity_dict[row][i] for row in similarity_dict) for i in range(len(self.selected_users))]
        print(min_in_clusters)
        # Step 2: Create new dictionary with 1s and 0s
        r_ik_dict = {}
        for row in similarity_dict:
            r_ik_dict[row] = [1 if similarity_dict[row][i] == min_in_clusters[i] else 0 for i in range(len(similarity_dict[row]))]

        return r_ik_dict

    def create_cluster(self, r_ik_dict):
        
        for row_key, row_values in r_ik_dict.items():
            self.cluster_dict[row_key] = [self.selected_users[i] for i, value in enumerate(row_values) if value == 1]

        # print(self.cluster_dict)
        


        
    def train(self):
        loss = []
        for n_clust in range(self.n_clusters):
            self.clusterhead_models[n_clust].apply(self.initialize_weights)
        
        
        for t in trange(self.num_glob_iters, desc="Global Rounds"):
            # params1 = self.flatten_params(self.clusterhead_models[0])
            # params2 = self.flatten_params(self.clusterhead_models[1])
            # difference =  torch.norm(params1 - params2, p=2)
            # print(difference.item())
            # input("press")
            self.selected_users = self.select_users(t, 10).tolist()
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)

            similarity_dict = self.similarity_check()
            print(similarity_dict)
            r_ik_dict = self.find_rik_dict(similarity_dict)
            print(r_ik_dict)
            self.create_cluster(r_ik_dict)
            print(self.cluster_dict)
            input("pause")
            self.aggregate_clusterhead(r_ik_dict)
            
            for clust_id in range(self.num_teams):
                    users = np.array(self.cluster_dict[clust_id])
                    for user in users:
                        user.train()

            
            
        
           #  self.evaluate_localmodel()
           #  self.evaluate_clusterhead()
           # self.evaluate()

        #self.save_results()
        #self.save_model()


    
    def test_error_and_loss(self, evaluate_model):
        num_samples = []
        tot_correct = []
        losses = []
        if evaluate_model == 'global':
            for c in self.selected_users:
                ct, ls,  ns = c.test(self.global_model.parameters())
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(ls)
        elif evaluate_model == 'local':
            for c in self.selected_users:
                ct, ls,  ns = c.test_local()
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(ls)
        else:
            for clust_id in range(self.num_teams):
                users = np.array(self.cluster_dict[clust_id])
                for c in users:
                    ct, ls,  ns = c.test(self.clusterhead_models[clust_id].parameters())
                    tot_correct.append(ct * 1.0)
                    num_samples.append(ns)
                    losses.append(ls)
                    

            
        return num_samples, tot_correct, losses

    def train_error_and_loss(self, evaluate_model):
        num_samples = []
        tot_correct = []
        losses = []
        
        if evaluate_model == 'global':
            for c in self.selected_users:
                ct, ls,  ns = c.train_error_and_loss(self.global_model.parameters())
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(ls)
        elif evaluate_model == 'local':
            for c in self.selected_users:
                ct, ls,  ns = c.train_error_and_loss_local()
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(ls)
        else:
            for clust_id in range(self.num_teams):
                users = np.array(self.cluster_dict[clust_id])
                for c in users:
                    ct, ls,  ns = c.train_error_and_loss(self.clusterhead_models[clust_id].parameters())
                    tot_correct.append(ct * 1.0)
                    num_samples.append(ns)
                    losses.append(ls)

        
        return num_samples, tot_correct, losses


    def evaluate(self):
        evaluate_model = "global"
        stats_test = self.test_error_and_loss(evaluate_model)
        stats_train = self.train_error_and_loss( evaluate_model)
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])

        self.global_train_acc.append(train_acc)
        self.global_test_acc.append(test_acc)
        self.global_train_loss.append(train_loss)
        self.global_test_loss.append(test_loss)

        print("Global Trainning Accurancy: ", train_acc)
        print("Global Trainning Loss: ", train_loss)
        print("Global test accurancy: ", test_acc)
        print("Global test_loss:",test_loss)

    def evaluate_clusterhead(self, combine_dict):
        evaluate_model = "cluster"
        stats_test = self.test_error_and_loss(evaluate_model)
        stats_train = self.train_error_and_loss(evaluate_model)
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])

        self.cluster_train_acc.append(train_acc)
        self.cluster_test_acc.append(test_acc)
        self.cluster_train_loss.append(train_loss)
        self.cluster_test_loss.append(test_loss)

        print("Cluster Trainning Accurancy: ", train_acc)
        print("Cluster Trainning Loss: ", train_loss)
        print("Cluster test accurancy: ", test_acc)
        print("Cluster test_loss:",test_loss)

    def evaluate_localmodel(self):
        evaluate_model = "local"
        stats_test = self.test_error_and_loss(evaluate_model)
        stats_train = self.train_error_and_loss(evaluate_model)
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])

        self.local_train_acc.append(train_acc)
        self.local_test_acc.append(test_acc)
        self.local_train_loss.append(train_loss)
        self.local_test_loss.append(test_loss)

        print("Local Trainning Accurancy: ", train_acc)
        print("Local Trainning Loss: ", train_loss)
        print("Local test accurancy: ", test_acc)
        print("Local test_loss:",test_loss)
    

    def plot_result(self):
        
        print(self.global_train_acc)

        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0].plot(self.global_train_acc, label= "Train_accuracy")
        ax[0].plot(self.global_test_acc, label= "Test_accuracy")
        ax[0].set_xlabel("Global Iteration")
        ax[0].set_ylabel("accuracy")
        ax[0].set_xticks(range(0, self.num_glob_iters, int(self.num_glob_iters/5)))#
        ax[0].legend(prop={"size":12})
        ax[1].plot(self.global_train_loss, label= "Train_loss")
        ax[1].plot(self.global_test_loss, label= "Test_loss")
        ax[1].set_xlabel("Global Iteration")
        #ax[1].set_xscale('log')
        ax[1].set_ylabel("Loss")
        #ax[1].set_yscale('log')
        ax[1].set_xticks(range(0, self.num_glob_iters, int(self.num_glob_iters/5)))
        ax[1].legend(prop={"size":12})
        
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" +"plot"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)

        

        plt.draw()
       
        plt.savefig(self.current_directory + "/results/" + directory_name  + "/_global_iters_" + str(self.num_glob_iters) + '.png')

        # Show the graph
        plt.show()
    

    # Save loss, accurancy to h5 fiel
    def save_results(self):
        file = "_exp_no_" + str(self.exp_no) + "_GR_" + str(self.num_glob_iters) + "_BS_" + str(self.batch_size)
        
        print(file)
       
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" +"h5"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)



        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            hf.create_dataset('Global rounds', data=self.num_glob_iters)
            hf.create_dataset('Local iters', data=self.local_iters)
            hf.create_dataset('Learning rate', data=self.learning_rate)
            hf.create_dataset('Batch size', data=self.batch_size)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('per_test_loss', data=self.local_test_loss)
            hf.create_dataset('per_train_loss', data=self.local_train_loss)
            hf.create_dataset('per_test_accuracy', data=self.local_test_acc)
            hf.create_dataset('per_train_accuracy', data=self.local_train_acc)
            hf.close()