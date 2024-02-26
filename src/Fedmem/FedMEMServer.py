import torch
import os
import h5py
from src.Fedmem.FedMEMUser import Fedmem_user
import numpy as np
import copy
from datetime import date
from tqdm import trange
from tqdm import tqdm
import numpy as np
# from sklearn.cluster import SpectralClustering
import time
from sklearn.cluster import KMeans
# Implementation for FedAvg Server
import matplotlib.pyplot as plt

class Fedmem():
    def __init__(self,device, model, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.user_ids = args.user_ids
        print(f"user ids : {self.user_ids}")
        self.total_users = len(self.user_ids)
        print(f"total users : {self.total_users}")
        self.num_users = self.total_users * args.users_frac    #selected users
        self.num_teams = args.num_teams
        self.group_division = args.group_division
        self.total_train_samples = 0
        self.exp_no = exp_no
        self.n_clusters = args.num_teams
        self.gamma = args.gamma # scale parameter for RBF kernel 
        self.lamda = args.lamda # similarity tradeoff
        self.current_directory = current_directory
        self.algorithm = args.algorithm
        self.clusterhead_models = []
        self.cluster_dict = {}

        self.c = [[] for _ in range(args.num_teams)]


        """
        Global model
        
        """

        self.global_model = copy.deepcopy(model)
        # print(self.global_model)
        self.global_model.to(self.device)
        self.global_model_name = args.model_name

        """
        Clusterhead models
        """

        for n_clust in range(self.n_clusters):
            self.clusterhead_models.append(self.global_model)

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


        
        # data = read_data(args, current_directory)
        # self.tot_users = len(data[0])
        # print(self.tot_users)

        for i in trange(self.total_users, desc="Data distribution to clients"):
            # id, train, test = read_user_data(i, data)
            user = Fedmem_user(device, self.global_model, args, self.user_ids[i], exp_no, current_directory)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Finished creating Fedmem server.")

        
    def send_global_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model.parameters())
    
    def send_cluster_parameters(self):
        for clust_id in range(self.num_teams):
            users = np.array(self.cluster_dict[clust_id])
            print(f"cluster {clust_id} model has been sent to {len(users)} users")
            if len(users) != 0:
                #for param in self.c[clust_id]:
                    # print(f" cluster {clust_id} parameters :{param.data}")
                    # input("press")
                for user in users:
                    user.set_parameters(self.c[clust_id])

    def add_parameters(self, cluster_model, ratio):
        for server_param, cluster_param in zip(self.global_model.parameters(), cluster_model.parameters()):
            server_param.data += cluster_param.data.clone() * ratio

    def global_update(self):
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        
        for cluster_model in self.clusterhead_models:
            self.add_parameters(cluster_model, 1/len(self.clusterhead_models))

    def add_parameters_clusters(self, user, ratio, cluster_id):
        
        for cluster_param, user_param in zip(self.clusterhead_models[cluster_id].parameters(), user.get_parameters()):
            cluster_param.data += ratio*user_param.data.clone()
            # print(f"cluster {cluster_id} model after adding user {user.id}'s local model : {cluster_param.data} ")
        self.c[cluster_id] = copy.deepcopy(list(self.clusterhead_models[cluster_id].parameters()))
    def aggregate_clusterhead(self):

        for clust_id in range(self.num_teams):
            for param in self.clusterhead_models[clust_id].parameters():
                param.data = torch.zeros_like(param.data)
        
            users = np.array(self.cluster_dict[clust_id])
            # print(users)
            # input("press")
            print(f"number of users are {len(users)} in cluster {clust_id} ")
            if len(users) != 0:
                for user in users:
                    self.add_parameters_clusters(user, 1/len(users), clust_id)


    def select_users(self, round, subset_users):
        np.random.seed(round)
        return np.random.choice(self.users, subset_users, replace=False)

        
    def pearson_correlation(self, tensor1, tensor2):
        mean1, mean2 = torch.mean(tensor1), torch.mean(tensor2)
        numerator = torch.sum((tensor1 - mean1) * (tensor2 - mean2))
        denominator = torch.sqrt(torch.sum((tensor1 - mean1) ** 2)) * torch.sqrt(torch.sum((tensor2 - mean2) ** 2))
        return numerator / denominator


    def flatten_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.view(-1))
        return torch.cat(params)
    
    def find_similarity(self, similarity_metric, params1, params2, params_g):
                            
        if similarity_metric == "cosign similarity":
            similarity = torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))
        elif similarity_metric == "euclidian":
            similarity_u =  torch.exp(-self.gamma * torch.sqrt(torch.sum((params1 - params2) ** 2)))
            similarity_g =  torch.exp(-self.gamma * torch.sqrt(torch.sum((params1 + params2 - 2* params_g) ** 2)))
            
            similarity = (1-self.lamda)*similarity_u + self.lamda*similarity_g
            # print(f"similarity_u : {similarity_u}, similarity_g : {similarity_g}, similarity : {similarity}")
            # print("RBF: ",similarity.item())

            #simi = torch.sqrt(torch.sum((params1 - params2) ** 2))

            
        elif similarity_metric == "manhattan":
            similarity = torch.sum(torch.abs(params1 - params2))
        elif similarity_metric == "pearson_correlation":
            similarity = self.pearson_correlation(params1, params2)

        return similarity


    def similarity_check(self):
        similarity_matrix = {}
        # similarity_metric = "manhattan"
        similarity_metric = "euclidian"
        #print("computing cosign similarity")
        params_g = self.flatten_params(self.global_model)
        for user in tqdm(self.selected_users, desc="participating clients"):
            if user.id not in similarity_matrix:
                similarity_matrix[user.id] = []
            #print(similarity_matrix)
            params1 = self.flatten_params(user.local_model)

            
            for comp_user in self.selected_users:
                # if user != comp_user:
                    
                params2 = self.flatten_params(comp_user.local_model)

                similarity = self.find_similarity(similarity_metric, params1, params2, params_g)


               #print("user_id["+ str(user.id)+"] and user_id["+str(comp_user.id)+"] = ",similarity.item())
                    

                similarity_matrix[user.id].extend([similarity.item()])
                
        
        return similarity_matrix

    def eigen_decomposition(self, laplacian_matrix, n_components):
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
        # Sort eigenvectors by eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        return eigenvectors[:, :n_components]
    
    def compute_laplacian(self, similarity_matrix):
        degree_matrix = np.diag(similarity_matrix.sum(axis=1))
        
        return degree_matrix - similarity_matrix



    def spectral(self, similarity_dict, n_clusters):

        size = len(similarity_dict)
        # print(size)
        matrix = np.zeros((size, size))
        # print(matrix)
        i = 0
        for key, values in similarity_dict.items():
           # print(key)
           # print(values)
            matrix[i] = values
            i+=1
        laplacian_matrix = self.compute_laplacian(matrix)
        # input("laplacian")
        # print(laplacian_matrix)

        eigenvectors = self.eigen_decomposition(laplacian_matrix, n_clusters)
        # input("eigenvectors")
        # print(eigenvectors)

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(eigenvectors)
        return kmeans.labels_
  
                    
    def combine_cluster_user(self,clusters):
        self.cluster_dict = {}
        self.cluster_dict_user_id = {}
        user_ids = []
        for user in self.selected_users:
            user_ids.append(user.id)
        for key, value, UID in zip(clusters, self.selected_users, user_ids):
            if key not in self.cluster_dict:
                self.cluster_dict[key] = []
                self.cluster_dict_user_id[key] = []
            self.cluster_dict[key].append(value)
            self.cluster_dict_user_id[key].append(UID)

        print(self.cluster_dict)
        print(self.cluster_dict_user_id)

    def train(self):
        loss = []
        
        for t in trange(self.num_glob_iters, desc="Global Rounds :"):
            if t == 0:
                self.send_global_parameters()
            else:
                self.send_cluster_parameters()
            
            self.selected_users = self.select_users(t, int(self.num_users)).tolist()
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)

            if t == 0:
                for user in tqdm(self.selected_users, desc="running selected clients"):
                    user.train(self.global_model, t)  # * user.train_samples
            else:
                for clust_id in range(self.num_teams):
                    users = np.array(self.cluster_dict[clust_id])
                    for user in tqdm(users, desc=f"running cluster {clust_id}"):
                        
                        user.train(self.clusterhead_models[clust_id], t)


            similarity_matrix = self.similarity_check()
            # print(f"similarity_matrix : {similarity_matrix}")
            
            #clustering

            # cluters = kmeans(similarity_matrix)
            # spectral cluster

            clusters = self.spectral(similarity_matrix, self.n_clusters).tolist()

            print(clusters)
            self.combine_cluster_user(clusters)

            self.aggregate_clusterhead()
            self.global_update()

        
            self.evaluate_localmodel()
            # input("press")
            self.evaluate_clusterhead()
            # self.evaluate()

            
            # self.save_cluster_model(t)
            # self.save_global_model(t)
        self.save_results()
        self.plot_result()

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

            hf.create_dataset('cluster_test_loss', data=self.cluster_test_loss)
            hf.create_dataset('cluster_train_loss', data=self.cluster_train_loss)
            hf.create_dataset('cluster_test_accuracy', data=self.cluster_test_acc)
            hf.create_dataset('cluster_train_accuracy', data=self.cluster_train_acc)

            hf.create_dataset('per_test_loss', data=self.local_test_loss)
            hf.create_dataset('per_train_loss', data=self.local_train_loss)
            hf.create_dataset('per_test_accuracy', data=self.local_test_acc)
            hf.create_dataset('per_train_accuracy', data=self.local_train_acc)
            hf.close()
        
    def save_global_model(self, t):
        file = "_exp_no_" + str(self.exp_no) + "_GR_" + str(t) 
        
        print(file)
       
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" +"global_model"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/models/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/"+ directory_name)
        
        torch.save(self.global_model,self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")

    def save_cluster_model(self, t):
        

        for cluster in range(self.num_teams):
            file = "cluster_model_" + str(cluster) + "_exp_no_" + str(self.exp_no) + "_GR_" + str(t)
        
            print(file)
            directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" +"cluster_model_" + str(cluster) 
            # Check if the directory already exists
            if not os.path.exists(self.current_directory + "/models/"+ directory_name):
            # If the directory does not exist, create it
                os.makedirs(self.current_directory + "/models/"+ directory_name)
        
            torch.save(self.clusterhead_models[cluster],self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")


    def test_error_and_loss(self, evaluate_model):
        num_samples = []
        tot_correct = []
        losses = []
        if evaluate_model == 'global':
            for c in self.selected_users:
                ct, ls,  ns = c.test(self.global_model)
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
                    ct, ls,  ns = c.test(self.clusterhead_models[clust_id])
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
                ct, ls,  ns = c.train_error_and_loss(self.global_model)
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
                    ct, ls,  ns = c.train_error_and_loss(self.clusterhead_models[clust_id])
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

    def evaluate_clusterhead(self):
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

        print(f"Cluster Trainning Accurancy: {train_acc}" )
        print(f"Cluster Trainning Loss: {train_loss}")
        print(f"Cluster test accurancy: {test_acc}" )
        print(f"Cluster test_loss: {test_loss}")

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

        print(f"Local Trainning Accurancy: {train_acc}")
        print(f"Local Trainning Loss: {train_loss}")
        print(f"Local test accurancy: {test_acc}")
        print(f"Local test_loss: {test_loss}")
    
    def plot_result(self):
        
        print(self.global_train_acc)

        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0].plot(self.local_train_acc, label= "Train_accuracy")
        ax[0].plot(self.local_test_acc, label= "Test_accuracy")
        ax[0].set_xlabel("Global Iteration")
        ax[0].set_ylabel("accuracy")
        ax[0].set_xticks(range(0, self.num_glob_iters, int(self.num_glob_iters/5)))#
        ax[0].legend(prop={"size":12})
        ax[1].plot(self.local_train_loss, label= "Train_loss")
        ax[1].plot(self.local_test_loss, label= "Test_loss")
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
    