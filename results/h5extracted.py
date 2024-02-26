import os
import h5py
import numpy as np
attributes = []
hf = h5py.File('/proj/sourasb-220503/FedMEM/results/ResNet50TL/FedAvg/h5/_exp_no_0_GR_100_BS_124.h5','r')
id = 0

for key in hf.keys():
    attributes.append(key)
    print("id [",id,"] :", key)
    id+=1
print(attributes)
tsl = hf.get('global_test_loss')
tsa = hf.get('global_test_accuracy')
trl = hf.get('global_train_loss')
tra = hf.get('global_train_accuracy')
# eta = hf.get('eta_0')
# lamda = hf.get('lambda_0')
# print("eta 0", np.array(eta))
# print("lamda 0", np.array(lamda))
print("train accuracy",np.array(tra))
print("test accuracy",np.array(tsa))
print("train loss",np.array(trl))
print("test loss",np.array(tsl))
#print(np.array(gtra))


# print("maximum test accuracy test global :",max_acc_test_global)
# n1 = hf.get('server_aggregation_test_accuracy')
# n1 = np.array(n1)
#print(len(n1))
#print(n1[:100])