import os
import h5py
import numpy as np
attributes = []
# path = '/proj/sourasb-220503/FedMEM/results/ResNet50TL/Fedmem/data_silo_40/target_3/dynamic/16.0/h5/'
path = '/proj/sourasb-220503/FedMEM/results/fixed_client_16/'
hf = h5py.File(path + '_exp_no_0_GR_5_BS_64_data_silo_100_num_user_8.0'+'.h5','r')
id = 0

for key in hf.keys():
    attributes.append(key)
    print("id [",id,"] :", key)
    id+=1
print(attributes)
key_0 = hf.get('maximum_per_test_accuracy')
key_1 = hf.get('maximum_per_test_accuracy_list')
key_2 = hf.get('std_dev')
key_3 = hf.get('client_16_accuracy_array')
key_4 = hf.get('client_16_f1_array')
key_5 = hf.get('client_16_val_loss_array')
#print("train accuracy",np.array(tra))
print(f"0 : {np.array(key_0)}")
print(f"1 : {np.array(key_1)}")
print(f"2 : {np.array(key_2)}")
print(f"3 : {np.array(key_3)}")
print(f"4 : {np.array(key_4)}")
print(f"5 : {np.array(key_5)}")
