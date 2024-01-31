import os
from src.utils.data_generation import load_data

current_directory = os.getcwd()
print(current_directory)

data_dir = current_directory + "/dataset/r3_refined"

csv_file = current_directory + "/dataset/refined_training_file.csv"

load_data(data_dir, csv_file)
