import os
from src.utils.data_generation import load_data

current_directory = os.getcwd()
print(current_directory)

load_data(current_directory)
