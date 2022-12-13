import os
import glob as gb
import pandas as pd
root_dir = './PBL_data/data_0/71_sensing_data/71_sensing_data/'
folder_name = []
for folder in os.listdir(root_dir):
    folder2 = gb.glob(root_dir+folder)
    for inner_folder in folder2:
        temp = inner_folder[-17:]
        folder_name.append(temp)
    

# print(folder_name)
    