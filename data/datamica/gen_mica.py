import os
import shutil

parent_folder_path = "/home/anhhoang/New_Term/AlignedReID/datamica/201911051610_outdoor_right_10ids/"
output_folder_path ="/home/anhhoang/New_Term/AlignedReID/datamica/20191105_all/gallery/"


if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]

count = 1

for folder_path in subfolders:
    folder_name = os.path.basename(folder_path)
    p_num, t_num = folder_name.split("_")[0], folder_name.split("_")[1]
    
    files = os.listdir(folder_path)
    
    for file_name in files:
        new_file_name = f"{str(p_num[1:]).zfill(4)}_c{t_num[1:]}s1_{str(count).zfill(6)}_00.jpg"
        shutil.copy(os.path.join(folder_path, file_name), os.path.join(output_folder_path, new_file_name))
        
        count += 1