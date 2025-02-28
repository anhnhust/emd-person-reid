import os
import shutil
import random


folder_path = "/home/micalab/AlignedReID/datamica/20191105_left/gallery"
destination_folder ="/home/micalab/AlignedReID/datamica/20191105_left_one/gallery"



if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
else:
    for filename in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Cant delete {file_path}: {e}")

image_dict = {}

for filename in os.listdir(folder_path):
    parts = filename.split('_')
    y_part = parts[1]
    y_value = int(y_part.replace("c", "").replace("s", ""))
    
    if y_value not in image_dict:
        image_dict[y_value] = []
    image_dict[y_value].append(filename)

for y_value, image_list in image_dict.items():
    random_image = random.choice(image_list)
    source_path = os.path.join(folder_path, random_image)
    shutil.copy(source_path,destination_folder)



folder_path = "/home/micalab/AlignedReID/datamica/20191105_left/query/"
destination_folder ="/home/micalab/AlignedReID/datamica/20191105_left_one/query/"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
else:
    for filename in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Cant delete {file_path}: {e}")

image_dict = {}

for filename in os.listdir(folder_path):
    parts = filename.split('_')
    y_part = parts[1]
    y_value = int(y_part.replace("c", "").replace("s", ""))
    
    if y_value not in image_dict:
        image_dict[y_value] = []
    image_dict[y_value].append(filename)

for y_value, image_list in image_dict.items():
    random_image = random.choice(image_list)
    source_path = os.path.join(folder_path, random_image)
    shutil.copy(source_path,destination_folder)

