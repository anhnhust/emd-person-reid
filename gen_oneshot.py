import os
import random
import shutil


source_folder = "/home/anhhoang/New_Term/AlignedReID/datamica/20191104_all/query/"


destination_folder ="/home/anhhoang/New_Term/AlignedReID/datamica/20191104_all_one/query/"



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


image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]


image_dict = {}

for image_file in image_files:
    parts = image_file.split('_')
    x = int(parts[0].replace("c", ""))
    if x not in image_dict:
        image_dict[x] = []
    image_dict[x].append(image_file)


for x, images_for_x in image_dict.items():
    random_image = random.choice(images_for_x)
    source_path = os.path.join(source_folder, random_image)
    destination_path = os.path.join(destination_folder, random_image)
    shutil.copy(source_path, destination_path)

print("Succes!")


source_folder = "/home/anhhoang/New_Term/AlignedReID/datamica/20191104_all/gallery/"


destination_folder ="/home/anhhoang/New_Term/AlignedReID/datamica/20191104_all_one/gallery/"



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


image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]


image_dict = {}

for image_file in image_files:
    parts = image_file.split('_')
    x = int(parts[0].replace("c", ""))
    if x not in image_dict:
        image_dict[x] = []
    image_dict[x].append(image_file)


for x, images_for_x in image_dict.items():
    random_image = random.choice(images_for_x)
    source_path = os.path.join(source_folder, random_image)
    destination_path = os.path.join(destination_folder, random_image)
    shutil.copy(source_path, destination_path)

print("Succes!")
