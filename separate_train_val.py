'''
Utility script for separating original images and their annotations into train and validation sets
'''


import os
import shutil
import tqdm


image_path = '/mnt/data/psemchyshyn/top-detection/mlc_training_data/images'
label_path = '/mnt/data/psemchyshyn/top-detection/mlc_training_data/ground_truth_files'
train_size = 0.8

names = os.listdir(image_path)
names = list(map(lambda x: x.split('.')[0], names))

train_names = names[:int(train_size*len(names))]
val_names = names[int(train_size*len(names)):]


out_train_labels = '/mnt/data/psemchyshyn/top-detection/train/labels'
out_train_images = '/mnt/data/psemchyshyn/top-detection/train/images'
os.makedirs(out_train_labels)
os.makedirs(out_train_images)
for name in tqdm.tqdm(train_names):
    shutil.copy(os.path.join(image_path, f"{name}.png"), out_train_images)
    shutil.copy(os.path.join(label_path, f"{name}.json"), out_train_labels)


out_val_labels = '/mnt/data/psemchyshyn/top-detection/val/labels'
out_val_images = '/mnt/data/psemchyshyn/top-detection/val/images'
os.makedirs(out_val_labels)
os.makedirs(out_val_images)
for name in tqdm.tqdm(val_names):
    shutil.copy(os.path.join(image_path, f"{name}.png"), out_val_images)
    shutil.copy(os.path.join(label_path, f"{name}.json"), out_val_labels)
