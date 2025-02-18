# training_resnet.py
# This was done on phobos server so far can be easily adapted to run on emmy server may be push over the datsets that sounds important if it is required

import os
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torchvision import transforms

from ageself.training_resnet_functions import (
    AgeGenderDataset, AgeGenderResNet, FocalLoss, train_model, get_val_transform, get_train_transform,load_age_gender_resnet
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir("/usr/users/vhassle/datasets")#/lagenda"

# Data augmentation
image_size = 150
max_rotation = 90
batch_size = 48
sample_percent = 1

transform_train = get_train_transform(image_size=image_size, max_rotation=max_rotation)
transform_val = get_val_transform(image_size)

# data = pd.read_csv(os.path.join("/usr/users/vhassle/datasets/lagenda/cropped_data_age_classes_faces.csv"))
# Load data1
data_path1 = "/usr/users/vhassle/datasets/Wortschatzinsel/annotation_faces/final_annotations.csv"
data1 = pd.read_csv(os.path.join(data_path1))

# Load data2
data_path2 = "/usr/users/vhassle/datasets/lagenda/cropped_data_age_classes_faces.csv"
data2 = pd.read_csv(os.path.join(data_path2))

# Split data1 into training and validation (80% train, 20% validation)
train_data1, val_data1 = train_test_split(data1, test_size=0.2, random_state=22)
train_data1 = train_data1.reset_index(drop=True)
val_data1 = val_data1.reset_index(drop=True)

# Upsample data1 to match the scale of data2 (50 times bigger)
# Calculate how much we need to upsample data1
upsample_factor = (len(data2) // len(train_data1))

# Upsample data1 by repeating samples with replacement
upsampled_train_data1 = resample(train_data1, 
                                 replace=True,     # sample with replacement
                                 n_samples=upsample_factor * len(train_data1), 
                                 random_state=22)

combined_train_data = pd.concat([upsampled_train_data1, data2], ignore_index=True)
combined_train_data = combined_train_data.reset_index(drop=True)

# Create datasets for training and validation
train_dataset = AgeGenderDataset(data=combined_train_data, root_dir="/usr/users/vhassle/datasets", transform=transform_train)
val_dataset = AgeGenderDataset(data=val_data1, root_dir="/usr/users/vhassle/datasets", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


pretrained_model_path = "/usr/users/vhassle/psych_track/AgeSelf/models/faces_a_g_img_size_150_rot_90/age_gender_classification_model_final.pth"
#model = AgeGenderResNet()
model = load_age_gender_resnet(pretrained_model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

alpha = torch.tensor([1, 0.3, 0.2])
gamma = 2
age_criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
#age_criterion = nn.CrossEntropyLoss()
gender_criterion = nn.CrossEntropyLoss()

run_name = f"faces_a_g_img_size_{image_size}_rot_{max_rotation}_Wortschatz_combined"
base_path_model_save = f"/usr/users/vhassle/psych_track/AgeSelf/models/{run_name}"
wandb.init(project='Age_gender_estimation_focal_faces', name=run_name)

config = wandb.config
config.alpha = alpha
config.gamma = gamma
config.learning_rate = 0.000025
config.sample_percent = sample_percent
config.batch_size = batch_size
config.image_size_input = image_size
config.max_rotation = max_rotation
config.age_gender_loss = [1,1/50]
config.pretrained_model_path = pretrained_model_path
config.data_path1 = data_path1
config.data_path2 = data_path2
config.num_epochs = 3

optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

config.base_path_model_save = base_path_model_save

os.makedirs(base_path_model_save, exist_ok=True)
model = train_model(model, train_loader, val_loader, age_criterion, gender_criterion, optimizer, base_path_model_save, num_epochs=config.num_epochs, age_gender_loss=config.age_gender_loss)

torch.save(model.state_dict(), os.path.join(base_path_model_save, 'age_gender_classification_model_final.pth'))
wandb.finish()