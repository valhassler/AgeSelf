# training_resnet.py

import os
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms

from ageself.training_resnet_functions import (
    AgeGenderDataset, AgeGenderResNet, FocalLoss, train_model, get_val_transform, get_train_transform
)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.chdir("/usr/users/vhassle/datasets/lagenda")

# Data augmentation
image_size = 150
max_rotation = 90

transform_train = get_train_transform(image_size=image_size, max_rotation=max_rotation)
transform_val = get_val_transform(image_size)

data = pd.read_csv(os.path.join("/usr/users/vhassle/datasets/lagenda/cropped_data_age_classes_faces.csv"))

sample_percent = 1
sampled_data = data.sample(frac=sample_percent, random_state=22)

train_data, val_data = train_test_split(sampled_data, test_size=0.2, random_state=22)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True) 

train_dataset = AgeGenderDataset(data=train_data, root_dir="/usr/users/vhassle/datasets/lagenda", transform=transform_train)
val_dataset = AgeGenderDataset(data=val_data, root_dir="/usr/users/vhassle/datasets/lagenda", transform=transform_val)

batch_size = 48
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

model = AgeGenderResNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

alpha = torch.tensor([1, 0.3, 0.1])
gamma = 2
age_criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
gender_criterion = nn.CrossEntropyLoss()

wandb.init(project='Age_gender_estimation_focal_faces')
config = wandb.config
config.alpha = alpha
config.gamma = gamma
config.learning_rate = 0.000025
config.sample_percent = sample_percent
config.batch_size = batch_size
config.image_size_input = image_size
config.max_rotation = max_rotation
config.age_gender_loss = [1,1/50]

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

base_path_model_save = f"/usr/users/vhassle/psych_track/AgeSelf/models/faces_a_g_img_size_{image_size}_rot_{max_rotation}"
config.base_path_model_save = base_path_model_save

os.makedirs(base_path_model_save, exist_ok=True)
model = train_model(model, train_loader, val_loader, age_criterion, gender_criterion, optimizer, base_path_model_save, num_epochs=50, age_gender_loss=config.age_gender_loss)

torch.save(model.state_dict(), os.path.join(base_path_model_save, 'age_gender_classification_model_final.pth'))
wandb.finish()