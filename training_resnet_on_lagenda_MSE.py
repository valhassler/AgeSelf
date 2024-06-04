import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.chdir("/usr/users/vhassle/datasets/lagenda")

class AgeDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
            self.data = data
            self.root_dir = root_dir
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path)
        age = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, age
    
# Training and validation loops
def train_model(model, train_loader, val_loader, criterion, optimizer,base_path_model_save, num_epochs=30):
    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_mse = 0.0
            for i, (inputs, labels) in tqdm(enumerate(loader)):
                print(f"{i}\{len(loader)}", end='\r')

                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #on both here exp if you want to change it
                    loss = criterion(outputs, labels)
                    mse = nn.functional.mse_loss(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_mse += mse.item() * inputs.size(0)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_mse = running_mse / len(loader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} MSE: {epoch_mse:.4f}')

            # Log to wandb
            wandb.log({f"{phase} Loss": epoch_loss, f"{phase} MSE": epoch_mse, "epoch": epoch})

            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
        if epoch > 0 and epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(base_path_model_save,f'age_classification_model_{epoch}_Cross.pth'))
        # Each epoch has a training and validation phase

    print(f'Best val loss: {best_loss:.4f}')

    # Load best model weights
    #model.load_state_dict(best_model_wts) #just keep thenormal model
    return model


# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResizeToMaxDim:
    def __init__(self, max_size):
        self.max_size = max_size
    
    def __call__(self, img):
        # Get current size
        w, h = img.size
        if w > h:
            new_w = self.max_size
            new_h = int(h * (self.max_size / w))
        else:
            new_h = self.max_size
            new_w = int(w * (self.max_size / h))
        return img.resize((new_w, new_h), Image.LANCZOS)
    
class PadToSquare:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill
    
    def __call__(self, img):
        w, h = img.size
        pad_w = (self.size - w) // 2
        pad_h = (self.size - h) // 2
        padding = (pad_w, pad_h, self.size - w - pad_w, self.size - h - pad_h)
        return transforms.functional.pad(img, padding, fill=self.fill)

transform_val = transforms.Compose([
            ResizeToMaxDim(448),
            PadToSquare(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
# Load dataset

data = pd.read_csv(os.path.join("/usr/users/vhassle/datasets/lagenda", "cropped_data.csv"))

#split of the data
sample_percent = 1.00
sampled_data = data.sample(frac=sample_percent)

train_data, val_data = train_test_split(sampled_data, test_size=0.2)

train_dataset = AgeDataset(data = train_data, root_dir="/usr/users/vhassle/datasets/lagenda", transform=transform_train)
val_dataset = AgeDataset(data = val_data, root_dir="/usr/users/vhassle/datasets/lagenda", transform=transform_val)

batch_size = 48
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Modify the final layer for regression
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

wandb.init(project='Age_estimation')
config = wandb.config
config.learning_rate = 0.0001
config.exp_train = False
config.batch_size = batch_size
#config.transformations = ['RandomResizedCrop', 'RandomHorizontalFlip']

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

base_path_model_save = "/usr/users/vhassle/psych_track/AgeSelf/models"

# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer,base_path_model_save, num_epochs=30)

# Save the model
torch.save(model.state_dict(), 'age_classification_model_final.pth')

wandb.finish()
