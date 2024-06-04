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
        age_class = self.data.iloc[idx, 7]
        if self.transform:
            image = self.transform(image)
        return image, int(age_class)
    
# Training and validation loops
def train_model(model, train_loader, val_loader, criterion, optimizer, base_path_model_save, num_epochs=10):
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
            running_corrects = 0

            for i, (inputs, labels) in tqdm(enumerate(loader)):
                print(i, len(loader), end='\r')
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            wandb.log({f"{phase} Loss": epoch_loss, f"{phase} Accuracy": epoch_acc, "epoch": epoch})

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
        if epoch > 0 and epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(base_path_model_save,f'age_classification_model_{epoch}_focal.pth'))


    print(f'Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# Loss function and optimizer
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        targets = nn.functional.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Compute the softmax and log-softmax
        logpt = nn.functional.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)

        # Compute the focal loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt * targets

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device) * targets + (1 - self.alpha.to(inputs.device)) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# Data augmentation

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
    
image_size = 150
transform_train = transforms.Compose([
    #transforms.RandomResizedCrop(448),
    ResizeToMaxDim(image_size),
    PadToSquare(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


transform_val = transforms.Compose([
            ResizeToMaxDim(image_size),
            PadToSquare(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



data = pd.read_csv(os.path.join("/usr/users/vhassle/datasets/lagenda", "cropped_data_age_classes.csv"))

#split of the data
sample_percent = 1#decimal
sampled_data = data.sample(frac=sample_percent, random_state=22)

train_data, val_data = train_test_split(sampled_data, test_size=0.2, random_state=22)

train_dataset = AgeDataset(data = train_data, root_dir="/usr/users/vhassle/datasets/lagenda", transform=transform_train)
val_dataset = AgeDataset(data = val_data, root_dir="/usr/users/vhassle/datasets/lagenda", transform=transform_val)

batch_size = 48
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Modify the final layer for classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Specify alpha as a tensor for three classes with different weights
alpha = torch.tensor([1, 0.3, 0.1])  # High weight for class 0, medium for class 1, and low for class 2
gamma = 2
criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
#criterion = nn.CrossEntropyLoss()
wandb.init(project='Age_estimation_focal_faces')
config = wandb.config
config.alpha = alpha
config.gamma = gamma
config.learning_rate = 0.00005
config.sample_percent = sample_percent
config.batch_size = batch_size
config.image_size_input = image_size

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)




# Train the model
base_path_model_save = "/usr/users/vhassle/psych_track/AgeSelf/models/faces"
os.makedirs(base_path_model_save, exist_ok=True)
model = train_model(model, train_loader, val_loader, criterion, optimizer,base_path_model_save, num_epochs=30)

# Save the model
torch.save(model.state_dict(), os.path.join(base_path_model_save,'age_classification_model_final_focalpth'))

wandb.finish()