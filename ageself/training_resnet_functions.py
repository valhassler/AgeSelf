# training_resnet_functions.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import models
from tqdm import tqdm
from PIL import Image

class AgeGenderDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.loc[idx, "img_name"])
        image = Image.open(img_path)
        age_class = self.data.loc[idx, "age_class"]
        gender = 0 if self.data.loc[idx, "gender"] == 'F' else 1
        if self.transform:
            image = self.transform(image)
        return image, int(age_class), int(gender)

class AgeGenderResNet(nn.Module):
    def __init__(self, num_age_classes=3, num_gender_classes=2):
        super(AgeGenderResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.age_fc = nn.Linear(num_ftrs, num_age_classes)
        self.gender_fc = nn.Linear(num_ftrs, num_gender_classes)

    def forward(self, x):
        x = self.resnet(x)
        age_out = self.age_fc(x)
        gender_out = self.gender_fc(x)
        return age_out, gender_out

def load_age_gender_resnet(model_path):
    model_a_g = AgeGenderResNet()
    model_a_g.load_state_dict(torch.load(model_path))
    model_a_g = model_a_g.to("cuda")
    model_a_g.eval()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = nn.functional.one_hot(targets, num_classes=inputs.size(1)).float()
        logpt = nn.functional.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
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

class ResizeToMaxDim:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, img):
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

image_size = 450 #quite small but images are also in the distance
max_rotation = 90
def get_val_transform(image_size=150):
    return transforms.Compose([
        ResizeToMaxDim(image_size),
        PadToSquare(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
def get_train_transform(image_size=150, max_rotation=90):
    return transforms.Compose([
        ResizeToMaxDim(image_size),
        PadToSquare(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(max_rotation),
        transforms.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def train_model(model, train_loader, val_loader, age_criterion, gender_criterion, optimizer, base_path_model_save, num_epochs=10, age_gender_loss = [1,1/50]):
    import wandb 
    device = "cuda"
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
            running_corrects_age = 0
            running_corrects_gender = 0
            for i, (inputs, age, gender) in tqdm(enumerate(loader)):
                print(i, len(loader), end='\r')
                inputs = inputs.to(device)
                age = age.to(device)
                gender = gender.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    age_outputs, gender_outputs = model(inputs)
                    _, age_preds = torch.max(age_outputs, 1)
                    _, gender_preds = torch.max(gender_outputs, 1)
                    age_loss = age_criterion(age_outputs, age)
                    gender_loss = gender_criterion(gender_outputs, gender)

                    loss = age_loss * age_gender_loss[0] + gender_loss * age_gender_loss[1]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects_age += torch.sum(age_preds == age.data)
                running_corrects_gender += torch.sum(gender_preds == gender.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc_age = running_corrects_age.double() / len(loader.dataset)
            epoch_acc_gender = running_corrects_gender.double() / len(loader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Age Acc: {epoch_acc_age:.4f} Gender Acc: {epoch_acc_gender:.4f}')

            wandb.log({
                f"{phase} Loss": epoch_loss,
                f"{phase} Age Accuracy": epoch_acc_age,
                f"{phase} Gender Accuracy": epoch_acc_gender,
                "epoch": epoch
            })

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
        if epoch > 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(base_path_model_save, f'age_gender_classification_model_{epoch}.pth'))

    print(f'Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model
