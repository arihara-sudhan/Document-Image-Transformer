import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam,lr_scheduler
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.nn.parallel import DataParallel
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import pickle
import random
import time
import cv2
np.object = np.object_
np.int = np.int_
np.bool = np.bool_
from transformers import AutoModel, AutoFeatureExtractor
model_name = "microsoft/dit-base"
model = AutoModel.from_pretrained(model_name)


class Triplet:
    def __init__(self, train_folder):
        self.train_folder = train_folder
        self.labels = [label for label in os.listdir(train_folder) if label != '.ipynb_checkpoints']
        self.label_to_path = {label: os.path.join(train_folder, label) for label in self.labels}
    
    def get_triplet(self):
        anchor_label = random.choice(self.labels)
        anchor_path = random.choice(os.listdir(self.label_to_path[anchor_label]))
        positive_label = anchor_label
        positive_path = random.choice(os.listdir(self.label_to_path[positive_label]))
        negative_label = random.choice([label for label in self.labels if label != anchor_label])
        negative_path = random.choice(os.listdir(self.label_to_path[negative_label]))
        
        anchor_image = os.path.join(self.label_to_path[anchor_label], anchor_path)
        positive_image = os.path.join(self.label_to_path[positive_label], positive_path)
        negative_image = os.path.join(self.label_to_path[negative_label], negative_path)
        
        anchor_label_num = self.labels.index(anchor_label)
        positive_label_num = self.labels.index(positive_label)
        negative_label_num = self.labels.index(negative_label)
        
        return anchor_image, positive_image, negative_image

class TripletDataset(Dataset):
    def __init__(self, train_folder, length, transform=None,):
        self.triplet_generator = Triplet(train_folder)
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        anchor_image, positive_image, negative_image = self.triplet_generator.get_triplet()
        anchor = self._load_image(anchor_image)
        positive = self._load_image(positive_image)
        negative = self._load_image(negative_image)
        return anchor, positive, negative

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_triplet_names(self, index):
        anchor_image, positive_image, negative_image = self.triplet_generator.get_triplet()
        return anchor_image, positive_image, negative_image



class TEmbeddingNet(nn.Module):
    def __init__(self, modelt):
        super(TEmbeddingNet, self).__init__()
        self.modelt = modelt

    def forward(self, x):
        x = self.modelt(x)  # Shape: (batch_size, 2048, H, W)
        return x.last_hidden_state

    def get_embedding(self, x):
        x = self.modelt(x)  # Shape: (batch_size, 2048, H, W)
        return x.last_hidden_state

tmodel = TEmbeddingNet(model)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.enet = embedding_net

    def forward(self, x1, x2=None, x3=None):
        if x2 is None and x3 is None:
            return self.enet.get_embedding(x1)
        return self.enet.get_embedding(x1),self.enet.get_embedding(x2),self.enet.get_embedding(x3)

    def get_embedding(self, x):
        return self.enet.get_embedding(x)


# In[4]:


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


# In[6]:


train_folder = "/kaggle/input/ocredmergeddocs/MINI-DOCUMENTS/train"
bs = 8  # You can adjust this as nee
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),           # Convert to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the tensor
])

triplet_dataset = TripletDataset(train_folder, length=1714, transform=transform)  # Adjust the length as needed
# Create a DataLoader
train_data_loader = torch.utils.data.DataLoader(triplet_dataset, batch_size=bs, shuffle=True)

margin = 1
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)  # Learning rate scheduler
loss_fn = TripletLoss(margin)
clip_value = 0.5  # You can adjust this value as needed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TripletNet(tmodel)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(TripletNet(tmodel))
else:
    model = TripletNet(tmodel)

# Move the model to the selected device (CPU or GPU)
model = model.to(device)


# In[7]:


def fit(model, num_epochs, train_loader, bs):
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        train_loss = 0.0

        for idx, batch in enumerate(train_loader):
            anchor, positive, negative = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            anchor_embedding.requires_grad_(True)
            positive_embedding.requires_grad_(True)
            negative_embedding.requires_grad_(True)
            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_loss += loss.item()
            print(f"EPOCH: {epoch+1}. ({idx + 1}).  LOSS : {loss.item()}  SEEN : {bs * (idx + 1)}/{len(train_loader.dataset)}")
        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, TIME: {time.time()-start}")
        scheduler.step()

fit(model, n_epochs:=int(input("NO OF EPOCHS : ")), train_data_loader, bs)
