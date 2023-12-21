import torch
import torch.nn as nn
import torchvision.models as models
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
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
import faiss
import os
import pickle
import random
import time
import cv2
from transformers import AutoModel, AutoFeatureExtractor
from prettytable import PrettyTable

model_name = "microsoft/dit-base"
model = AutoModel.from_pretrained(model_name)

np.object = np.object_
np.int = np.int_
np.bool = np.bool_

def count_images(folder_path):
    image_formats = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif']  # Add more formats if needed
    total_image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in image_formats):
                total_image_count += 1
    return total_image_count



class Triplet:
    def __init__(self, train_folder):
        self.train_folder = train_folder
        self.labels = [label for label in os.listdir(train_folder) if label != '.ipynb_checkpoints']
        self.label_to_path = {}
        
        for label in self.labels:
            label_path = os.path.join(train_folder, label)
            subdirectories = [subdir for subdir in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, subdir))]
            leaf_nodes = [os.path.join(label_path, subdir) for subdir in subdirectories if self.has_images(os.path.join(label_path, subdir))]
            LEAF_NODES = []
            for i in range(len(leaf_nodes)):
                sub_sub = self.list_subfolders(leaf_nodes[i])
                if sub_sub is not None:
                    for patH in sub_sub:
                        LEAF_NODES.append(os.path.join(leaf_nodes[i],patH))
                else:
                    LEAF_NODES.append(leaf_nodes[i])
            self.label_to_path[label] = LEAF_NODES

    def list_subfolders(self, folder_path):
        if os.path.isdir(folder_path):
            subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
            if subfolders:
                return subfolders
            return None
        return None

    def has_images(self, directory):
        # Check if the directory contains images
        for root, dirs, files in os.walk(directory):
            image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif'))]
            if image_files:
                return True
        return False
    def get_triplet(self):
        # Select anchor class with available images
        anchor_label = random.choice([label for label in self.labels if any(self.label_to_path[label])])
        anchor_subdir = random.choice(self.label_to_path[anchor_label])
        anchor_image = self.get_random_image(anchor_subdir)

        # Select positive example from the same class
        positive_label = anchor_label
        positive_subdir = anchor_subdir
        positive_image = self.get_random_image(positive_subdir)

        # Ensure the anchor and positive images are different
        while anchor_image == positive_image:
            positive_image = self.get_random_image(positive_subdir)

        # Select negative example from a different class with available images
        available_labels = [label for label in self.labels if label != anchor_label and any(self.label_to_path[label])]
        negative_label = random.choice(available_labels)
        negative_subdir = random.choice(self.label_to_path[negative_label])
        negative_image = self.get_random_image(negative_subdir)

        anchor_label_num = self.labels.index(anchor_label)
        positive_label_num = self.labels.index(positive_label)
        negative_label_num = self.labels.index(negative_label)

        anchor_label_name = f"{anchor_label}-{os.path.basename(anchor_subdir)}"
        positive_label_name = f"{positive_label}-{os.path.basename(positive_subdir)}"
        negative_label_name = f"{negative_label}-{os.path.basename(negative_subdir)}"

        return anchor_image, positive_image, negative_image, anchor_label_num, positive_label_num, negative_label_num, anchor_label_name, positive_label_name, negative_label_name

    def get_random_image(self, directory):
        # Recursively search for images in the directory
        for root, dirs, files in os.walk(directory):
            image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif'))]
            if image_files:
                return os.path.join(root, random.choice(image_files))


class TripletDataset(Dataset):
    def __init__(self, train_folder, length, transform=None):
        self.triplet_generator = Triplet(train_folder)
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        anchor_image, positive_image, negative_image, anum, pnum, nnum, aname, pname, nname = self.triplet_generator.get_triplet()
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
        anchor_image, positive_image, negative_image ,_ ,_, _, a,p,n= self.triplet_generator.get_triplet()
        return anchor_image, positive_image, negative_image

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),           # Convert to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the tensor
])

TRAIN_FOLDER = "./Datasets/DOCZ-II/train"
TEST_FOLDER = "./Datasets/DOCZ-II/test"
bs = 1
image_count = count_images(TRAIN_FOLDER)


class TEmbeddingNet(nn.Module):
    def __init__(self, modelt):
        super(TEmbeddingNet, self).__init__()
        self.modelt = modelt
        self.conv = nn.Conv1d(in_channels=197, out_channels=1, kernel_size=3)
        
    def forward(self, x):
        x = self.modelt(x)  # Shape: (batch_size, 2048, H, W)
        x = x.last_hidden_state
        x = self.conv(x)
        return x
    
    def get_embedding(self, x):
        x = self.modelt(x)  # Shape: (batch_size, 2048, H, W)
        x = x.last_hidden_state
        x = self.conv(x)
        return x

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

#Loss, Device, Parameters

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

model = TripletNet(tmodel)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(TripletNet(tmodel))
else:
    model = TripletNet(tmodel)
# Move the model to the selected device (CPU or GPU)
model = model.to(device)

margin = 1
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)  # Learning rate scheduler
loss_fn = TripletLoss(margin)
clip_value = 0.5  # You can adjust this value as needed
triplet_dataset = TripletDataset(TRAIN_FOLDER, length= image_count, transform=transform)
train_loader = torch.utils.data.DataLoader(triplet_dataset, batch_size=1)

from tqdm import tqdm

def fit(model, num_epochs, bs):
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        train_loss = 0.0

        for idx, batch in tqdm(enumerate(train_loader)):
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
        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, TIME: {time.time()-start}")
        torch.save(model.state_dict(), f"model{epoch+1}.pth")
        scheduler.step()

print(f"TOTAL IMAGES : {image_count}")
fit(model, n_epochs:=int(input("NO OF EPOCHS : ")), bs)


import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
        self.transform = transform

    def _find_classes(self):
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            for root, dirs, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        path = os.path.join(root, file)
                        rel_path = os.path.relpath(path, self.root_dir)
                        samples.append((path, target_class, rel_path.split(os.path.sep)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target_class, rel_path = self.samples[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target_class, rel_path

train_dataloader = DataLoader(CustomDataset(TRAIN_FOLDER,transform=transform))
test_dataloader = DataLoader(CustomDataset(TEST_FOLDER,transform=transform))

def xtract(loader):
    LABELS = []
    embs = None
    print("EXTRACTING EMBS...")
    for i in tqdm(loader):
        img, cls, path = i
        if len(path)==4:
            LABELS.append([cls[0], path[1][0], path[2][0]])
        else:
            LABELS.append([cls[0], path[1][0]])
        emb = model(img.to(device)).detach()
        if embs is None:
            embs = emb
        else:
            embs = torch.cat((embs, emb), dim=0)
    return embs, LABELS

trainembs, trainlabs = xtract(train_dataloader)
testembs, testlabs = xtract(test_dataloader)

print("INDEX BEING CREATED....")
embs_cpu_np = trainembs.cpu().numpy()
del trainembs
embs_cpu_np = embs_cpu_np.reshape(embs_cpu_np.shape[0], -1)
index = faiss.IndexHNSWFlat(embs_cpu_np.shape[1], 32)  # M = 32 for the HNSW index
index.add(embs_cpu_np)


import os
from PIL import Image

def print_accuracy_table(accuracy_dict):
    keys = list(accuracy_dict.keys())

    # Create a table with the keys in the first row and first column
    table = PrettyTable()
    table.field_names = [''] + keys

    # Fill in the table with accuracy values
    for key_row in keys:
        row_data = [key_row]
        for key_col in keys:
            if key_col in accuracy_dict[key_row]:
                accuracy = f'{accuracy_dict[key_row][key_col]:.2f}%'
            else:
                accuracy = '0%'
            row_data.append(accuracy)

        table.add_row(row_data)

    # Print the table
    print(table)

def tempimgcount(root_dir):
    count = 0

    for first_level_folder in os.listdir(root_dir):
        first_level_path = os.path.join(root_dir, first_level_folder)

        if os.path.isdir(first_level_path):
            for second_level_folder in os.listdir(first_level_path):
                second_level_path = os.path.join(first_level_path, second_level_folder)

                if os.path.isdir(second_level_path):
                    for third_level_folder in os.listdir(second_level_path):
                        third_level_path = os.path.join(second_level_path, third_level_folder)

                        if os.path.isdir(third_level_path):
                            # Assuming images have common extensions like .png, .jpg, .jpeg
                            image_files = [file for file in os.listdir(third_level_path) if file.lower().endswith(('.png', '.jpg','.tif','.jpeg'))]
                            count += len(image_files)

    return count

temp_img_count = tempimgcount(TEST_FOLDER)

def count_imgs_in_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.gif', '.tif']  # Add more extensions if needed
    image_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1

    return image_count

def percentage(d):
    D = d
    for key in d:
        LEN = count_imgs_in_folder(os.path.join(TEST_FOLDER,key))
        for clss in d[key]:
            D[key][clss] = d[key][clss]/LEN*100
    return D

def evaluate_with_faiss(embs, index):
    outer_class_wise = {}
    inner_class_wise = {}
    outer_wrongly = {}
    inner_wrongly = {}
    TOTAL = len(embs)
    start = time.time()
    tempwise = 0
    outer = 0
    inner = 0
    # Initialize the tqdm progress bar
    with tqdm(total=TOTAL) as pbar:
        for idx, emb in enumerate(embs):
            #search
            label = index.search(emb.reshape(1, -1), 1)[1][0][0]
            #percentage of predictions
            if testlabs[idx][0] not in outer_wrongly:
                outer_wrongly[testlabs[idx][0]] = {testlabs[idx][0]:0}
            if trainlabs[label][0] not in outer_wrongly[testlabs[idx][0]]: 
                outer_wrongly[testlabs[idx][0]][trainlabs[label][0]] = 1
            else:
                outer_wrongly[testlabs[idx][0]][trainlabs[label][0]] += 1
                
            if testlabs[idx][1] not in inner_wrongly:
                inner_wrongly[testlabs[idx][1]] = {testlabs[idx][1]:0}
            
            if trainlabs[label][1] not in inner_wrongly[testlabs[idx][1]]:
                inner_wrongly[testlabs[idx][1]][trainlabs[label][1]] = 1
            else:
                inner_wrongly[testlabs[idx][1]][trainlabs[label][1]]+=1

            if trainlabs[label][0] == testlabs[idx][0]:
                if testlabs[idx][0] not in outer_class_wise:
                    outer_class_wise[testlabs[idx][0]]=1
                else:
                    outer_class_wise[testlabs[idx][0]]+=1
                    
                outer += 1
                if trainlabs[label][1] == testlabs[idx][1]:
                    if testlabs[idx][1] not in inner_class_wise:
                        inner_class_wise[testlabs[idx][1]]=1
                    else:
                        inner_class_wise[testlabs[idx][1]]+=1
                    inner += 1
                    if len(trainlabs[label])==3 and len(testlabs[idx])==3 and trainlabs[label][2]==testlabs[idx][2]:
                        tempwise+=1
            pbar.update(1)  # Update the progress bar
    oaccuracy = (outer / TOTAL) * 100
    iaccuracy = (inner / TOTAL) * 100
    taccuracy = (tempwise / temp_img_count) * 100
    elapsed_time = time.time() - start
    print_accuracy_table(percentage(outer_wrongly))
    return f'OUTER Accuracy: {oaccuracy:.2f}%, INNER Accuracy: {iaccuracy:.2f}%, TEMPWISE Accuracy: {taccuracy:.2f}% '

embs2_cpu_np = testembs.cpu().numpy()
embs2_cpu_np = embs2_cpu_np.reshape(embs2_cpu_np.shape[0], -1)
print(f'IndexHNSWFlat : {evaluate_with_faiss(embs2_cpu_np,index)}')
