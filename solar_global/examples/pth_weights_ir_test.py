import glob
from itertools import chain
import os
import random
import zipfile
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import faiss
from solar_global.networks.imageretrievalnet import init_network


PATH_TRAIN = "/home/human/Diana_Iakovleva/datasets/gauge_image_retrieval/pytorch_flower/train/"
PATH_VALID = "/home/human/Diana_Iakovleva/datasets/gauge_image_retrieval/pytorch_flower/val/"
PATH_TEST = "/home/human/Diana_Iakovleva/datasets/gauge_image_retrieval/pytorch_flower/test/"
PATH_WEIGHTS = "/home/human/Diana_Iakovleva/image_retrieval/SOLAR/data/networks/model_best.pth"

class TripletData(Dataset):
    def __init__(self, path, transforms, split="train"):
        self.path = path
        self.split = split    # train or valid
        self.cats = 4       # number of categories
        self.transforms = transforms
    
    def __getitem__(self, idx):
        # our positive class for the triplet
        idx = str(idx%self.cats + 1)
        
        # choosing our pair of positive images (im1, im2)
        positives = os.listdir(os.path.join(self.path, idx))
        im1, im2 = random.sample(positives, 2)
        
        # choosing a negative class and negative image (im3)
        negative_cats = [str(x+1) for x in range(self.cats)]
        negative_cats.remove(idx)
        negative_cat = str(random.choice(negative_cats))
        negatives = os.listdir(os.path.join(self.path, negative_cat))
        im3 = random.choice(negatives)
        
        im1,im2,im3 = os.path.join(self.path, idx, im1), os.path.join(self.path, idx, im2), os.path.join(self.path, negative_cat, im3)
        
        im1 = self.transforms(Image.open(im1))
        im2 = self.transforms(Image.open(im2))
        im3 = self.transforms(Image.open(im3))
        
        return [im1, im2, im3]
        
    # we'll put some value that we want since there can be far too many triplets possible
    # multiples of the number of images/ number of categories is a good choice
    def __len__(self):
        return self.cats*8
    

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Datasets and Dataloaders
train_data = TripletData(PATH_TRAIN, train_transforms)
val_data = TripletData(PATH_VALID, val_transforms)

model_params = {}
# model_params['architecture'] = models.resnet101()
# model_params['pooling'] = 'gem'
# model_params['p'] = 3
# # model_params['local_whitening'] = True
# # model_params['regional'] = args.regional
# # model_params['whitening'] = args.whitening
# # # model_params['mean'] = ...  # will use default
# # # model_params['std'] = ...  # will use default
# # model_params['pretrained'] = args.pretrained
# # model_params['pretrained_type'] = args.pretrained_type
# # model_params['flatten_desc'] = args.flatten_desc
# # model_params['soa'] = args.soa
# model_params['soa_layers'] = 45

model = init_network(model_params)

model.cuda()
checkpoint = torch.load(PATH_WEIGHTS) # ie, model_best.pth.tar
model.load_state_dict(checkpoint['state_dict'])



# print(model)
# model = models.resnet101()

# checkpoint = torch.load(PATH_WEIGHTS, map_location=torch.device('cuda:0'))['state_dict']
# for key in list(checkpoint.keys()):
#     if 'features.' in key:
#         checkpoint[key.replace('features.', '')] = checkpoint[key]
#         del checkpoint[key]
# model.load_state_dict(checkpoint)

# model.load_state_dict(torch.load(PATH_WEIGHTS))
# model = model.cuda()
# print(model)
model.eval()

faiss_index = faiss.IndexFlatL2(2048)   # build the index

im_indices = []
with torch.no_grad():
    for f in glob.glob(os.path.join(PATH_TRAIN, '*/*')):
        im = Image.open(f)
        im = im.resize((224,224))
        im = torch.tensor([val_transforms(im).numpy()]).cuda()
    
        preds = model(im)
        preds = np.array([preds[0].cpu().numpy()])
        # print(len(preds[0]))
        faiss_index.add(preds) #add the representation to index
        im_indices.append(f)   #store the image name to find it later on

with torch.no_grad():
    
    f = '4.jpg'
    im = Image.open(os.path.join(PATH_TEST,f))
    im = im.resize((224,224))
    im.show()
    im = torch.tensor([val_transforms(im).numpy()]).cuda()
    
    test_embed = model(im).cpu().numpy()

    _, I = faiss_index.search(test_embed, 5)
    print("Retrieved Image: {}".format(im_indices[I[0][0]]))
    print(im_indices[I[0][0]])
    for i in range(5):
        path_str = str(im_indices[I[0][i]])
        print(path_str)
        res = cv2.imread(path_str)

        width = 640
        height = 640
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(res, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('Res', resized)
        cv2.waitKey(0)