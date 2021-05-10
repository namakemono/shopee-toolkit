import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, log_loss
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

import cv2
import pickle

# Preliminaries
from tqdm import tqdm
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# albumentations for augs
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

#torch
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import shopee.registry

class Config:
    debug = False
    BASE_DIR = '../input'
    num_workers = 4
    batch_size = 32
    epochs = 10
    seed = 100
    n_fold = 2
    trn_fold = [0,1]
    lr = 3e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scheduler = 'CosineAnnealingWarmRestarts' #['CosineAnnealingLR', 'ReduceLROnPlateau']
    patience=4 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    T_max=10 # CosineAnnealingLR
    T_0=4 # CosineAnnealingWarmRestarts
    min_lr=1e-6
    model_params = {
        'n_classes':11014,
        'model_name': 'efficientnet_b3', #'resnext50_32x4d'
        'use_fc':False,
        'fc_dim':512,
        'dropout':0.0,
        'loss_module':'arcface', #['cosface', 'adacos', 'softmax']
        's':30.0,
        'margin':0.50,
        'easy_margin':False,
        'ls_eps':0.0,
        'theta_zero':0.785,
        'pretrained':False,
    }


    def __init__(self,
        image_size:int, # e.g. 256, 512
        weights_filepath:str
    ):
        self.dim = (image_size, image_size)
        self.weights_filepath = weights_filepath

config = Config(
    image_size=256,
    weights_filepath="../input/effnetb3-256x256-arcface/train-embeddings-effnet-b3_256x256-kf0.npy"
)

class ShopeeDataset(Dataset):
    def __init__(self, csv, transforms=None):
        self.csv = csv.reset_index()
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=config.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class ShopeeNet(nn.Module):

    def __init__(self, n_classes, model_name='resnext50_32x4d', use_fc=False, fc_dim=512, dropout=0.0,
                 loss_module='softmax', s=30.0, margin=0.50, easy_margin=False,ls_eps=0.0, theta_zero=0.785,
                 pretrained=True):

        super(ShopeeNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.model_name = model_name

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if 'efficient' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        elif 'resne' in model_name:
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        elif 'nfnet' in model_name:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        elif 'vit' in model_name:
            final_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise Exception('unknown model found...')

        self.pooling =  nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        if self.use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module

        if self.loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif self.loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif self.loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return feature, logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        if 'vit' not in self.model_name:
            x = self.pooling(x).view(batch_size, -1)
        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x

def eval_fn(data_loader,model,device):
    model.eval()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))

    embeds = []

    with torch.no_grad():
        for bi,image in tk0:
            image = image.to(device)
            feature = model.extract_feat(image)

            image_embeddings = feature.detach().cpu().numpy()
            embeds.append(image_embeddings)

    image_embeddings = np.concatenate(embeds)

    return image_embeddings

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(config.dim[0],config.dim[1],always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

def get_image_embeddings(df:pd.DataFrame, image_size:int, weights_filepath:str) -> np.ndarray:
    config.dim = (image_size, image_size)
    config.weights_filepath = weights_filepath

    #Defining dataloader
    valid_dataset = ShopeeDataset(
            csv=df,
            transforms=get_valid_transforms(),
        )

    valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    # Defining Device
    device = config.device

    #Defining model
    model = ShopeeNet(**config.model_params)
    model.load_state_dict(torch.load(weights_filepath))
    model = model.to(device)

    image_embeddings = eval_fn(valid_loader, model, device)
    return image_embeddings



