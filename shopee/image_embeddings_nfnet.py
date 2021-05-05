import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2

import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pickle

import timm

import torch
import torch.nn as nn
import shopee

class ShopeeNet(nn.Module):
    def __init__(self, model_name):
        super(ShopeeNet, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.head.fc = nn.Identity()
        
    def extract_feat(self, x):
        x = self.backbone(x)
        return x
    
    
def get_image_embeddings(df):
    model_name = config.model_name
    valid_dataset = shopee.image_embeddings_pytorch.ShopeeDataset(
            csv=df,
            transforms=shopee.image_embeddings_pytorch.get_valid_transforms(),
        )

    valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=64,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ShopeeNet("dm_nfnet_f0")
    model = model.to(device)
    image_embeddings = shopee.image_embeddings_pytorch.eval_fn(valid_loader, model, device)
    
    return image_embeddings


