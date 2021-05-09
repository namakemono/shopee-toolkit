# ニューラルネットワーク
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import f1_score



def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# データセット
class TrainDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return (self.features.shape[0])
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx], dtype=torch.long)
        }
        return dct
class TestDataset:
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return (self.features.shape[0])
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct

# モデル本体
class NNModel(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout_rate,layer_num):
        super(NNModel, self).__init__()
        self.layer_num = layer_num
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        self.batch_norm_mid = nn.BatchNorm1d(hidden_size)
        self.dropout_mid = nn.Dropout(dropout_rate)
        self.dense_mid = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def recalibrate_layer(self, layer):
        if(torch.isnan(layer.weight_v).sum() > 0):
            print ('recalibrate layer.weight_v')
            layer.weight_v = torch.nn.Parameter(torch.where(torch.isnan(layer.weight_v), torch.zeros_like(layer.weight_v), layer.weight_v))
            layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)
        if(torch.isnan(layer.weight).sum() > 0):
            print ('recalibrate layer.weight')
            layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight)
            layer.weight += 1e-7
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        for _ in range(self.layer_num-2):
            x = self.batch_norm_mid(x)
            x = self.dropout_mid(x)
            x = F.relu(self.dense_mid(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        x = torch.clamp(x, min=-5.0, max=5.0)
        return x

# train,validation,inference
def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        #outputs = torch.clamp(outputs, min=-5.0, max=5.0)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    final_loss /= len(dataloader)
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        #outputs = torch.clamp(outputs, min=-5.0, max=5.0)
        loss = loss_fn(outputs, targets)
        final_loss += loss.item()
        if loss.item()>3:
            print(loss.item())
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    for data in dataloader:
        inputs = data['x'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        #outputs = torch.clamp(outputs, min=-5.0, max=5.0)
        preds.append(outputs.sigmoid().detach().cpu().numpy())
    preds = np.concatenate(preds)
    return preds

def get_scheduler(optimizer,CFG):
    if CFG["scheduler"]=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG["factor"], patience=CFG["patience"], verbose=True, eps=CFG["eps"])
    elif CFG["scheduler"]=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG["T_max"], eta_min=CFG["min_lr"], last_epoch=-1)
    elif CFG["scheduler"]=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"], T_mult=1, eta_min=CFG["min_lr"], last_epoch=-1)
    return scheduler

class NeuralNet():
    def __init__(self, fold_num):
        self.fold_num = fold_num
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, X_allvalid, params):
        seed_torch(params["seed"])
        DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
        EPOCHS = params["epochs"]
        BATCH_SIZE = params["bs"]
        LEARNING_RATE = params["lr"]
        EARLY_STOPPING_STEPS = params["early_stopping_step"]
        EARLY_STOP = params["early_stop"]

        # データセットを生成する
        num_features=X_train.shape[1]
        num_targets= params["num_class"]
        hidden_size= params["hidden_size"]
        dropout_rate= params["dropout_rate"]
        layer_num = params["layer_num"]

        # 上記のパラメータでモデルを学習する
        train_dataset = TrainDataset(X_train, y_train)
        valid_dataset = TrainDataset(X_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = NNModel(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
            dropout_rate = dropout_rate,
            layer_num = layer_num
        )

        model.to(DEVICE)


        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_scheduler(optimizer,CFG=params)
        weights = torch.tensor([1.0, params["weight_rate"]]).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        early_step = 0
        best_loss = 1e9
        for epoch in range(EPOCHS):
            train_loss = train_fn(model, optimizer,scheduler, loss_fn, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
            print(f"FOLD:{self.fold_num}, EPOCH: {epoch}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                y_valid_pred = valid_preds
                torch.save(model.state_dict(), f"./best.pth")
                early_step = 0
            elif(EARLY_STOP == True):
                early_step += 1
                if (early_step >= EARLY_STOPPING_STEPS):
                    break

        testdataset = TestDataset(X_test)
        allvaliddataset = TestDataset(X_allvalid)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
        allvalidloader = torch.utils.data.DataLoader(allvaliddataset, batch_size=BATCH_SIZE, shuffle=False)
        model = NNModel(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
            dropout_rate = dropout_rate,
            layer_num = layer_num
        )
        model.load_state_dict(torch.load(f"./best.pth"))
        model.to(DEVICE)
        y_pred = inference_fn(model, testloader, DEVICE)
        y_allvalid_pred = inference_fn(model, allvalidloader, DEVICE)

        return y_pred, y_allvalid_pred
