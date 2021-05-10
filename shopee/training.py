
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
import networkx as nx
import xgboost as xgb
import Levenshtein
import shopee
from .feature_extraction import to_edit_distance, add_features, add_graph_features
from .model_neuralnet import NeuralNet
from catboost import CatBoostClassifier
from catboost import Pool

def split_train_test_pair(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame,
    pair_df:pd.DataFrame,
):
    # 訓練データとテストデータの分離
    condition1 = pair_df["posting_id"].isin(test_df["posting_id"].unique())
    condition2 = pair_df["candidate_posting_id"].isin(test_df["posting_id"].unique())
    is_test_data = (condition1 & condition2)
    train_pair_df = pair_df[~is_test_data].reset_index(drop=True)
    test_pair_df = pair_df[is_test_data].reset_index(drop=True)
    return train_pair_df, test_pair_df

def train(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame,
    train_pair_df:pd.DataFrame,
    test_pair_df:pd.DataFrame,
    weight_rate:float,
    num_kfolds:int,
    feature_names:List[str],
    model_name:str,          # xgb or nn で指定
    nn_params:dict = {
                "epochs": 20,
                "bs": 512,
                "hidden_size": 100,
                "dropout_rate": 0.05,
                "layer_num": 3,
                "scheduler": "CosineAnnealingWarmRestarts",
                "T_0": 5,
                "lr": 5e-2,
                "min_lr": 5e-4,
                "momentum": 0.9,
                "early_stopping_step": 5,
                "early_stop": True,
                "seed": 41,
                "num_class": 2,
            }
):

    if model_name=="nn": # 欠損値の処理
        all_df = pd.concat([train_pair_df,test_pair_df])
        all_df[feature_names] = all_df[feature_names].fillna(all_df[feature_names].mean())
        train_pair_df = all_df.iloc[:len(train_pair_df)].reset_index(drop=True)
        test_pair_df = all_df.iloc[len(train_pair_df):].reset_index(drop=True)

    y_preda_list = []

    X = train_pair_df[feature_names].values
    y = train_pair_df["matched"].values
    X_test = test_pair_df[feature_names].values
    oof = np.zeros(len(y))

    posting_id_to_fold = train_df.set_index("posting_id")["fold"].to_dict()
    train_pair_df["fold_pid"] = train_pair_df["posting_id"].map(posting_id_to_fold)
    train_pair_df["fold_cpid"] = train_pair_df["candidate_posting_id"].map(posting_id_to_fold)
    for kfold_index in range(num_kfolds):
        # ペア時にリークが入らないよう分割
        # 学習に使うだけで推論だと別途対応する必要あり
        if True:
            # validが入らない
            train_fold_list = [i for i in train_pair_df["fold_pid"].unique() if (i % num_kfolds != kfold_index)]
            pid_train = train_pair_df["fold_pid"].isin(train_fold_list)
            pid_valid = train_pair_df["fold_pid"] % num_kfolds == kfold_index
            pid_test = train_pair_df["posting_id"].str.contains("test_", na=False)
            cpid_train = train_pair_df["fold_cpid"].isin(train_fold_list)
            cpid_valid = train_pair_df["fold_cpid"] % num_kfolds == kfold_index
            cpid_test = train_pair_df["candidate_posting_id"].str.contains("test_", na=False)

            is_even_pid = train_pair_df["fold_pid"] % 2 == 0
            is_even_cpid = train_pair_df["fold_cpid"] % 2 == 0

            train_index = train_pair_df[
                (pid_train & cpid_train) |
                (pid_train & cpid_test & ~is_even_pid) |
                (pid_test & cpid_train & ~is_even_cpid)
            ].index
            valid_index = train_pair_df[
                (pid_valid & cpid_valid)
            ].index

        else: # pidとcpidの反転例が存在しちゃってまずいペア
            train_index = train_pair_df[(train_pair_df["fold_pid"] % num_kfolds) != kfold_index].index
            valid_index = train_pair_df[(train_pair_df["fold_pid"] % num_kfolds) == kfold_index].index

        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        pid_valid = train_pair_df["fold_pid"] % num_kfolds == kfold_index
        pid_test = train_pair_df["posting_id"].str.contains("test_", na=False)
        cpid_valid = train_pair_df["fold_cpid"] % num_kfolds == kfold_index
        cpid_test = train_pair_df["candidate_posting_id"].str.contains("test_", na=False)
        is_even_pid = train_pair_df["fold_pid"] % 2 == 0
        is_even_cpid = train_pair_df["fold_cpid"] % 2 == 0
        valid_index = train_pair_df[
            (pid_valid & cpid_valid) |
            (pid_valid & cpid_test & ~is_even_pid) |
            (pid_test & cpid_valid & ~is_even_cpid)
        ].index

        if model_name=="xgb":
            # 正例の重みを weight_rate, 不例を1にする
            sample_weight = np.ones(y_train.shape)
            sample_weight[y_train==1] = weight_rate
            sample_weight_val = np.ones(y_valid.shape)
            sample_weight_val[y_valid==1] = weight_rate
            sample_weight_eval_set = [sample_weight, sample_weight_val]

            clf = xgb.XGBClassifier(
                objective       = "binary:logistic",
                max_depth       = 5,                    # 6,
                n_estimators    = 1463,                 # 1000,
                learning_rate   = 0.2832336307209388,
                tree_method     = 'gpu_hist'
            )
            print(clf)
            clf.fit(
                X_train, y_train,
                eval_set=[
                    (X_train, y_train),
                    (X_valid, y_valid)
                ],
                eval_metric             = "logloss",
                verbose                 = 10,
                early_stopping_rounds   = 20,
                callbacks               = [],
                sample_weight           = sample_weight,
                sample_weight_eval_set  = sample_weight_eval_set,
            )
            # pidがvalidを含む項目を全て予測に入れる
            oof[valid_index] = clf.predict_proba(X[valid_index])[:,1]
            y_preda = clf.predict_proba(X_test)[:,1]
            y_preda_list.append(y_preda)

        elif model_name=="cat":
            train_pool = Pool(X_train, label=y_train)
            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.011730134049031867,
                depth=10,
                task_type="GPU", # gpu mode
            ) #調整可能
            #model = CatBoostClassifier( iterations=1000,learning_rate=0.021730134049031867, depth=8,         )
            model.fit(train_pool, verbose=True,metric_period=100)
            # pidがvalidを含む項目を全て予測に入れる
            valid_pool = Pool(X[valid_index], label=y[valid_index])
            test_pool = Pool(X_test)
            oof[valid_index] = model.predict_proba(valid_pool)[:,1]
            y_preda = model.predict_proba(test_pool)[:,1]
            y_preda_list.append(y_preda)

        elif model_name=="nn":
            nn_params["weight_rate"] = weight_rate
            allvalid_index = valid_index
            X_allvalid = X[allvalid_index]
            print("X_train:",len(X_train), "X_valid:",len(X_valid)," X_allvalid:", len(X_allvalid))

            classifier = NeuralNet(fold_num=kfold_index)
            y_pred,  y_allvalid_pred = classifier.train_and_predict(
                X_train,
                X_valid,
                y_train,
                y_valid,
                X_test,
                X_allvalid,
                nn_params
            )
            oof[allvalid_index] = y_allvalid_pred[:,1]
            y_preda_list.append(y_pred[:,1])


    train_pair_df[f"confidence_{model_name}"] = oof
    test_pair_df[f"confidence_{model_name}"] = np.mean(y_preda_list, axis=0)
    return train_pair_df, test_pair_df
