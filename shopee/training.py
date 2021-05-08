
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
import networkx as nx
import xgboost as xgb
import Levenshtein
import shopee
from .feature_extraction import to_edit_distance, add_features, add_graph_features

def run(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame,
    pair_df:pd.DataFrame,
    weight_rate:float,
    use_graph_features:bool,
    num_kfolds:int,
    feature_names:List[str],
):
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    
    # 特徴量の追加
    to_image_phash = df.set_index("posting_id")["image_phash"].to_dict()
    add_features(pair_df, to_image_phash)
    if use_graph_features:
        add_graph_features(pair_df)

    # 訓練データとテストデータの分離
    condition1 = pair_df["posting_id"].isin(test_df["posting_id"].unique())
    condition2 = pair_df["candidate_posting_id"].isin(test_df["posting_id"].unique())
    is_test_data = (condition1 & condition2)
    train_pair_df = pair_df[~is_test_data].reset_index(drop=True)
    test_pair_df = pair_df[is_test_data].reset_index(drop=True)
    test_pair_df["confidence"] = -1

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
            pid_valid = train_pair_df["fold_pid"] == kfold_index
            pid_test = train_pair_df["posting_id"].str.contains("test_", na=False)
            cpid_train = train_pair_df["fold_cpid"].isin(train_fold_list)
            cpid_valid = train_pair_df["fold_cpid"] == kfold_index
            cpid_test = train_pair_df["candidate_posting_id"].str.contains("test_", na=False)
            train_index = train_pair_df[
                (pid_train & cpid_train) | 
                (pid_train & cpid_test) |
                (pid_test & cpid_train)
            ].index
            valid_index = train_pair_df[
                (pid_valid & cpid_valid) |
                (pid_valid & cpid_test) |
                (pid_test & cpid_valid)
            ].index
        else: # pidとcpidの反転例が存在しちゃってまずいペア
            train_index = train_pair_df[(train_pair_df["fold_pid"] % num_kfolds) != kfold_index].index
            valid_index = train_pair_df[(train_pair_df["fold_pid"] % num_kfolds) == kfold_index].index
       
        # 分割前に構築
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # 正例の重みを weight_rate, 不例を1にする
        sample_weight = np.ones(y_train.shape)
        sample_weight[y_train==1] = weight_rate
        sample_weight_val = np.ones(y_valid.shape)
        sample_weight_val[y_valid==1] = weight_rate
        sample_weight_eval_set = [sample_weight, sample_weight_val]

        clf = xgb.XGBClassifier(
            objective       = "binary:logistic",
            max_depth       = 6,
            n_estimators    = 1000
        )
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
        valid_index = train_pair_df[(train_pair_df["fold_pid"] % num_kfolds) == kfold_index].index
        oof[valid_index] = clf.predict_proba(X[valid_index])[:,1]
        y_preda = clf.predict_proba(X_test)[:,1]
        y_preda_list.append(y_preda)

    train_pair_df["confidence"] = oof
    test_pair_df["confidence"] = np.mean(y_preda_list, axis=0)
    return train_pair_df, test_pair_df


