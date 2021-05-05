import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
import networkx as nx
import xgboost as xgb
import Levenshtein
import shopee

def to_edit_distance(row:dict) -> float:
    a = row["posting_id_phash"]
    b = row["candidate_posting_id_phash"]
    return Levenshtein.distance(a, b)

def add_features(df:pd.DataFrame, to_image_phash) -> pd.DataFrame:
    df["posting_id_phash"] = df["posting_id"].map(to_image_phash)
    df["candidate_posting_id_phash"] = df["candidate_posting_id"].map(to_image_phash)
    df["edit_distance"] = df.apply(to_edit_distance, axis=1)
    df["img_text_feat"] = df["img_feat"]*df["text_feat"]
    return df

def add_graph_features(pair_df:pd.DataFrame):
    print("graph features ...")
    # グラフ構築
    G = nx.Graph()
    for idx in range(len(pair_df)):
        # posting_id から candidate_positing id への辺
        G.add_edge(pair_df.posting_id[idx], pair_df.candidate_posting_id[idx])

    # 連結成分の数
    cc_num_dict = {}
    for cc in list(nx.connected_components(G)):
        cc_num = len(cc)
        for posting_id in cc:
            cc_num_dict[posting_id] = cc_num
    # 2つのペアは同じ連結成分内にあるので一つだけで十分
    pair_df["cc_num"] = pair_df["posting_id"].map(cc_num_dict)

    # 次数中心性
    degree_cent_dict = nx.degree_centrality(G)
    pair_df["degree_centrality_pid"] = pair_df["posting_id"].map(degree_cent_dict)
    pair_df["degree_centrality_cpid"] = pair_df["candidate_posting_id"].map(degree_cent_dict)
    return pair_df

def run(
    config,
    train_df:pd.DataFrame,
    test_df:pd.DataFrame,
    pair_df:pd.DataFrame
):
    use_graph_features = config.use_graph_features
    num_kfolds = config.num_kfolds
    feature_names = config.feature_names

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
        train_index = train_pair_df[(train_pair_df["fold_pid"] % num_kfolds) != kfold_index].index
        valid_index = train_pair_df[(train_pair_df["fold_pid"] % num_kfolds) == kfold_index].index
       
        # 分割前に構築
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # 正例の重みを weight_rate, 不例を1にする
        weight_rate = 3.
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
            sample_weight           = sample_weight, # クラス１の重みを２倍にする
            sample_weight_eval_set  = sample_weight_eval_set,
        )
        oof[valid_index] = clf.predict_proba(X[valid_index])[:,1]
        y_preda = clf.predict_proba(X_test)[:,1]
        y_preda_list.append(y_preda)

    train_pair_df["confidence"] = oof
    test_pair_df["confidence"] = np.mean(y_preda_list, axis=0)
    return train_pair_df, test_pair_df


