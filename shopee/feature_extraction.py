import numpy as np
import pandas as pd

def to_edit_distance(row:dict) -> float:
    a = row["posting_id_phash"]
    b = row["candidate_posting_id_phash"]
    return Levenshtein.distance(a, b)
    
def add_features(pair_df:pd.DataFrame, to_image_phash:dict) -> pd.DataFrame:
    pair_df["posting_id_phash"] = pair_df["posting_id"].map(to_image_phash)
    pair_df["candidate_posting_id_phash"] = pair_df["candidate_posting_id"].map(to_image_phash)
    pair_df["edit_distance"] = pair_df.apply(to_edit_distance, axis=1)
    pair_df["img_text_feat"] = pair_df["img_feat"]*df["text_feat"]
    return df

def make_graph_features(pair_df:pd.DataFrame):
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


