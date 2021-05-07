import numpy as np
import pandas as pd
import gc
import Levenshtein
import networkx as nx
import igraph

def to_edit_distance(row:dict) -> float:
    a = row["posting_id_phash"]
    b = row["candidate_posting_id_phash"]
    return Levenshtein.distance(a, b)

def add_features(df:pd.DataFrame, to_image_phash) -> pd.DataFrame:
    df["posting_id_phash"] = df["posting_id"].map(to_image_phash)
    df["candidate_posting_id_phash"] = df["candidate_posting_id"].map(to_image_phash)
    df["edit_distance"] = df.apply(to_edit_distance, axis=1)
    feature_columns = [c for c in df.columns if "feat_" in c]
    n = len(feature_columns)
    for i in range(n):
        for j in range(i+1, n):
            a = feature_columns[i]
            b = feature_columns[j]
            df[f"{a}_x_{b}"] = df[a] * df[b]
    return df

def add_graph_features(pair_df:pd.DataFrame, weight_col = "feat_effnet-b0_256x256_x_feat_tfidf-v1"):
    eps = 1e-8
    print("making graph features ...")
    # グラフ構築
    posting_ids = pair_df["posting_id"].unique()
    N = len(posting_ids)
    V = len(pair_df)
    posting_ids_index = [x for x in range(N)] #pid の一意な番号
    posting_ids_dic = dict(zip(posting_ids, posting_ids_index))
    pair_df["posting_id_index"] = pair_df["posting_id"].map(posting_ids_dic) 
    pair_df["candidate_posting_id_index"] = pair_df["candidate_posting_id"].map(posting_ids_dic) 
    del posting_ids_dic
    _ = gc.collect()
    

    weights = "weight"
    graph= igraph.Graph(n=N, edges=pair_df[["posting_id_index","candidate_posting_id_index"]].values,
      edge_attrs={'weight': pair_df[[weight_col]].values})
    graph.simplify(combine_edges=dict(weight="sum"))


    # 隣接頂点の数。重みありなしは関係なさそう。（密度はちょっと時間がかかる？やり方をもう少し考える）
    ## 1つとなり
    print(" neighborhood_size1")
    neighborhood_size1_dict = dict(zip(posting_ids_index, np.array(graph.neighborhood_size(order=1))))
    pair_df["neighborhood_size1_pid"] = pair_df["posting_id_index"].map(neighborhood_size1_dict) 
    pair_df["neighborhood_size1_cpid"] = pair_df["candidate_posting_id_index"].map(neighborhood_size1_dict) 
    pair_df["neighborhood_size1_diff"] = (pair_df["neighborhood_size1_pid"] - pair_df["neighborhood_size1_cpid"]).abs()
    #pair_df["neighborhood_size1_diff"] = (pair_df["neighborhood_size1_pid"]/(pair_df["neighborhood_size1_cpid"]+eps))
    ## 2つとなり
    print(" neighborhood_size2")
    neighborhood_size2_dict = dict(zip(posting_ids_index, np.array(graph.neighborhood_size(order=2))))
    pair_df["neighborhood_size2_pid"] = pair_df["posting_id_index"].map(neighborhood_size2_dict) 
    pair_df["neighborhood_size2_cpid"] = pair_df["candidate_posting_id_index"].map(neighborhood_size2_dict) 
    pair_df["neighborhood_size2_diff"] = (pair_df["neighborhood_size2_pid"] - pair_df["neighborhood_size2_cpid"]).abs()
    #pair_df["neighborhood_size2_diff"] = (pair_df["neighborhood_size2_pid"] /(pair_df["neighborhood_size2_cpid"]+eps))
    del neighborhood_size1_dict, neighborhood_size2_dict
    _ = gc.collect()
    
    # authority_score, hub_score : weights=weights
    print(" authority_score")
    authority_score_dict = dict(zip(posting_ids_index, np.array(graph.authority_score(weights=weights))))
    pair_df["authority_score_pid"] = pair_df["posting_id_index"].map(authority_score_dict) 
    pair_df["authority_score_cpid"] = pair_df["candidate_posting_id_index"].map(authority_score_dict) 
    pair_df["authority_score_diff"] = (pair_df["authority_score_pid"] - pair_df["authority_score_cpid"]).abs()
    #pair_df["authority_score_diff"] = (pair_df["authority_score_pid"] / (pair_df["authority_score_cpid"]+eps))
    del authority_score_dict
    _ = gc.collect()
    
    # constraint : weights=weights
    print(" constraint")
    constraint_dict = dict(zip(posting_ids_index, np.array(graph.constraint(weights=weights))))
    pair_df["constraint_pid"] = pair_df["posting_id_index"].map(constraint_dict) 
    pair_df["constraint_cpid"] = pair_df["candidate_posting_id_index"].map(constraint_dict) 
    pair_df["constraint_diff"] = (pair_df["constraint_pid"] - pair_df["constraint_cpid"]).abs()
    #pair_df["constraint_diff"] = (pair_df["constraint_pid"] / (pair_df["constraint_cpid"]+eps))
    del constraint_dict
    _ = gc.collect()

    # pagerank: weights=weights
    print(" pagerank")
    pagerank_dict = dict(zip(posting_ids_index, np.array(graph.pagerank(weights=weights))))
    pair_df["pagerank_pid"] = pair_df["posting_id_index"].map(pagerank_dict) 
    pair_df["pagerank_cpid"] = pair_df["candidate_posting_id_index"].map(pagerank_dict) 
    pair_df["pagerank_diff"] = (pair_df["pagerank_pid"] - pair_df["pagerank_cpid"]).abs()
    #pair_df["pagerank_diff"] = (pair_df["pagerank_pid"] / (pair_df["pagerank_cpid"]+eps))
    del pagerank_dict
    _ = gc.collect()

    # strength: weights=weights
    print(" strength")
    strength_dict = dict(zip(posting_ids_index, np.array(graph.strength(weights=weights))))
    pair_df["strength_pid"] = pair_df["posting_id_index"].map(strength_dict) 
    pair_df["strength_cpid"] = pair_df["candidate_posting_id_index"].map(strength_dict) 
    pair_df["strength_diff"] = (pair_df["strength_pid"] - pair_df["strength_cpid"]).abs()
    #pair_df["strength_diff"] = (pair_df["strength_pid"] / (pair_df["strength_cpid"]+eps))
    del strength_dict
    _ = gc.collect()
    
    # transitivity_local_undirected(clustering coefficient)
    print(" transitivity_local_undirected(clustering coefficient)")
    transitivity_local_undirected_dict = dict(zip(posting_ids_index, np.array(graph.transitivity_local_undirected())))
    pair_df["transitivity_local_undirected_pid"] = pair_df["posting_id_index"].map(transitivity_local_undirected_dict) 
    pair_df["transitivity_local_undirected_cpid"] = pair_df["candidate_posting_id_index"].map(transitivity_local_undirected_dict) 
    pair_df["transitivity_local_undirected_diff"] = (pair_df["transitivity_local_undirected_pid"] - pair_df["transitivity_local_undirected_cpid"]).abs()
    #pair_df["transitivity_local_undirected_diff"] = (pair_df["transitivity_local_undirected_pid"] /(pair_df["transitivity_local_undirected_cpid"]+eps))
    del transitivity_local_undirected_dict
    _ = gc.collect()

    # 近傍のsetを作成  
    print(" making neighbors set")
    pid_neighbors_dict = {}
    for pid in pair_df.posting_id_index.unique():
        neighbors = set(graph.neighbors(pid))
        pid_neighbors_dict[pid] = neighbors
        
    pair_df["posting_neighbors_set"] = pair_df["posting_id_index"].map(pid_neighbors_dict)
    pair_df["candidate_neighbors_set"] = pair_df["candidate_posting_id_index"].map(pid_neighbors_dict)
    del pid_neighbors_dict
    _ = gc.collect()
 
    # intersection
    print(" intersection")
    def intersection_neighbors_num(row):
        return len( row.posting_neighbors_set & row.candidate_neighbors_set  )
    pair_df["intersection_neighbors_num"] = pair_df[["posting_neighbors_set","candidate_neighbors_set"]].apply(intersection_neighbors_num, axis=1)
    
    # union
    print(" union")
    pair_df["union_neighbors_num"] = pair_df["neighborhood_size1_pid"] +  pair_df["neighborhood_size1_cpid"] - pair_df["intersection_neighbors_num"]
    pair_df["intersection_neighbors_rate"] = pair_df["intersection_neighbors_num"]/(pair_df["union_neighbors_num"]+eps)
    
    # xor
    print(" xor")
    pair_df["xor_neighbors_num"] = pair_df["union_neighbors_num"] - pair_df["intersection_neighbors_num"]
    # cleaning
    pair_df = pair_df.drop(["posting_id_index","candidate_posting_id_index","posting_neighbors_set","candidate_neighbors_set"], axis=1)
    
    return pair_df






