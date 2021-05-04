import numpy as np
import pandas as pd
import shopee

def find_optimal_threshold(df:pd.DataFrame, pair_df:pd.DataFrame) -> float:
    def f(th):
        pair_df["prediction"] = (pair_df["confidence"] > th).astype(int)
        return shopee.metrics.calc_f1_score(df, pair_df)
        
    lb, ub = 0, 0.5
    for k in range(10):
        th1 = (lb * 2 + ub) / 3
        th2 = (lb + ub * 2) / 3
        if f(th1) < f(th2):
            lb = th1
        else:
            ub = th2
    th = (lb + ub) / 2
    return f(th), th
