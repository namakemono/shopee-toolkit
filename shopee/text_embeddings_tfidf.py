import re
import unidecode
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .translation import translate_indo_to_eng, translate_eng_to_indo

def get_text_embeddings(df:pd.DataFrame, use_translate:bool=True) -> np.ndarray:
    texts = df["title"].apply(clean_text)
    if use_translate:
        print("translate indo to eng.")
        texts = texts.apply(translate_indo_to_eng) # 性能が+0.0001改善?
        # texts = texts.apply(translate_eng_to_indo) # 性能悪化する
    embeddings = TfidfVectorizer(
        stop_words='english',
        binary=True,
        max_features=25000
    ).fit_transform(texts)
    print("vocab size: %d" % embeddings.shape[1])
    return embeddings

# エスケープバイト
def string_escape(s, encoding='utf-8'): # https://www.kaggle.com/c/shopee-product-matching/discussion/233605
    return (
        s.encode('latin1')  # To bytes, required by 'unicode-escape'
        .decode('unicode-escape')  # Perform the actual octal-escaping decode
        .encode('latin1')  # 1:1 mapping back to bytes
        .decode(encoding)
    )  # Decode original encoding

# 絵文字を空白に変更
def remove_emoji(text): 
    emoji_list = ['©️', '®', '‼', '‼️', '™', '⏩', '⏪', '▫', '☀', '☂️', '☃', '☑️',\
                  '☪', '☮', '♑', '♥', '♥️', '♨', '♻️', '⚠️', '⚡',\
                 '⚫', '⛔', '⛩️', '✨', '❣️', '✅', '✔', '✔️', '❄', \
                 '❌', '❗', '❤', '❤️', '⭐', '🆕', '🇮🇩', '🇯🇵', '🇰🇷', '🇲🇨',\
                 '🌱', '👍', '💋', '💕', '💬', '📣', '🔥',]
    for emoji_chr in emoji_list:
        text = text.replace(emoji_chr, ' ')
    return text

def clean_text(text:str) -> str:
    # エスケープバイト
    text = string_escape(text)
    
    # 小文字変換
    text = text.lower()
    
    # b"title" を変換
    result = re.match(r'^b"(.*)"$', text)
    if result: #none以外の場合(マッチした場合)
        text = result.group(1)
        
    # アクセント記号などを削除
    #text = unidecode.unidecode(text) 削除しないほうが良い
    
    text = " " + text +  " " # 末尾や先頭に単位やら何やらがくることが多いので追加しておく
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('【', ' ')
    text = text.replace('】', ' ')
    
    # 単位の変換
    ## gram
    text = re.sub("(\d+) (gram) ", r"\1gram ", text)
    text = re.sub("(\d+) (gr) ", r"\1gram ", text)
    text = re.sub("(\d+)(gr) ", r"\1gram ", text)
    text = re.sub("(\d+) (g) ", r"\1gram ", text)
    text = re.sub("(\d+)(g) ", r"\1gram ", text)
    ## kg
    text = re.sub("(\d+) (kg) ", r"\1kg ", text)
    ## ampere
    text = re.sub("(\d+) (ampere) ", r"\1ampere ", text)
    text = re.sub("(\d+) (a) ", r"\1ampere ", text)
    text = re.sub("(\d+)(a) ", r"\1ampere ", text)
    ## volt
    text = re.sub("(\d+) (volt) ", r"\1volt ", text)
    text = re.sub("(\d+) (v) ", r"\1volt ", text)
    text = re.sub("(\d+)(v) ", r"\1volt ", text)
    # inch
    text = re.sub("(\d+) (inch) ", r"\1inch ", text)
    # cm
    text = re.sub("(\d+) (cm) ", r"\1cm ", text)
    # mm
    text = re.sub("(\d+) (mm) ", r"\1mm ", text)
    # ml
    text = re.sub("(\d+) (ml) ", r"\1ml ", text)
    # liter
    text = re.sub("(\d+) (liter) ", r"\1liter ", text)
    text = re.sub("(\d+) (l) ", r"\1liter ", text)
    # meter
    text = re.sub("(\d+) (meter) ", r"\1meter ", text)
    text = re.sub("(\d+) (m) ", r"\1meter ", text)
    text = re.sub("(\d+)(m) ", r"\1meter ", text)
    # tahun
    text = re.sub("(\d+) (tahun) ", r"\1tahun ", text)
    text = re.sub("(\d+) (thn) ", r"\1tahun ", text)
    text = re.sub("(\d+)(thn) ", r"\1tahun ", text)
    # bulan
    text = re.sub("(\d+) (bulan) ", r"\1bulan ", text)

    # 空白分割
    text = remove_emoji(text) # 大体を削除
    text = text.replace('✨', ' ')
    text = text.replace('❣️', ' ')
    text = text.replace('<', ' ')
    text = text.replace('>', ' ')
    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace('_', ' ')
    text = text.replace('|', ' ')
    text = text.replace('/', ' ')
    text = text.replace('\\', ' ')
    text = text.replace(':', ' ')
    text = text.replace('=', ' ')
    text = text.replace('•', ' ')
    text = text.replace(' - ', ' ') # ハイフンは確実に前後が空白のものだけを扱う
    #text = text.replace('-', '')
    
    # 関係ない単語を削除
    text = text.replace(' big sale ', ' ')
    text = text.replace(' bigsale ', ' ')
    text = text.replace(' best seller ', ' ')
    text = text.replace(' ready stock ', ' ')
    text = text.replace(' ready stock ', ' ')
    text = text.replace(' promo flash sale ', ' ')
    text = text.replace(' flash sale ', ' ')
    text = text.replace(' falsh sale ', ' ')
    text = text.replace(' hot sale ', ' ')
    text = text.replace(' super sale ', ' ')
    text = text.replace(' promo sale ', ' ')
    text = text.replace(' murah banget ', ' ') # とても安い
    text = text.replace(' termurah ', ' ') # 最も安い
    text = text.replace(' murah ', ' ') # 安い
     
    # high quality
          
    # terlaris: ベストセラー
    #text = text.replace(' terlaris ', ' ') # 安い
    
    # 支払い COD: その場で支払うことを表すっぽい
    # bisa bayar di , cod bayar di
    text = text.replace(' bisa bayar di tempata ', ' ')  # その場で支払えます
    text = text.replace(' bayar di tempat ', ' ')  # その場で支払う
    text = text.replace(' bayar di ', ' ') 
    text = text.replace(' cod ', ' ') 
    text = text.replace(' bisa ', ' ') 

    return text






