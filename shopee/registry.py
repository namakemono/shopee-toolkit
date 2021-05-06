def get_value(key):
    return D[key]

# 主に外部データ情報を一意に定まるように書く
D = {
    # EfficientNet B3 w/ ArcFace image size 256x256, 外部データで学習
    "effnet_b3_arcface_256x256": "../input/model-efficientnet-b3-256x256-arcface-previous/model_efficientnet_b3_IMG_SIZE_256_arcface_previous.bin",
    # EfficientNet B3(無印)の重み
    "effnet_b3_weights_keras": "../input/effnet-b3/effnet-b3.h5",
    # EfficientNet B3 w/ SwAVの重み
    # "effnet_b3_swav_weights_keras": "../input/swav-b3-previous-epoch10-lr003/SwAV_B3_previous_epoch10_lr0.03.h5",
    "effnet_b3_swav_weights_keras": "../input/swav-b3-previous-epoch20-lr01/SwAV_B3_previous_epoch20_lr0.1.h5",

    # 画像特徴量(EfficientNet B3で抽出)
    "effnet_b3_image_embeddings": "../input/tmp-shopee-image-embeddings-effnet-b3/image_embeddings_train.npy",
}

