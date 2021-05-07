import tensorflow as tf

class TextEntry:
    def __init__(self, id:str):
        self.id = id
    def to_dict(self):
        return {
            "id":   self.id
        }

class ImageEntry:
    def __init__(self, 
        id:str, 
        classname, 
        model_type:str, 
        weights_filepath:str, 
        image_size:int, 
        preprocess_input
    ):
        self.id = id
        self.classname = classname
        self.model_type = model_type
        self.weights_filepath = weights_filepath
        self.image_size = image_size
        self.train_embeddings_filepath = f"./data/train-embeddings-{self.id}.npy"
        self.preprocess_input = preprocess_input

    def to_dict(self):
        return {
            "id":                           self.id,
            "classname":                    str(self.classname),
            "model_type":                   self.model_type,
            "weights_filepath":             self.weights_filepath,
            "train_embeddings_filepath":    self.train_embeddings_filepath,
            "image_size":                   self.image_size
        }

def get_entries():
    return [
        ImageEntry(
            id                  = "effnet-b0_512x512",
            classname           = tf.keras.applications.EfficientNetB0,
            model_type          = "keras-origin",
            weights_filepath    = "../input/all-in-one-packages/models/effnet-b0.h5",
            image_size          = 512,
            preprocess_input    = tf.keras.applications.efficientnet.preprocess_input
        ),
         ImageEntry(
            id                  = "effnet-b3_512x512",
            classname           = tf.keras.applications.EfficientNetB3,
            model_type          = "keras-origin",
            weights_filepath    = "../input/all-in-one-packages/models/effnet-b3.h5",
            image_size          = 512,
            preprocess_input    = tf.keras.applications.efficientnet.preprocess_input
        ),
        ImageEntry(
            id                  = "effnet-b5_512x512",
            classname           = tf.keras.applications.EfficientNetB5,
            model_type          = "keras-origin",
            weights_filepath    = "../input/all-in-one-packages/models/effnet-b5.h5",
            image_size          = 512,
            preprocess_input    = tf.keras.applications.efficientnet.preprocess_input
        ),
        ImageEntry(
            id                  = "effnet-b0_256x256",
            classname           = tf.keras.applications.EfficientNetB0,
            model_type          = "keras-origin",
            weights_filepath    = "../input/all-in-one-packages/models/effnet-b0.h5",
            image_size          = 256,
            preprocess_input    = tf.keras.applications.efficientnet.preprocess_input
        ),
        ImageEntry(
            id                  = "effnet-b3_256x256",
            classname           = tf.keras.applications.EfficientNetB3,
            model_type          = "keras-origin",
            weights_filepath    = "../input/all-in-one-packages/models/effnet-b3.h5",
            image_size          = 256,
            preprocess_input    = tf.keras.applications.efficientnet.preprocess_input
        ),
        ImageEntry(
            id                  = "effnet-b5_256x256",
            classname           = tf.keras.applications.EfficientNetB5,
            model_type          = "keras-origin",
            weights_filepath    = "../input/all-in-one-packages/models/effnet-b5.h5",
            image_size          = 256,
            preprocess_input    = tf.keras.applications.efficientnet.preprocess_input
        ),
        ImageEntry(
            id                  = "mobilenet-v2_256x256",
            classname           = tf.keras.applications.MobileNetV2,
            model_type          = "keras-origin",
            weights_filepath    = "../input/all-in-one-packages/models/mobilenet-v2.h5",
            image_size          = 256,
            preprocess_input    = tf.keras.applications.mobilenet_v2.preprocess_input
        ),
        TextEntry(
            id                  = "tfidf-v1",
        )
    ]

def get_entry_by_id(id):
    for entry in get_entries():
        if entry.id == id:
            return entry
    raise ValueError(f"Undefined entry id: {id}")


