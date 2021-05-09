import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


def make_dictionaries():
    indo_to_eng_dict = {}
    eng_to_indo_dict = {}
    with open('../input/offline-translator-indonesean-to-english-reverse/indonesean_english_dict.txt') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if "\"" in line:
                print("ignoring: ", line) # skip non-sense line
                continue
            inputs = line[2:-3].split("', '")
            indo_word = inputs[0].lower()
            eng_word = inputs[1].lower()

            indo_to_eng_dict[indo_word] = eng_word
            eng_to_indo_dict[eng_word] = indo_word
    f.close()
    return indo_to_eng_dict, eng_to_indo_dict

INDO_TO_ENG_DICT, ENG_TO_INDO_DICT = make_dictionaries()

def translate_indo_to_eng(text):
    words = text.lower().split()
    translated_words = list(
        map(
            lambda x : x if x not in INDO_TO_ENG_DICT else INDO_TO_ENG_DICT[x], 
            words
        )
    )
    return " ".join(translated_words)

def translate_eng_to_indo(text):
    words = text.lower().split()
    translated_words = list(
        map(
            lambda x : x if x not in ENG_TO_INDO_DICT else ENG_TO_INDO_DICT[x],
            words
        ))
    return " ".join(translated_words)



