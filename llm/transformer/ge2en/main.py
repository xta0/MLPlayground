import os
import tarfile
import requests

import numpy as np

from tokenizer import TextTokenizer

DATA_DIR = "data"
TRAINING_ARCHIVE = os.path.join(DATA_DIR, "training.tar.gz")
TRAINING_FILES = (
    os.path.join(DATA_DIR, "train.de"),
    os.path.join(DATA_DIR, "train.en"),
)

def fetchTrainingData():
    url=("https://raw.githubusercontent.com/neychev/"
         "small_DL_repo/master/datasets/Multi30k/training.tar.gz")    #A
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TRAINING_ARCHIVE):    #B
        fb1=requests.get(url)
        fb1.raise_for_status()
        with open(TRAINING_ARCHIVE,"wb") as f:
            f.write(fb1.content)
    if not all(os.path.exists(path) for path in TRAINING_FILES):
        with tarfile.open(TRAINING_ARCHIVE) as train:    #C
            train.extractall(DATA_DIR)    #D

    with open(TRAINING_FILES[0], 'rb') as fb:
        trainde = fb.readlines()
    with open(TRAINING_FILES[1], 'rb') as fb:
        trainen = fb.readlines()
    trainde=[i.decode("utf-8").strip() for i in trainde] 
    trainen=[i.decode("utf-8").strip() for i in trainen] 
    print("Sample training data (German):")
    print(trainde[:1])
    print("Sample training data (English):")
    print(trainen[:1])
    return trainde, trainen


def prepare_batch_training_data(de_sentences, en_sentences, tokenizer_de,
                                tokenizer_en, batch_size=128):
    # 1. Sort sentences by length first.
    # 2. Consecutive rows now have similar lengths.
    # 3. Create batches from consecutive rows.
    # 4. Shuffle the batch order.
    out_de_ids = [tokenizer_de.tokenize(sentence)
                  for sentence in de_sentences]
    out_en_ids = [tokenizer_en.tokenize(sentence)
                  for sentence in en_sentences]

    sorted_ids = sorted(range(len(out_de_ids)),
                        key=lambda idx: len(out_de_ids[idx]))
    out_de_ids = [out_de_ids[idx] for idx in sorted_ids]
    out_en_ids = [out_en_ids[idx] for idx in sorted_ids]

    idx_list = np.arange(0, len(out_de_ids), batch_size)
    np.random.shuffle(idx_list)

    batch_indexes = []
    for idx in idx_list:
        batch_indexes.append(np.arange(idx, min(len(out_de_ids),
                                                idx + batch_size)))

    return out_de_ids, out_en_ids, batch_indexes


def main():
     # init tokenizers
    tokenizer_de = TextTokenizer("de")
    tokenizer_en = TextTokenizer("en")

    train_de, train_en = fetchTrainingData()
    
    # bulid vocab
    tokenizer_de.build_dictionary(train_de)
    tokenizer_en.build_dictionary(train_en)

   
    train_de_ids, train_en_ids, batch_indexes = prepare_batch_training_data(
        train_de, train_en, tokenizer_de, tokenizer_en
    )
    print(batch_indexes[0])



if __name__ == "__main__":
    main()
