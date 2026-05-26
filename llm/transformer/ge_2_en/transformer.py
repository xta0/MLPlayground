import os
import tarfile
import requests

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


def main():
     # init tokenizers
    tokenizer_de = TextTokenizer("de")
    tokenizer_en = TextTokenizer("en")

    train_de, train_en = fetchTrainingData()
    
    # bulid vocab
    tokenizer_de.build_dictionary(train_de)
    tokenizer_en.build_dictionary(train_en)

    test_de = tokenizer_de.tokenize(train_de[0])
    test_en = tokenizer_en.tokenize(train_en[0])

    print(test_de)
    print(test_en)
    
    detokenized_de = tokenizer_de.detokenize(test_de)
    detokenized_en = tokenizer_en.detokenize(test_en)

    print(detokenized_de)
    print(detokenized_en)


if __name__ == "__main__":
    main()
