import os
import tarfile

import requests
import spacy
from collections import Counter

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

def initTokenizer(language):
    if language == "de":
        model_name = "de_core_news_sm"
    elif language == "en":
        model_name = "en_core_web_sm"
    else:
        raise ValueError(f"Unsupported language: {language}")

    try:
        tokenizer = spacy.load(model_name)
    except OSError:
        tokenizer = spacy.blank(language)
    return tokenizer

def build_dictionary(training_set, language):
    tokens = [["BOS"] + tokenize(sentence, language) + ["EOS"] for sentence in training_set]
    PAD=0
    UNK=1
    word_count=Counter()
    for sentence in tokens:
        for word in sentence:
            word_count[word]+=1
    frequency=word_count.most_common(50000)        
    # total_en_words=len(frequency)+2

    # idx is the order of the word in the dictionary
    en_word_dict={w[0]:idx+2 for idx,w in enumerate(frequency)}
    en_word_dict["PAD"]=PAD
    en_word_dict["UNK"]=UNK

    # another dictionary to map indexes to tokens
    en_idx_dict={v:k for k,v in en_word_dict.items()}
    return en_idx_dict

def tokenize(tokenizer, vocab, text):
    tokens = tokenizer(text)
    return [vocab.get(token.text, vocab["UNK"]) for token in tokens]

def main():
    train_de, train_en = fetchTrainingData()
    
    # bulid vocab
    vocab_de = build_dictionary(train_de, "de")
    vocab_en = build_dictionary(train_en, "en")

    # init tokenizers
    tokenizer_de = initTokenizer("de")
    tokenizer_en = initTokenizer("en")

    test_de = tokenize(train_de[0])
    test_en = tokenize(train_en[0])

    print(tokenize(test_de, tokenizer_de, vocab_de))
    print(tokenize(test_en, tokenizer_en, vocab_en))

if __name__ == "__main__":
    main()
