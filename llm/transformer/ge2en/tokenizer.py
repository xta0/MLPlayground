from collections import Counter

import spacy


class TextTokenizer:
    PAD = 0
    UNK = 1
    MAX_VOCAB_SIZE = 50000

    def __init__(self, language):
        self.language = language
        self.tokenizer = self._init_tokenizer(language)
        self.vocab = {}

    def _init_tokenizer(self, language):
        if language == "de":
            model_name = "de_core_news_sm"
        elif language == "en":
            model_name = "en_core_web_sm"
        else:
            raise ValueError(f"Unsupported language: {language}")

        try:
            return spacy.load(model_name)
        except OSError:
            return spacy.blank(language)

    def build_dictionary(self, training_set):
        word_count = Counter()

        for sentence in training_set:
            tokens = ["BOS"] + [tok.text for tok in self.tokenizer(sentence)] + ["EOS"]
            word_count.update(tokens)

        # sort the dictionary by word frequency
        frequency = word_count.most_common(self.MAX_VOCAB_SIZE)
        self.vocab = {word: idx + 2 for idx, (word, count) in enumerate(frequency)}
        self.vocab["PAD"] = self.PAD
        self.vocab["UNK"] = self.UNK
        return self.vocab

    def tokenize(self, text):
        tokens = self.tokenizer(text)
        return [self.vocab.get(token.text, self.vocab["UNK"]) for token in tokens]

    def detokenize(self, tokens):
        idx2word = {v: k for k, v in self.vocab.items()}
        words = [idx2word.get(token, "<UNK>") for token in tokens]
        return " ".join(words)
