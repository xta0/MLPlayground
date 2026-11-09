import json
from collections import Counter
from os import path
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "flicker8k"
FLICKER_8K = DATA_DIR / "dataset_flickr8k.json"

"""
{
  "dataset": "flickr8k",
  "images": [
    {
      "sentids": [0, 1, 2, 3, 4],
      "imgid": 0,
      "filename": "2513260012_03d33305cf.jpg",
      "split": "train",
      "sentences": [
        {
          "tokens": [
            "a", "black", "dog", "is", "running", "after",
            "a", "white", "dog", "in", "the", "snow"
          ],
          "raw": "A black dog is running after a white dog in the snow .",
          "imgid": 0,
          "sentid": 0
        }
      ]
    },
       {
      "sentids": [5, 6, 7, 8, 9],
      "imgid": 1,
      "filename": "2903617548_d3e38d7f88.jpg",
      "split": "train",
      "sentences": [
        {
          "tokens": ["a", "little", "baby", "plays", "croquet"],
          "raw": "A little baby plays croquet .",
          "imgid": 1,
          "sentid": 5
        }
      ]
    }
  ]
}

"""

START_IDX = 1
END_IDX = 2
PAD_IDX = 0
UNK_IDX = 3
START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

MAX_CAPTION_LENGTH = 50

def prepare_data(file_path=FLICKER_8K):
    with open(file_path, "r") as f:
        data = json.load(f)
    train_image_paths = []
    train_image_captions = []
    test_image_paths = []
    test_image_captions = []

    word_freq = Counter()

    for image in data["images"]:
        caption_tokens = []
        for sentence in image["sentences"]:
            tokens = sentence["tokens"]
            word_freq.update(tokens)
            if len(tokens) <= MAX_CAPTION_LENGTH:
                caption_tokens.append(tokens)
        if len(caption_tokens) == 0:
            continue

        image_name = image["filename"]
        image_path = str(DATA_DIR / "Images" / image_name)
        split = image["split"]

        if split in {"train", "val", "restval"}:
            train_image_paths.append(image_path)
            train_image_captions.append(caption_tokens)
        elif split in {"test"}:
            test_image_paths.append(image_path)
            test_image_captions.append(caption_tokens)

    words = [w for w in word_freq.keys() if word_freq[w] > 0]
    vocab = {w: i + 4 for i, w in enumerate(words)}
    vocab[PAD_TOKEN] = PAD_IDX
    vocab[START_TOKEN] = START_IDX
    vocab[END_TOKEN] = END_IDX
    vocab[UNK_TOKEN] = UNK_IDX
    return train_image_paths, train_image_captions, test_image_paths, test_image_captions, vocab


class Flickr(Dataset):
    def __init__(self, image_paths, image_captions, vocab):
        self.image_paths = image_paths
        self.image_captions = image_captions
        self.vocab = vocab
        self.max_caption_length = MAX_CAPTION_LENGTH
        self.image_size = 128
        self.image_transform = self._construct_image_transform(self.image_size)
        self.data = self._group_image_captions_pairs()
        self.data_size = len(self.data)

    def load_and_preprocess_single_image(self, image_path):
        img_raw = Image.open(image_path).convert("RGB")
        img_resized = self._center_crop(img_raw).resize((self.image_size, self.image_size))
        img_tensor = self.image_transform(img_resized)
        return img_tensor

    def load_and_preprocess_images(self):
        return [self.load_and_preprocess_single_image(image_path) for image_path in self.image_paths]

    def word_to_index(self, word):
        if self.vocab is None:
            raise ValueError("Vocabulary has not been initialized.")
        return self.vocab.get(word, self.vocab[UNK_TOKEN])

    def index_to_word(self, index):
        if self.vocab is None:
            raise ValueError("Vocabulary has not been initialized.")
        for word, idx in self.vocab.items():
            if idx == index:
                return word
        return UNK_TOKEN


    def _construct_image_transform(self, image_size):
        # ImageNet normalization statistics
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        preprocessing = transforms.Compose([         
            transforms.ToTensor(),
            normalize,])
        return preprocessing

    def _center_crop(self, img):   
        width, height = img.size   
        new=min(width,height)
        left = (width - new)/2
        top = (height - new)/2
        right = (width + new)/2
        bottom = (height + new)/2
        im = img.crop((left, top, right, bottom))
        return im

    def _group_image_captions_pairs(self):
        grouped = []
        for image_file_path, captions in zip(self.image_paths, self.image_captions):
            for caption in captions:
                grouped.append((image_file_path, caption))
        return grouped

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
       image_file_path, caption_tokens = self.data[idx]
       image_tensor = self.load_and_preprocess_single_image(image_file_path)
        # Pad the token and label sequences
       caption_tokens = caption_tokens[:self.max_caption_length - 1]
       caption_tokens = [token.strip().lower() for token in caption_tokens]
       caption_tokens = [START_TOKEN] + caption_tokens + [END_TOKEN]
        # Extract input and target output
       input_tokens = caption_tokens[:-1].copy()
       tgt_tokens = caption_tokens[1:].copy()

       sample_length = len(input_tokens)
       padding_length = self.max_caption_length - sample_length
       input_tokens += [PAD_TOKEN] * padding_length
       tgt_tokens += [PAD_TOKEN] * padding_length

       input_token_ids = [self.word_to_index(w) for w in input_tokens]
       tgt_token_ids = [self.word_to_index(w) for w in tgt_tokens]

       input_token_id_tensors = torch.tensor(input_token_ids, dtype=torch.long)
       tgt_token_id_tensors = torch.tensor(tgt_token_ids, dtype=torch.long)

       input_attn_mask = torch.zeros([self.max_caption_length], dtype=torch.long)
       input_attn_mask[:sample_length] = 1
       input_attn_mask = input_attn_mask.bool()
       return image_tensor, input_token_id_tensors, tgt_token_id_tensors, input_attn_mask