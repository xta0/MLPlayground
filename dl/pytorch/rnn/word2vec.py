import utils

# read in the extracted text file      
with open('/Users/taox/Downloads/text8') as f:
    text = f.read()

# print out the first 100 characters
# print(text[:100])

# get list of words
words = utils.preprocess(text)
# print(words[:30])

# print some stats about this word data
print("Total words in text: {}".format(len(words)))
print("Unique words: {}".format(len(set(words)))) # `set` removes any duplicate words

vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]

print(int_words[:30])

from collections import Counter
import random
import numpy as np

threshold = 1e-5
word_counts = Counter(int_words)
#print(list(word_counts.items())[0])  # dictionary of int_words, how many times they appear

total_count = len(int_words)
freqs = {word: count/total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
# discard some frequent words, according to the subsampling equation
# create a new list of words for training
train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

print("train_words: ", train_words[:30])

def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx+1:stop+1]
    
    return list(target_words)

## testing code
int_text = [i for i in range(10)]
print('Input: ', int_text)
idx=5 # word index of interest

target = get_target(int_text, idx=idx, window_size=5)
print('Target: ', target)  # you should get some indices around the idx
## 

def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words)//batch_size
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

## Testing code        
int_text = train_words[:10]
z = [int_to_vocab[ii] for ii in int_text]
print(z)
x,y = next(get_batches(int_text, batch_size=4, window_size=5))

print('x:\n', int_to_vocab[x[0]])
print('y:\n', int_to_vocab[y[0]])

## define the SkipGram model

import torch
from torch import nn
import torch.optim as optim

# for each input word, predict its context words surrounding it
class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        
        # complete this SkipGram model
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.fc = nn.Linear(n_embed, n_vocab)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
## Validation

def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """
    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities

## Traning 

# Note that, because we applied a softmax function to our model output, we are using NLLLoss as opposed to cross entropy. This is because Softmax in combination with NLLLoss = CrossEntropy loss .
# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_dim=300 # you can change, if you want

model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 500
steps = 0
epochs = 5

# train for some number of epochs
# for e in range(epochs):
    
#     # get input and target batches
#     for inputs, targets in get_batches(train_words, 512):
#         steps += 1
#         inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
#         inputs, targets = inputs.to(device), targets.to(device)
        
#         log_ps = model(inputs)
#         loss = criterion(log_ps, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if steps % print_every == 0:                  
#             # getting examples and similarities      
#             valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
#             _, closest_idxs = valid_similarities.topk(6) # topk highest similarities
            
#             valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
#             for ii, valid_idx in enumerate(valid_examples):
#                 closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
#                 print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
#             print("...")

## Negative Sampling
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # Initialize both embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        
    def forward_input(self, input_words):
        # return input vector embeddings
        return self.in_embed(input_words)
    
    def forward_output(self, output_words):
        # return output vector embeddings

        return None
    
    def forward_noise(self, batch_size, n_samples):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)
        
        device = "cuda" if model.out_embed.weight.is_cuda else "cpu"
        noise_words = noise_words.to(device)
        
        ## TODO: get the noise embeddings
        # reshape the embeddings so that they have dims (batch_size, n_samples, n_embed)

        
        return None