import torch
import torch.nn as nn
import torch.nn.functional as F

EmbeddingDims = 256

class SelfAttention(nn.Module): 
                            
    def __init__(self, d_model=EmbeddingDims,  
                 row_dim=0, 
                 col_dim=1):
        ## d_model = the number of embedding values per token.
        ##           Because we want to be able to do the math by hand, we've
        ##           the default value for d_model=2.
        ##           However, in "Attention Is All You Need" d_model=512
        ##
        ## row_dim, col_dim = the indices we should use to access rows or columns
        super().__init__()
        
        ## Initialize the Weights (W) that we'll use to create the
        ## query (q), key (k) and value (v) for each token
        self.W_q = nn.Linear(in_features=EmbeddingDims, out_features=128, bias=False)
        self.W_k = nn.Linear(in_features=EmbeddingDims, out_features=128, bias=False)
        self.W_v = nn.Linear(in_features=EmbeddingDims, out_features=EmbeddingDims, bias=False)
        
        self.row_dim = row_dim
        self.col_dim = col_dim

        
    def forward(self, token_encodings):
        ## token_encodings: word_embedding + positional encoding
        ## Create the query, key and values using the encoding numbers
        ## associated with each token (token encodings)
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        ## Compute similarities scores: (q * k^T)
        ## transpose swap the two dimensions: dim0, dim1 = dim1, dim0
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        print(sims.shape)

        ## Scale the similarities by dividing by sqrt(k.col_dim)
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        ## Apply softmax to determine what percent of each tokens' value to
        ## use in the final attention values.
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        print(attention_percents.shape)

        ## Scale the values by their associated percentages and add them up.
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

def main():
    # 8 words, 256 embedding values per word
    encodings_matrix = torch.randn(8,EmbeddingDims)
    print("Encoding Matrix:", encodings_matrix.shape)

    torch.manual_seed(42)
    selfAttention = SelfAttention(d_model=EmbeddingDims)
    attention_scores = selfAttention(encodings_matrix)
    print("Attention Scores:", attention_scores.shape)


if __name__ == "__main__":
    main()