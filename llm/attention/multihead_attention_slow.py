import torch ## torch let's us create tensors and also provides helper functions
import torch.nn as nn ## torch.nn gives us nn.module() and nn.Linear()
import torch.nn.functional as F # This gives us the softmax()

class Attention(nn.Module): 
                            
    def __init__(self, d_model=2,  
                 row_dim=0, 
                 col_dim=1):
        
        super().__init__()
        
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        self.row_dim = row_dim
        self.col_dim = col_dim


    ## The only change from SelfAttention and attention is that
    ## now we expect 3 sets of encodings to be passed in...
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        ## ...and we pass those sets of encodings to the various weight matrices.
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
            
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
    
class MultiHeadAttention(nn.Module):

    def __init__(self, 
                 d_model=2,  
                 row_dim=0, 
                 col_dim=1, 
                 num_heads=1):
        
        super().__init__()

        ## create a bunch of attention heads
        self.heads = nn.ModuleList(
            [Attention(d_model, row_dim, col_dim) 
             for _ in range(num_heads)]
        )

        self.col_dim = col_dim
        
    def forward(self, 
                encodings_for_q, 
                encodings_for_k,
                encodings_for_v):

        ## run the data through all of the attention heads
        return torch.cat([head(encodings_for_q, 
                               encodings_for_k,
                               encodings_for_v) 
                          for head in self.heads], dim=self.col_dim)

def test_encoder_decoder_attention():
    ## create matrices of token encodings...
    encodings_for_q = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_k = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_v = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    ## set the seed for the random number generator
    torch.manual_seed(42)

    ## create an attention object
    attention = Attention(d_model=2,
                        row_dim=0,
                        col_dim=1)

    ## calculate encoder-decoder attention
    attention(encodings_for_q, encodings_for_k, encodings_for_v)


def test_multi_head_attention():
    ## create matrices of token encodings...
    encodings_for_q = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_k = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    encodings_for_v = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])
    ## set the seed for the random number generator
    torch.manual_seed(42)

    ## create an attention object
    multiHeadAttention = MultiHeadAttention(d_model=2,
                                            row_dim=0,
                                            col_dim=1,
                                            num_heads=1)

    ## calculate encoder-decoder attention
    multiHeadAttention(encodings_for_q, encodings_for_k, encodings_for_v)

    ## set the seed for the random number generator
    torch.manual_seed(42)

    ## create an attention object
    multiHeadAttention = MultiHeadAttention(d_model=2,
                                            row_dim=0,
                                            col_dim=1,
                                            num_heads=2)

    ## calculate encoder-decoder attention
    multiHeadAttention(encodings_for_q, encodings_for_k, encodings_for_v)

def main():
    test_encoder_decoder_attention()
    test_multi_head_attention()


if __name__ == "__main__":
    main()