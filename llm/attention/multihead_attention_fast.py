import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # 
        # (2, 4, 8) -> linear(8,8) -> (2, 4, 8)
        # Naively, we should have two queries: 
        # Q1 = self.query(x) # (2, 4, 4)
        # Q2 = self.query(x) # (2, 4, 4)
        # Those two matrix multiplications are independent.
        # This is simply fusing two GEMMs into one larger GEMM.
        # This is not multi-head parallelization, but it is a step towards it.
        Q = self.query(x)
        # (2, 4, 8) -> (2, 4, 2, 4) # (batch, token, head, dimensions_per_head) -> "Each token contains multiple heads."
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # (2, 4, 2, 4) -> (2, 2, 4, 4) # (batch, head, token, dimensions_per_head) -> "Each head contains all the token vectors."
        # Each head can independently compute a token-to-token attention matrix.
        # The parallelism comes from torch.matmul, 
        # which interprets all leading dimensions (B and H) (2,4) as 
        # a collection of independent matrix multiplications that can be scheduled concurrently
        Q = Q.transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        print("Q.shape:", Q.shape)
        print("K.shape:", K.shape)
        print("V.shape:", V.shape)

        # this is where the parallelization happens,
        # We compute the attention scores for all heads at once
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        print("scores.shape:", scores.shape)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        print("attention_weights.shape:", attention_weights.shape)

        ### START CODE HERE ###
        attention_output = torch.matmul(attention_weights, V)
        print("attention_output.shape:", attention_output.shape)
        ### END CODE HERE ###

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        print("attention_output.shape after transpose and view:", attention_output.shape)

        output = self.out(attention_output)
        print("output.shape:", output.shape)
        return output

# quick check
mha = MultiheadAttention(embed_dim=8, num_heads=2)
x = torch.randn(2, 4, 8)
print("x.shape:", x.shape, "-> output.shape:", mha(x).shape)