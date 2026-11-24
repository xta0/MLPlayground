import torch
from torch import nn
from image2text.utils import extract_patches, SinusoidalPosEmb
from image2text.config import Config

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=4, masking=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads, batch_first=True, dropout=0.1)
        self.masking = masking # True for decoder attention masking

    def forward(self, x, kv_in, key_mask=None):
        # x: [batch_size, seq_len, hidden_size]
        if self.masking:
            _, seq_len, _ = x.size()
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        else:
            mask = None
        attn_mask = mask
        """
        x supplies the queries, and kv_in supplies both keys and values.
        MultiheadAttention learns separate projections for Q, K, and V,
        so the same kv_in generally produces different key and value vectors.
        For self-attention, pass x as kv_in. For cross-attention, pass another
        sequence as kv_in, such as the encoder output.
        """
        attn_output, _ = self.attention(x, kv_in, kv_in, key_padding_mask=key_mask, attn_mask=attn_mask)
        return attn_output

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=4, masking=True, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = AttentionBlock(hidden_size, 
                                    num_attention_heads, 
                                    masking=masking)
        
        if self.decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.attn2 = AttentionBlock(hidden_size, 
                                        num_attention_heads, 
                                        masking=False)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, kv_cross=None, input_key_mask=None, cross_key_mask=None):
        """
        x:              [B, query_len, hidden_size]
        kv_cross:       [B, source_len, hidden_size]  # required in decoder mode
        input_key_mask: [B, query_len]
        cross_key_mask: [B, source_len]
        """
        x = self.attn1(x, x, key_mask=input_key_mask) + x
        x = self.norm1(x)
        if self.decoder:
            x = self.attn2(x, kv_cross, key_mask=cross_key_mask) + x
            x = self.norm2(x)
        x = self.mlp(x) + x
        return self.mlp_norm(x)

class VisionEncoder(nn.Module):
    def __init__(self, 
                 image_size,
                 input_channels=3, 
                 hidden_size=128, 
                 patch_size=16, 
                 num_transformer_blocks=6, 
                 num_attention_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * patch_size * input_channels, hidden_size)
        self.seq_len = (image_size // patch_size) ** 2 # num of patches

        # for image inputs, we use a learned positional embedding for each patch
        self.pos_embeddings = nn.Parameter(torch.empty(1, self.seq_len, hidden_size).normal_(mean=0.0, std=0.02))

        # transformer blocks for the encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_attention_heads, masking=False, decoder=False) for _ in range(num_transformer_blocks)
        ])

    def forward(self, x):
        # x: [batch_size, input_channels, height, width]
        bs = x.shape[0]
        # convert images into a sequence of batches
        patches = extract_patches(x, patch_size=self.patch_size) # [1, 64, 768]
        # project each patch to the hidden size
        patch_emb = self.projection(patches) # [1, 64, 128]
        # add positional embeddings to the patch embeddings
        embs = patch_emb + self.pos_embeddings

        for block in self.blocks:
            embs = block(embs)
        return embs


class VisionDecoder(nn.Module):
    def __init__(self, 
                 num_emb = 256,
                 hidden_size=128, 
                 num_transformer_blocks=6, 
                 num_attention_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_emb, hidden_size)  # assuming a vocabulary size of 256 for image tokens
        self.embedding.weight.data = 0.001 * self.embedding.weight.data
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_attention_heads, decoder=True) for _ in range(num_transformer_blocks)
        ])
        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, x, kv_cross, input_key_mask=None, cross_key_mask=None):
        # x: [B, seq_len] (token ids)
        x = self.embedding(x)  # [B, seq_len, hidden_size]
        # Create position indices 0 through seq_len - 1 on the same device as x.
        # [0, 1, 2, 3, ..., seq_len - 1]
        seq_index = torch.arange(x.size(1), device=x.device)
        # Compute one sinusoidal vector per position and share it across the batch.
        pos_emb = self.pos_emb(seq_index).unsqueeze(0)
        x = x + pos_emb
        for block in self.blocks:
            x = block(x, kv_cross=kv_cross, input_key_mask=input_key_mask, cross_key_mask=cross_key_mask)
        x = self.fc_out(x)
        return x

class ImageCaptioner(nn.Module):
    def __init__(self, config: Config, num_emb: int):
        self.config = config
        image_size = config.image_size
        patch_size = config.patch_size
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_encoder_transformer_blocks = config.encoder_transformer_blocks
        num_decoder_transformer_blocks = config.decoder_transformer_blocks
        channels_in = config.channels
        super().__init__()
        self.vision_encoder = VisionEncoder(image_size=image_size, 
                                            input_channels=channels_in, 
                                            hidden_size=hidden_size, 
                                            patch_size=patch_size, 
                                            num_transformer_blocks=num_encoder_transformer_blocks, 
                                            num_attention_heads=num_attention_heads)
        self.vision_decoder = VisionDecoder(num_emb=num_emb, 
                                            hidden_size=hidden_size, 
                                            num_transformer_blocks=num_decoder_transformer_blocks, 
                                            num_attention_heads=num_attention_heads)

    def forward(self, image, target_seq, atten_mask=None):
        # MultiheadAttention uses True for keys to ignore.
        padding_mask = None if atten_mask is None else atten_mask == 0
        encoder_seq = self.vision_encoder(image)
        decoder_seq = self.vision_decoder(target_seq, 
                                          kv_cross=encoder_seq, 
                                          input_key_mask=padding_mask)
        return decoder_seq
