from llm.attention.multihead_attention_slow import MultiHeadAttention
import torch
import torch.nn as nn

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # output_size = (image_size - patch_size) / stride + 1
        self.conv2d = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.conv2d(x)  # (1, 3, 32, 32) -> (1, 48, 8, 8) 
        x = x.flatten(2)  # start flattening at third dimension: (1, 48, 8, 8) -> (1, 48, 8*8)
        x = x.transpose(1, 2)  # (1, 48, 64) -> (1, 64, 48)
        return x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.patch_embeddings = PatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # class token
        # add the CLS token to the beginning of the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size)) 
        # use a learnable position embedding for each patch plus the class token
        # (1, 65, 48)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, self.hidden_size))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embeddings(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (1, 64, 48) -> (1, 65, 48)
        x = x + self.position_embeddings # (1, 65, 48) -> (1, 65, 48)
        return x

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, head_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size # 4 head -> hidden_size / 4 = 48 / 4 = 12
        self.query = nn.Linear(hidden_size, head_size, bias=bias)
        self.key = nn.Linear(hidden_size, head_size, bias=bias)
        self.value = nn.Linear(hidden_size, head_size, bias=bias)

    def forward(self, x):
        Q = self.query(x) # (1, 65, 12) -> (1, 65, 12), Wq -> 12x12
        K = self.key(x) # (1, 65, 12) -> (1, 65, 12), Wk -> 12x12
        V = self.value(x) # (1, 65, 12) -> (1, 65, 12), Wv -> 12x12
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5) # (1, 65, 12) x (1, 12, 65) -> (1, 65, 65)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_probs, V) # (1, 65, 65) x (1, 65, 12) -> (1, 65, 12)
        return (out, attention_probs)

    class MultiHeadAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hidden_size = config["hidden_size"]
            self.num_heads = config["num_attention_heads"]
            self.qkv_bias = config.get("qkv_bias", True)
            self.dropout = nn.Dropout(config.get("attention_probs_dropout_prob", 0.1))
            
            self.head_size = self.hidden_size // self.num_heads
            self.heads = []
            for _ in range(self.num_heads):
                head = AttentionHead(self.hidden_size, self.head_size, bias=self.qkv_bias)
                self.heads.append(head)
            self.output_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)


        def forward(self, x):
            out = []
            attention_probs = []
            for head in self.heads:
                head_out, head_attention_probs = head(x)
                out.append(head_out)
                attention_probs.append(head_attention_probs)
            out = torch.cat(out, dim=-1) # (1, 65, 12*4) -> (1, 65, 48)
            """
            Mix and re-combine the heads’ features
            Technically, you could omit this layer; 
            the tensor shape would still work. But then the heads’ outputs are only concatenated, 
            and that attention layer has no learned step to mix their channels. 
            The projection is part of the standard multi-head attention design.
            """
            out = self.output_projection(out)
            return (out, attention_probs)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size) # 48 -> 192
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size) # 192 -> 48
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.norm1 = nn.LayerNorm(config["hidden_size"])
        self.norm2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        x = self.norm1(x)
        attn_out, attn_probs = self.attention(x)
        x = x + attn_out # Residual connection after attention layer
        x = self.norm2(x)
        x = x + self.mlp(x) # Residual connection after feed-forward layer
        return x, attn_probs

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config["num_hidden_layers"]
        self.blocks = nn.ModuleList([])
        for _ in range(self.num_layers):
            block = TransformerBlock(config)
            self.blocks.append(block)

    def forward(self, x):
        all_attn_probs = []
        for block in self.blocks:
            x, attn_probs = block(x)
            all_attn_probs.append(attn_probs)
        return x, all_attn_probs


class ViTForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.embedding(x)
        x, all_attn_probs = self.encoder(x)
        # Use the representation of the [CLS] token for classification
        cls_token = x[:, 0, :] # (1, 65, 48) → (1, 48)
        logits = self.classifier(cls_token)
        return logits, all_attn_probs

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
        elif isinstance(module, TransformerBlock):
            module.attention.query.weight.data = nn.init.trunc_normal_(
                module.attention.query.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.attention.query.weight.dtype)
            module.attention.key.weight.data = nn.init.trunc_normal_(
                module.attention.key.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.attention.key.weight.dtype)
            module.attention.value.weight.data = nn.init.trunc_normal_(
                module.attention.value.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.attention.value.weight.dtype)
            module.attention.out.weight.data = nn.init.trunc_normal_(
                module.attention.out.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.attention.out.weight.dtype)
            module.attention.value.weight.data = nn.init.trunc_normal_(
                module.attention.value.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.attention.value.weight.dtype)
            module.attention.out.weight.data = nn.init.trunc_normal_(
                module.attention.out.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.attention.out.weight.dtype)