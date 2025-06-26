import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        mask = mask[:, None, :]
        
        # print('x: ', x.shape)
        # print('mask:', mask.shape)
        
        attention = self.attention(x, x, x)[0] # , attn_mask=mask)[0]
        x = self.dropout1(attention) + x
        x = self.norm1(x)
        
        forward = self.feed_forward(x)
        x = self.dropout2(forward) + x
        x = self.norm2(x)
        
        return x

class Transformer_Decoder(nn.Module):
    def __init__(self, d_model, num_layers, heads, dropout, max_length):
        super(Transformer_Decoder, self).__init__()
        
        d_ff = d_model * 2
        
        self.embed_size = d_model
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, heads, dropout, d_ff) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs_embeds, attention_mask):
        B, L, C = inputs_embeds.shape
        
        positions = torch.arange(0, L).expand(B, L).to(inputs_embeds.device)
        
        hidden_states = self.dropout(inputs_embeds + self.position_embedding(positions))
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        return hidden_states