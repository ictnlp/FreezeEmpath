import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPoolingLayer(nn.Module):
    def __init__(self, input_dim, num_latent_queries, num_heads):
        super().__init__()
        self.num_latent_queries = num_latent_queries
        self.latent_queries = nn.Parameter(torch.randn(num_latent_queries, input_dim))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        returns: Tensor of shape (batch_size, num_latent_queries, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        queries = self.latent_queries.unsqueeze(1).expand(-1, batch_size, -1)
        key_value = x.permute(1, 0, 2)

        attn_output, _ = self.multihead_attn(queries, key_value, key_value)
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output


class EmotionExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim   = config.speech_encoder_hidden_size
        self.llm_dim       = config.hidden_size
        self.layer_num     = 32

        self.gate_fc1 = nn.Linear(self.encoder_dim, 128)
        self.gate_relu = nn.ReLU()
        self.gate_fc2 = nn.Linear(128, 1)

        self.layernorm = nn.LayerNorm(self.encoder_dim)

        self.attn_pool = AttentionPoolingLayer(
            input_dim=self.encoder_dim,
            num_latent_queries=1,
            num_heads=4
        )

        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)

    def forward(self, x):
        """
        x: Tensor, shape (batch_size, layer_num, seq_len, encoder_dim)
        returns: Tensor, shape (batch_size, llm_dim)
        """
        batch_size, num_layers, seq_len, feat_dim = x.shape
        assert num_layers == self.layer_num

        layer_means = x.mean(dim=2)

        gate_hidden = self.gate_fc1(layer_means)
        gate_hidden = self.gate_relu(gate_hidden)
        gate_scores = self.gate_fc2(gate_hidden).squeeze(-1)
        weights = F.softmax(gate_scores, dim=1)
        weights = weights.view(batch_size, num_layers, 1, 1)
        weighted_sum = (x * weights).sum(dim=1)

        normalized = self.layernorm(weighted_sum)

        pooled = self.attn_pool(normalized)                   
        context = pooled

        hidden = self.linear1(context)
        hidden = self.relu(hidden)
        out = self.linear2(hidden)

        return out