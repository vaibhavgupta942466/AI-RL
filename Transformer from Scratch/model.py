import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    """
    Input Embedding Layer which converts input tokens to their corresponding embeddings
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding Layer which adds positional information to the input embeddings
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class LayerNormalization(nn.Module):
    """
    Layer Normalization Layer
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class FeedForward(nn.Module):
    """
    Feed Forward Layer
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer
    Query -> Key -> value
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.linear_q(q).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.linear_o(output)

class ResidualConnection(nn.Module):
    """
    Residual Connection Layer
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    """
    Encoder Layer
    """
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super().__init__()
        self.N = N
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = x + self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    """
    Decoder Layer
    """
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.multi_head_attention(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super().__init__()
        self.N = N
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.embedding(x)
        x = x + self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """
    Projection Layer
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    """
    Transformer
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, N, heads, d_ff, dropout)
        self.projection = ProjectionLayer(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.projection(decoder_output)
    
def build_transformer(src_vocab_size, tgt_vocab_size, d_model=512, N=6, heads=8, d_ff=2048, dropout=0.1):
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, N, heads, d_ff, dropout)
    return model