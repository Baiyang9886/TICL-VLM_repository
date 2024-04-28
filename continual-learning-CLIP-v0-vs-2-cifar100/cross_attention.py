from functools import partial
import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    '''Feed forward层'''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    '''正则化操作'''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    '''将各个组件连接起来构成Decoder的基本单元'''
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DecoderLayer(nn.Module):
    '''Decoder 的基本结构单元'''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))  # 一个self attention layer
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))   # 一个 attention layer
        return self.sublayer[2](x, self.feed_forward)      # 一个 feed forward layer

class Decoder(nn.Module):
    '''将多个基本结构单元依次连接构成完整的 Decoder'''
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class CrossAttention(nn.Module):
<<<<<<< HEAD
    def __init__(self, device=None, in_dim=768, N=1, d_ff=1024, h=4, dropout=0.2):
=======
    def __init__(self, device=None, N=1, d_model=512, d_ff=1024, h=4, dropout=0.2):
>>>>>>> 25c00f178511b3454d08d3135c61295bcdab0f0d
        super(CrossAttention, self).__init__()
        self.dtype = torch.float16
        c = copy.deepcopy
        self. in_dim = in_dim
        self.s_len = 197
        self.out_dim = 512
        if in_dim == 1024:
            self.s_len = 257
            self.out_dim = 768
        scale = self. in_dim ** -0.5
        attn = MultiHeadedAttention(h, self.out_dim).to(device)
        ff = PositionwiseFeedForward(self.out_dim, d_ff, dropout).to(device)
        self.cross_attention = Decoder(DecoderLayer(self.out_dim, c(attn), c(attn),
                             c(ff), dropout).to(device), N).to(device)
        self.proj = nn.Parameter(scale * torch.randn(self. in_dim, self.out_dim)).type(self.dtype)
        self.text_projection = nn.Parameter(torch.empty(self.out_dim, self.out_dim)).type(self.dtype)
        self.proj = self.proj.to(device)
        self.text_projection = self.text_projection.to(device)
        self.ln_final = LayerNorm(self.out_dim)

    def forward(self, img_feature, task_feature):
        # 第一阶段 embedding
        B = img_feature.shape[0]
        task_features = task_feature
        for i in range(B-1):
            task_features = torch.cat([task_features, task_feature], dim=0)

        img_feature = img_feature.reshape(-1, self. in_dim).type(self.dtype)
        img_feature = img_feature @ self.proj
        img_feature = img_feature.reshape(-1, self.s_len, self.out_dim)
        task_features = task_features.unsqueeze(dim=1)

        img_feature = img_feature.type(torch.float32)
        task_features = task_features.type(torch.float32)
        feature = self.cross_attention(img_feature, task_features)   # 用img_feature的Query与text_feature（memory）的Key和Value做运算。
        # feature = feature.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(feature[:, 0, :]).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), task_texts.argmax(dim=-1)] @ self.text_projection

        return x
