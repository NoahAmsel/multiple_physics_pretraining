import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from functools import partial
from timm.layers import DropPath

from sympy import factorint, divisors

# from flash_cosine_sim_attention import flash_cosine_sim_attention
import math
try:
    from .shared_modules import RelativePositionBias, ContinuousPositionBias1D, MLP
except:
    from shared_modules import RelativePositionBias, ContinuousPositionBias1D, MLP

def build_time_block(params):
    """
    Builds a time block from the parameter file. 
    """
    if params.time_type == 'attention':
        return partial(AttentionBlock, params.embed_dim, params.num_heads, bias_type=params.bias_type, token_mixing_struct=params.token_mixing_struct)
    else:
        raise NotImplementedError

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class BilinearBTT_W_K_Projection(nn.Module):
    def __init__(self, shape):
        super().__init__()
        # self.weight.shape = (c, d, Hbs)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self, x):
        '''x.shape = (c, BT, d)'''

        if True:
            RMS_W_K = torch.sqrt(torch.mean(self.weight**2.) + 1e-8)
            d_in = self.weight.size(-2)
            d_out = self.weight.size(-1)
            init_scale_W_K = (min(d_in, d_out) / (d_in * d_in))**0.5
            W_K_normed = self.weight / max(1, RMS_W_K / init_scale_W_K)

        # output.shape = (c, BT, Hbs) = (c, BT, d) * (c, d, Hbs)
        return torch.bmm(x, W_K_normed)

class BilinearBTT_W_Q_Projection(nn.Module):
    def __init__(self, shape):
        super().__init__()
        # self.weight.shape = (Hb, a, cs)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self, x):
        '''x.shape = (Hb, cs, BT)'''

        if True:
            RMS_W_Q = torch.sqrt(torch.mean(self.weight**2.) + 1e-8)
            d_in = self.weight.size(-1)
            d_out = self.weight.size(-2)
            init_scale_W_Q = (min(d_in, d_out) / (d_in * d_in))**0.5
            W_Q_normed = self.weight / max(1, RMS_W_Q / init_scale_W_Q)

        # output.shape = (Hb, a, BT) = (Hb, a, cs) * (Hb, cs, BT)
        return torch.bmm(W_Q_normed, x)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, drop_path=0, layer_scale_init_value=1e-6, bias_type='rel', token_mixing_struct='low_rank'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm1 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None

        self.token_mixing_struct = token_mixing_struct

        if self.token_mixing_struct == "bilinearbtt":
            self.a, self.b = self.c, self.d = self.factorize(hidden_dim)

            # TODO: assume s=1 for now!!!
            self.s = 1

            # define BTT projection matrices
            self.bilinear_btt_wk = BilinearBTT_W_K_Projection((self.c, self.d, self.num_heads*self.b*self.s))
            self.bilinear_btt_wq = BilinearBTT_W_Q_Projection((self.num_heads*self.b, self.a, self.c*self.s))

            self.bilinear_btt_ln_PLPRXT = LayerNorm(hidden_dim, bias=False)
            self.bilinear_btt_ln_X = LayerNorm(hidden_dim, bias=False)

            self.c_attn_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        elif self.token_mixing_struct == "low_rank":
            self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
            self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
            self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        else:
            raise NotImplementedError

        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == 'continuous':
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def factorize(self, x, n=2):
        if n == 2:
            bigger = next(factor for factor in divisors(x) if factor > math.sqrt(x))
            return [x//bigger, bigger]

        # Get prime factors and their counts
        prime_factors = factorint(x)

        # Initialize the n integers
        numbers = [1] * n

        # Distribute the prime factors
        for prime, count in reversed(list(prime_factors.items())):
            for _ in range(count):
                # Find the number with the smallest product to assign the prime factor
                min_index = min(range(n), key=lambda i: numbers[i])
                numbers[min_index] *= prime

        # return in ascending order
        return sorted(numbers)

    def forward(self, x):
        # input is t x b x c x h x w 
        T, B, C, H, W = x.shape
        input = x.clone()
        # Rearrange and prenorm
        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.norm1(x)
        
        rel_pos_bias = self.rel_pos_bias(T, T)

        if self.token_mixing_struct == "bilinearbtt":
            v = self.c_attn_v(rearrange(x, '(t b) c h w -> (b h w) t c',t=T,b=B))
            v = rearrange(v, '(b h w) t (he d) -> (b h w) he t d', b=B, h=H, w=W, he=self.num_heads)

            att = rearrange(x, '(T B) (c d) H W ->  c (B H W T) d', T=T, c=self.c, d=self.d)
            att = self.bilinear_btt_wk(att)

            # att.shape = (Hb, cs, BT)
            att = rearrange(att, "c (B H W T) (he b s) -> (he b) (c s) (B H W T)", b=self.b, c=self.c, s=self.s, B=B, H=H, W=W, T=T, he=self.num_heads)

            # (Hb, a, BT) = (Hb, a, cs) * (Hb, cs, BT)
            att = self.bilinear_btt_wq(att)

            # att.shape (B, HT, ab)
            att = rearrange(att, "(he b) a (B H W T) -> (B H W) (he T) (a b)", he=self.num_heads, a=self.a, b=self.b, B=B, H=H, W=W, T=T)

            # # att_einsum.shape = (B, T, HT) | Final Contraction
            att = torch.bmm(self.bilinear_btt_ln_X(rearrange(x, '(t b) c h w -> (b h w) t c',t=T,b=B)), self.bilinear_btt_ln_PLPRXT(att).transpose(-2, -1))

            att = att * (1.0 / math.sqrt(self.hidden_dim)) # SP attn logits scaling

            att = rearrange(att, "(B H W) T (he N) -> (B H W) he T N", B=B, H=H, W=W, T=T, he=self.num_heads, N=T)

            att = att + rel_pos_bias

            att = torch.softmax(att, dim=-1)
            x = att @ v

        elif self.token_mixing_struct == "low_rank":

            x = self.input_head(x) # Q, K, V projections
            # Rearrange for attention
            x = rearrange(x, '(t b) (he c) h w ->  (b h w) he t c', t=T, he=self.num_heads)

            q, k, v = x.tensor_split(3, dim=-1)
            q, k = self.qnorm(q), self.knorm(k)
                
            if rel_pos_bias is not None:
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_pos_bias) 
            else:
                x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        else:
            raise NotImplementedError

        # Rearrange after attention
        x = rearrange(x, '(b h w) he t c -> (t b) (he c) h w', h=H, w=W)
        x = self.norm2(x) 
        x = self.output_head(x)
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)
        output = self.drop_path(x*self.gamma[None, None, :, None, None]) + input
        return output
