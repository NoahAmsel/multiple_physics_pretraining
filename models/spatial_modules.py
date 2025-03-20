import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
from functools import partial
from timm.layers import DropPath
try:
    from .shared_modules import RelativePositionBias, ContinuousPositionBias1D, MLP
except:
    from shared_modules import RelativePositionBias, ContinuousPositionBias1D, MLP
    
from sympy import factorint, divisors

# Param builder func

    
def build_space_block(params):
    if params.space_type == 'axial_attention':
        return partial(AxialAttentionBlock, params.embed_dim, params.num_heads, bias_type=params.bias_type, token_mixing_struct=params.token_mixing_struct)
    else:
        raise NotImplementedError

### Space utils

class RMSInstanceNorm2d(nn.Module):
    def __init__(self, dim, affine=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim)) # Forgot to remove this so its in the pretrained weights
    
    def forward(self, x):
        std, mean = torch.std_mean(x, dim=(-2, -1), keepdims=True)
        x = (x) / (std + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None]  
        return x

    
class SubsampledLinear(nn.Module):
    """
    Cross between a linear layer and EmbeddingBag - takes in input 
    and list of indices denoting which state variables from the state
    vocab are present and only performs the linear layer on rows/cols relevant
    to those state variables
    
    Assumes (... C) input
    """
    def __init__(self, dim_in, dim_out, subsample_in=True):
        super().__init__()
        self.subsample_in = subsample_in
        self.dim_in = dim_in
        self.dim_out = dim_out
        temp_linear = nn.Linear(dim_in, dim_out)
        self.weight = nn.Parameter(temp_linear.weight)
        self.bias = nn.Parameter(temp_linear.bias)
    
    def forward(self, x, labels):
        # Note - really only works if all batches are the same input type
        labels = labels[0] # Figure out how to handle this for normal batches later
        label_size = len(labels)
        if self.subsample_in:
            scale = (self.dim_in / label_size)**.5 # Equivalent to swapping init to correct for given subsample of input
            x = scale * F.linear(x, self.weight[:, labels], self.bias)
        else:
            x = F.linear(x, self.weight[labels], self.bias[labels])
        return x

class hMLP_stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=(16,16), in_chans=3, embed_dim =768):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.in_proj = torch.nn.Sequential(
            *[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4, bias=False),
            RMSInstanceNorm2d(embed_dim//4, affine=True),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False),
            RMSInstanceNorm2d(embed_dim//4, affine=True),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),
            RMSInstanceNorm2d(embed_dim, affine=True),
            ]
            )
    
    def forward(self, x):
        x = self.in_proj(x)
        return x
    
    
class hMLP_output(nn.Module):
    """ Patch to Image De-bedding
    """
    def __init__(self, patch_size=(16,16), out_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.out_proj = torch.nn.Sequential(
            *[nn.ConvTranspose2d(embed_dim, embed_dim//4, kernel_size=2, stride=2, bias=False),
            RMSInstanceNorm2d(embed_dim//4, affine=True),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False),
            RMSInstanceNorm2d(embed_dim//4, affine=True),
            nn.GELU(),
            ])
        out_head = nn.ConvTranspose2d(embed_dim//4, out_chans, kernel_size=4, stride=4)
        self.out_kernel = nn.Parameter(out_head.weight)
        self.out_bias = nn.Parameter(out_head.bias)
    
    def forward(self, x, state_labels):
        x = self.out_proj(x)#.flatten(2).transpose(1, 2)
        x = F.conv_transpose2d(x, self.out_kernel[:, state_labels], self.out_bias[state_labels], stride=4)
        return x

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

class AxialAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12,  drop_path=0, layer_scale_init_value=1e-6, bias_type='rel', token_mixing_struct='low_rank'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.token_mixing_struct = token_mixing_struct

        if self.token_mixing_struct == "bilinearbtt":
            self.a, self.b = self.c, self.d = self.factorize(hidden_dim) # -> not true anymore...???

            # TODO: assume s=1 for now!!!
            self.s = 1
            
            # breakpoint()

            # # define BTT projection matrices
            # self.bilinear_btt_wk = BilinearBTT_W_K_Projection((self.c, self.d, self.num_heads*self.b*self.s))
            # self.bilinear_btt_wq = BilinearBTT_W_Q_Projection((self.num_heads*self.b, self.a, self.c*self.s))

            # self.bilinear_btt_ln_PLPRXT = LayerNorm(hidden_dim, bias=False)
            # self.bilinear_btt_ln_X = LayerNorm(hidden_dim, bias=False)

            # self.c_attn_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif self.token_mixing_struct == "low_rank":
            pass
        else:
            raise NotImplementedError
        
        # breakpoint()

        self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == 'continuous':
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = MLP(hidden_dim)
        self.mlp_norm = RMSInstanceNorm2d(hidden_dim, affine=True)

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

    def forward(self, x, bcs):
        # input is t x b x c x h x w 
        B, C, H, W = x.shape
        input = x.clone()
        x = self.norm1(x)
        x = self.input_head(x)

        # breakpoint()

        x = rearrange(x, 'b (he c) h w ->  b he h w c', he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)

        # breakpoint()

        # Do attention with current q, k, v matrices along each spatial axis then average results
        # X direction attention
        qx, kx, vx = map(lambda x: rearrange(x, 'b he h w c ->  (b h) he w c'), [q,k,v])
        rel_pos_bias_x = self.rel_pos_bias(W, W, bcs[0, 0])
        # Functional doesn't return attention mask :(
        if rel_pos_bias_x is not None:
            xx = F.scaled_dot_product_attention(qx, kx, vx, attn_mask=rel_pos_bias_x)
        else:
            xx = F.scaled_dot_product_attention(qx.contiguous(), kx.contiguous(), vx.contiguous())
        xx = rearrange(xx, '(b h) he w c -> b (he c) h w', h=H)
        # Y direction attention 
        qy, ky, vy = map(lambda x: rearrange(x, 'b he h w c ->  (b w) he h c'), [q,k,v])
        rel_pos_bias_y = self.rel_pos_bias(H, H, bcs[0, 1])

        if rel_pos_bias_y is not None:
            xy = F.scaled_dot_product_attention(qy, ky, vy, attn_mask=rel_pos_bias_y)
        else: # I don't understand why this was necessary but it was
            xy = F.scaled_dot_product_attention(qy.contiguous(), ky.contiguous(), vy.contiguous())
        xy = rearrange(xy, '(b w) he h c -> b (he c) h w', w=W)
        # Combine
        x = (xx + xy) / 2
        x = self.norm2(x)
        x = self.output_head(x)
        x = self.drop_path(x*self.gamma_att[None, :, None, None]) + input

        # MLP
        input = x.clone()
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.mlp(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None, :, None, None] * x)

        return output