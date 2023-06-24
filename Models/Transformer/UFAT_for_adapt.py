'''
UNet architecture: Factorized attention Transformer encoder, CNN decoder
Encoder is from MPViT
'''

import math
from pyexpat import features
import torch
from torch import nn, einsum
from einops import rearrange
import sys
from typing import Tuple
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')

from Models.Transformer.mpvit import FactorAtt_ConvRelPosEnc, ConvRelPosEnc, ConvPosEnc, Mlp, Conv2d_BN
from Models.Decoders import UnetDecodingBlock, UnetDecodingBlockTransformer, UnetDecodingBlockTransformer_M
from Models.Transformer.UFAT_for_adapt_KT import DWConv2d_BN_M, Conv2d_BN_M, DWCPatchEmbed_M



class DWConv2d_BN(nn.Module):
    """Depthwise Separable Convolution with BN module.
    Modify on MPViT DWConv2d_BN, this is for input output are different channel dim"""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=in_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()
            # elif isinstance(m, nn.InstanceNorm2d):
            #     m.weight.data.fill_(bn_weight_init)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        foward function
        """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding. The same as the module in MPViT"""
    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 conv_norm=nn.BatchNorm2d,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            norm_layer=conv_norm,
            act_layer=act_layer,
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class FactorAtt_ConvRelPosEnc_SEadapt(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class.
    Modified for domain attention. Follow Domain-attentive universal decoder
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.transform = nn.Sequential(nn.Conv1d(dim, self.num_heads,kernel_size=1,bias=False),
            nn.Sigmoid(),
            )

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)

        # Factorized attention.  Different from COAT
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)
        factor_att = self.scale*factor_att+crpe  # (B,H,N,dim)

        # TODO for domain attention 
        x = rearrange(x, 'b n c -> b c n').contiguous()
        domain_att = self.average_pool(x)   # (B,C,1)
        domain_att = self.transform(domain_att)  # (B,H,1)
        domain_att = domain_att.unsqueeze(3)  # (B,H,1,1)
        x = domain_att*factor_att+factor_att   # (B,H,N,dim)


        # # Convolutional relative position encoding.
        # crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        # x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FactorAtt_ConvRelPosEnc_SE1adapt(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class.
    Modified for domain attention. Follow Domain-attentive universal decoder
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
        r=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        hidden_dim = max(dim//r,32)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.transform = nn.Sequential(nn.Conv1d(dim, hidden_dim,kernel_size=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, self.num_heads, kernel_size=1, bias=False),
            nn.Sigmoid(),
            )

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)

        # Factorized attention.  Different from COAT
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)
        factor_att = self.scale*factor_att+crpe  # (B,H,N,dim)

        # TODO for domain attention 
        domain_att = rearrange(factor_att, 'b h n k -> b (h k) n').contiguous()  # (B,H*C,N)
        domain_att = self.average_pool(domain_att)  # (B,H*C,1)
        domain_att = self.transform(domain_att)   # (B,H,1)
        domain_att = domain_att.unsqueeze(3)  # (B,H,1,1)
        x = domain_att*factor_att+factor_att
        # x = rearrange(x, 'b n c -> b c n').contiguous()
        # domain_att = self.average_pool(x)   # (B,C,1)
        # domain_att = self.transform(domain_att)  # (B,H,1)
        # domain_att = domain_att.unsqueeze(3)  # (B,H,1,1)
        # x = domain_att*factor_att+factor_att   # (B,H,N,dim)
        #  domain_att = torch.sum(factor_att,dim=1,keepdim=False)  # (b,n,k)
        # domain_att = rearrange(domain_att, 'b n k -> b k n')  # (b,k,n)
        # domain_att = self.average_pool(domain_att)   # (B,k,1)
        # domain_att = self.transform(domain_att)  # (B,hidden,1)
        # domain_att = self.fc_select(domain_att)  # (B,h*k,1)
        # domain_att = rearrange(domain_att, 'b (h k) c -> b h c k', h=self.num_heads).contiguous()  # (b,h,1,k)
        # domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        # factor_att = domain_att*factor_att   # (B,H,N,dim)       


        # # Convolutional relative position encoding.
        # crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        # x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x        


class FactorAtt_ConvRelPosEnc_SK(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class.
    Modified for domain attention. Follow Selective kernel
    r: ratio, max(32,n//r) is the hidden size for the fc layer in domain attention
    """
    def __init__(
        self,
        seq_length,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
        r=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        hidden_dim = max(head_dim//r,4)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.transform = nn.Sequential(nn.Conv1d(head_dim, hidden_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            )
        self.fc_select = nn.Conv1d(hidden_dim,self.num_heads*head_dim,kernel_size=1,bias=False)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.  Different from COAT
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)

        crpe = self.crpe(q, v, size=size)
        factor_att = self.scale * factor_att + crpe  

        # TODO for domain attention 
        domain_att = torch.sum(factor_att,dim=1,keepdim=False)  # (b,n,k)
        domain_att = rearrange(domain_att, 'b n k -> b k n')  # (b,k,n)
        domain_att = self.average_pool(domain_att)   # (B,k,1)
        domain_att = self.transform(domain_att)  # (B,hidden,1)
        domain_att = self.fc_select(domain_att)  # (B,h*k,1)
        domain_att = rearrange(domain_att, 'b (h k) c -> b h c k', h=self.num_heads).contiguous()  # (b,h,1,k)
        domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        factor_att = domain_att*factor_att   # (B,H,N,dim)


        # Convolutional relative position encoding.
        # crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        # x = self.scale * factor_att + crpe
        x = factor_att
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class FactorAtt_ConvRelPosEnc_SupSK(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class.
    Modified for domain attention. Follow Selective kernel. Add domain label 
    r: ratio, max(32,n//r) is the hidden size for the fc layer in domain attention
    """
    def __init__(
        self,
        seq_length,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
        r=2,
        num_domains=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        hidden_dim = max(int(head_dim/r),4)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.transform = nn.Sequential(nn.Conv1d(head_dim, hidden_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            )
        self.fc_select = nn.Conv1d(hidden_dim,self.num_heads*head_dim,kernel_size=1,bias=False)

        self.domain_layer = nn.Sequential(
            nn.Linear(num_domains, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size, domain_label):
        """foward function
        domain_label is one_hot vector
        """
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.  Different from COAT
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)
        factor_att = self.scale * factor_att + crpe  # (b,h,n,k)

        # TODO for domain attention 
        domain_att = torch.sum(factor_att,dim=1,keepdim=False)  # (b,n,k)
        domain_att = rearrange(domain_att, 'b n k -> b k n')  # (b,k,n)
        domain_att = self.average_pool(domain_att)   # (B,k,1)
        domain_att = self.transform(domain_att)  # (B,hidden,1)
        domain_label_up = self.domain_layer(domain_label).unsqueeze(2)  # (B,hidden,1)
        domain_att = domain_att+domain_label_up   # (B,hidden,1)
        domain_att = self.fc_select(domain_att)  # (B,h*k,1)
        domain_att = rearrange(domain_att, 'b (h k) c -> b h c k', h=self.num_heads).contiguous()  # (b,h,1,k)
        domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        x = domain_att*factor_att   # (B,H,N,dim)

        # TODO change merge before the domain attention 
        # x = factor_att
        # Convolutional relative position encoding.
        # crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        # x = self.scale * factor_att + crpe  
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FactorAtt_ConvRelPosEnc_Sup(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class.
    Modified for domain attention. Follow Selective kernel. Add domain label 
    r: ratio, max(32,n//r) is the hidden size for the fc layer in domain attention
    """
    def __init__(
        self,
        seq_length,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
        r=2,
        num_domains=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        hidden_dim = max(dim//r,4)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.average_pool = nn.AdaptiveAvgPool1d(1)
        # self.transform = nn.Sequential(nn.Conv1d(head_dim, hidden_dim,kernel_size=1,bias=False),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     )
        # self.fc_select = nn.Conv1d(hidden_dim,self.num_heads*head_dim,kernel_size=1,bias=False)

        self.domain_layer = nn.Sequential(
            nn.Linear(num_domains, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,self.num_heads*head_dim),
        )

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size, domain_label):
        """foward function
        domain_label is one_hot vector
        """
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.  Different from COAT
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)
        factor_att = self.scale * factor_att + crpe  

        # TODO for domain attention 
        domain_att = self.domain_layer(domain_label).unsqueeze(2)  # (B,H*K,1)
        domain_att = rearrange(domain_att, 'b (h k) c -> b h c k', h=self.num_heads).contiguous()  # (b,h,1,k)
        domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        x = domain_att*factor_att   # (B,H,N,dim)
        # domain_att = torch.sum(factor_att,dim=1,keepdim=False)  # (b,n,k)
        # domain_att = rearrange(domain_att, 'b n k -> b k n')  # (b,k,n)
        # domain_att = self.average_pool(domain_att)   # (B,k,1)
        # domain_att = self.transform(domain_att)  # (B,hidden,1)
        # domain_label_up = self.domain_layer(domain_label).unsqueeze(2)  # (B,hidden,1)
        # domain_att = domain_att+domain_label_up   # (B,hidden,1)
        # domain_att = self.fc_select(domain_att)  # (B,h*k,1)
        # domain_att = rearrange(domain_att, 'b (h k) c -> b h c k', h=self.num_heads).contiguous()  # (b,h,1,k)
        # domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        # factor_att = domain_att*factor_att   # (B,H,N,dim)

        # TODO change merge before the domain attention 
        # x = factor_att
        # Convolutional relative position encoding.
        # crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        # x = self.scale * factor_att + crpe  
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FactorAtt_ConvRelPosEnc_Sup2(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class.
    Modified for Sup2 domain attention. 
    Use domain label to attend different heads
    r: ratio, max(32,n//r) is the hidden size for the fc layer in domain attention
    """
    def __init__(
        self,
        seq_length,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
        r=2,
        num_domains=4,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        hidden_dim = max(dim//r,4)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.domain_layer = nn.Sequential(
            nn.Linear(num_domains, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32,self.num_heads),
        )

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size, domain_label):
        """foward function
        domain_label is one_hot vector
        """
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.  Different from COAT
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)
        factor_att = self.scale * factor_att + crpe  

        # TODO for domain attention 
        domain_att = self.domain_layer(domain_label).unsqueeze(2).unsqueeze(2)  # (B,H,1)
        domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        x = domain_att*factor_att   # (B,H,N,dim)


        # TODO change merge before the domain attention 
        # x = factor_att
        # Convolutional relative position encoding.
        # crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        # x = self.scale * factor_att + crpe  
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SerialBlock_adapt(nn.Module):
    """ Serial block class. For UFAT
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. 
        input: x (B,N,C), (H,W)  output: out (B,N,C)"""
    def __init__(self, seq_length, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpe=None, shared_crpe=None, 
                 adapt_method=None, num_domains=4):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe
        self.norm1 = norm_layer(dim)
        self.adapt_method = adapt_method

        if self.adapt_method =='SE':
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc_SEadapt(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe
            )
        elif self.adapt_method == 'SE1':
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc_SE1adapt(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe
            )
        elif self.adapt_method == 'SK':
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc_SK(
                seq_length, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe
            ) 
        elif self.adapt_method == 'SupSK':
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc_SupSK(
                seq_length, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe, num_domains=num_domains,
            ) 
        elif self.adapt_method == 'Sup':
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc_Sup(
                seq_length, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe, num_domains=num_domains,
            ) 
        elif self.adapt_method == 'Sup2':
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc_Sup2(
                seq_length, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe, num_domains=num_domains,
            ) 
        else:
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Tuple[int, int], domain_label=None):
        # Conv-Attention.
        x = self.cpe(x, size)
        cur = self.norm1(x)
        if domain_label != None :
            cur = self.factoratt_crpe(cur, size, domain_label)
        else:
            cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur) 

        # MLP. 
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x


class SerialBlock_adapt_M(nn.Module):
    """ Serial block class. For UFAT
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. 
        input: x (B,N,C), (H,W)  output: out (B,N,C)"""
    def __init__(self, seq_length, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpe=None, shared_crpe=None, 
                 adapt_method=None, num_domains=4):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe
        # self.norm1 = norm_layer(dim)
        self.norm1s = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)])
        self.adapt_method = adapt_method

        if self.adapt_method == 'Sup':
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc_Sup(
                seq_length, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe, num_domains=num_domains,
            ) 
        else:
            self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        # self.norm2 = norm_layer(dim)
        self.norm2s = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Tuple[int, int], domain_label=None, d=None):
        # Conv-Attention.
        d = int(d)
        x = self.cpe(x, size)
        # cur = self.norm1(x)
        cur = self.norm1s[d](x)
        if domain_label != None :
            cur = self.factoratt_crpe(cur, size, domain_label)
        else:
            cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur) 

        # MLP. 
        # cur = self.norm2(x)
        cur = self.norm2s[d](x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x



class MHSA_stage_adapt(nn.Module):
    '''
    Multi-head self attention
    (B, N, C) --> (B, N, C)
    Combine several Serial blocks for a stage
    '''
    def __init__(self, seq_length, dim, num_layers, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_domains=4,
        norm_layer=nn.LayerNorm, adapt_method=None, crpe_window={3:2, 5:3, 7:3}):
        super(MHSA_stage_adapt, self).__init__()

        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim//num_heads, h=num_heads, window=crpe_window)

        self.mhca_blks = nn.ModuleList(
            [SerialBlock_adapt(
                seq_length, dim, num_heads, mlp_ratio, qkv_bias, qk_scale, 
                drop_rate, attn_drop_rate, drop_path_rate,
                nn.GELU, norm_layer, self.cpe, self.crpe, adapt_method,num_domains,
            ) for _ in range(num_layers)]
        )

    def forward(self, input, H, W, domain_label=None):
        for blk in self.mhca_blks:
            input = blk(input, size=(H,W)) if domain_label==None else blk(input, (H,W), domain_label)
        return input



class MHSA_stage_adapt_M(nn.Module):
    '''
    Multi-head self attention
    (B, N, C) --> (B, N, C)
    Combine several Serial blocks for a stage
    '''
    def __init__(self, seq_length, dim, num_layers, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_domains=4,
        norm_layer=nn.LayerNorm, adapt_method=None, crpe_window={3:2, 5:3, 7:3}):
        super(MHSA_stage_adapt_M, self).__init__()

        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim//num_heads, h=num_heads, window=crpe_window)

        self.mhca_blks = nn.ModuleList(
            [SerialBlock_adapt_M(
                seq_length, dim, num_heads, mlp_ratio, qkv_bias, qk_scale, 
                drop_rate, attn_drop_rate, drop_path_rate,
                nn.GELU, norm_layer, self.cpe, self.crpe, adapt_method,num_domains,
            ) for _ in range(num_layers)]
        )

    def forward(self, input, H, W, domain_label=None, d=None):
        for blk in self.mhca_blks:
            input = blk(input, size=(H,W),d=d) if domain_label==None else blk(input, (H,W), domain_label,d)
        return input



class FAT_Transformer_adapt(nn.Module):
    '''
    A Conv Position encoding + Factorized attention Transformer
    Input: an image
    Output: a list contains features from each stage
    '''
    def __init__(
        self,
        img_size=512,
        in_chans=3,
        num_stages=4,
        num_layers=[1, 1, 1, 1],
        embed_dims=[48, 96, 192, 384],
        mlp_ratios=[8, 8, 4, 4],
        num_heads=[8, 8, 8, 8],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d,
        adapt_method=None,
        num_domains=4,
        **kwargs,
    ):
        super(FAT_Transformer_adapt, self).__init__()
        self.num_stages = num_stages

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dims[idx] if idx==0 else embed_dims[idx-1],
                embed_dim=embed_dims[idx],
                patch_size=3,
                stride=1 if idx==0 else 2, 
                conv_norm=conv_norm,
            ) for idx in range(self.num_stages)
        ])
        
        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhsa_stages = nn.ModuleList([
            MHSA_stage_adapt(
                (img_size//2**(idx+2))**2,
                embed_dims[idx],
                num_layers=num_layers[idx],
                num_heads=num_heads[idx], 
                mlp_ratio=mlp_ratios[idx], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer,
                adapt_method=adapt_method,
                num_domains=num_domains,
            ) for idx in range(self.num_stages)
        ])
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, domain_label=None):
        # x (B,in_chans,H,W)
        x = self.stem(x)  # (B,embed_dim[0],H/4,W/4)
        out = []
        for idx in range(self.num_stages):
            x = self.patch_embed_stages[idx](x)  # (B, embed_dim[idx],H/(4*2^idx),W/(4*2^idx))
            B,C,H,W = x.shape
            x = rearrange(x, 'b c w h -> b (w h) c')
            x = self.mhsa_stages[idx](x, H, W) if domain_label==None else self.mhsa_stages[idx](x, H, W,domain_label)
            x = rearrange(x, 'b (w h) c -> b c w h', w=W, h=H).contiguous()
            out.append(x)
        
        return out
 



class UFAT_adapt(nn.Module):
    '''
    Unet architecture Factorized Transformer, used for segmentation
    tran_dim: dim between attention and mlp in transformer layer
    dim_head: dim in the attention
    '''
    def __init__(self, 
        img_size=512,
        in_chans=3, 
        num_stages = 4, 
        num_layers=[2, 2, 2, 2],
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        num_heads=[8, 8, 8, 8],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d,
        adapt_method=None,
        num_domains=4,  # used for input supervised domain information
        ):
        super(UFAT_adapt, self).__init__()
        # encoder
        #    0         1        2        3
        # [(48,128),(96,64),(192,32),(384,16)]
        self.encoder = FAT_Transformer_adapt(img_size,in_chans,num_stages,num_layers,embed_dims,mlp_ratios,
                                        num_heads,qkv_bias,qk_scale,
                                        drop_rate,attn_drop_rate,drop_path_rate,norm_layer,nn.InstanceNorm2d,adapt_method,num_domains)

        # bridge 
        self.bridge = nn.Sequential(
            nn.Conv2d(embed_dims[3],embed_dims[3],kernel_size=3,stride=1, padding=1),
            conv_norm(embed_dims[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[3],embed_dims[3]*2,kernel_size=3,stride=1, padding=1),
            conv_norm(embed_dims[3]*2),
            nn.ReLU(inplace=True)
        )

        # decoder
        self.decoder1 = UnetDecodingBlock(embed_dims[3]*2,embed_dims[3],conv_norm=conv_norm)  # 768,384
        self.decoder2 = UnetDecodingBlock(embed_dims[3],embed_dims[2],conv_norm=conv_norm)  # 384,192
        self.decoder3 = UnetDecodingBlock(embed_dims[2],embed_dims[1],conv_norm=conv_norm)   # 192,96
        self.decoder4 = UnetDecodingBlock(embed_dims[1],embed_dims[0],conv_norm=conv_norm)    # 96,48
        self.finalconv = nn.Sequential(
            nn.Conv2d(embed_dims[0], 1, kernel_size=1)
        )
        # self.finalconv = nn.Conv2d(embed_dims[0],1,kernel_size=1)  # 48,1

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        pass


    def forward(self,x,domain_label=None,out_feat=False,):
        # encoding
        #    0         1        2        3
        # [(48,128),(96,64),(192,32),(384,16)]
        B = x.shape[0]
        encoder_outs = self.encoder(x) if domain_label==None else self.encoder(x,domain_label)

        # bridge
        out = self.bridge(encoder_outs[3])

        # decoding
        out = self.decoder1(out, encoder_outs[3])  # (384,16,16)
        out = self.decoder2(out, encoder_outs[2])  # (192,32,32)
        out = self.decoder3(out, encoder_outs[1])  # (96,64,64)
        out = self.decoder4(out, encoder_outs[0])  # (48,128,128)
        
        # upsample
        out = nn.functional.interpolate(out,size = x.size()[2:],mode = 'bilinear', align_corners=False) # (48,512,512)
        out = self.finalconv(out)  # (1,512,512)
        
        if out_feat:
            return {'seg': out, 'feat': nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)}
        else:
            return out



class FATNet_adapt(nn.Module):
    '''
    A Conv Position encoding + Factorized attention Transformer
    use transformer encoder and decoder
    feature_dim is the 4th stage output dimension
    do_detach: ture means detach the feature from the last encoder, then pass into projection head
    Input: an image
    Output: a list contains features from each stage
    '''
    def __init__(
        self,
        img_size=512,
        in_chans=3,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        num_heads=[8, 8, 8, 8],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d,
        adapt_method=None,
        num_domains=4,
        feature_dim=512,
        do_detach = False,
        **kwargs,
    ):
        super(FATNet_adapt, self).__init__()
        self.num_stages = num_stages
        self.do_detach = do_detach

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dims[idx] if idx==0 else embed_dims[idx-1],
                embed_dim=embed_dims[idx],
                patch_size=3,
                stride=1 if idx==0 else 2, 
                conv_norm=conv_norm,
            ) for idx in range(self.num_stages)
        ])

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhsa_stages = nn.ModuleList([
           MHSA_stage_adapt(
                (img_size//2**(idx+2))**2,
                embed_dims[idx],
                num_layers=num_layers[idx],
                num_heads=num_heads[idx], 
                mlp_ratio=mlp_ratios[idx], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer,
                adapt_method=adapt_method,
                num_domains=num_domains
            ) for idx in range(self.num_stages)
        ])


        # bridge 
        self.bridge = nn.Sequential(
            nn.Conv2d(embed_dims[3],embed_dims[3],kernel_size=3,stride=1, padding=1),
            conv_norm(embed_dims[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[3],embed_dims[3]*2,kernel_size=3,stride=1, padding=1),
            conv_norm(embed_dims[3]*2),
            nn.ReLU(inplace=True)
        )


        # decoder
        self.mhsa_list = []
        for idx in range(self.num_stages):
            self.mhsa_list.append(
                MHSA_stage_adapt(
                (img_size//2**(idx+2))**2,
                embed_dims[idx],
                num_layers=num_layers[idx],
                num_heads=num_heads[idx], 
                mlp_ratio=mlp_ratios[idx], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer,
                adapt_method=adapt_method,
                num_domains=num_domains
                )            
            )

        self.decoder1 = UnetDecodingBlockTransformer(embed_dims[3]*2,embed_dims[3],self.mhsa_list[3],conv_norm=conv_norm)  # 768,384
        self.decoder2 = UnetDecodingBlockTransformer(embed_dims[3],embed_dims[2],self.mhsa_list[2],conv_norm=conv_norm)  # 384,192
        self.decoder3 = UnetDecodingBlockTransformer(embed_dims[2],embed_dims[1],self.mhsa_list[1],conv_norm=conv_norm)   # 192,96
        self.decoder4 = UnetDecodingBlockTransformer(embed_dims[1],embed_dims[0],self.mhsa_list[0],conv_norm=conv_norm)    # 96,48
        self.finalconv = nn.Sequential(
            nn.Conv2d(embed_dims[0], 1, kernel_size=1)
        )    

        # projection head for the 4th stage output
        self.feature_dim = feature_dim
        if feature_dim == embed_dims[3]:
            self.proj_head = nn.Identity()
        elif feature_dim == 15:
            # use ce loss
            self.proj_head = nn.Linear(embed_dims[3], 15) 
        else:
            self.proj_head = nn.Sequential(
                 nn.Linear(embed_dims[3], 512),
                 nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, feature_dim),
            )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x, domain_label=None, out_feat=False, out_seg=True):
        # out_feat if output mid features
        # out_seg  if output segmentation prediction
        # x (B,in_chans,H,W)
        img_size = x.size()[2:]
        x = self.stem(x)  # (B,embed_dim[0],H/4,W/4)
        encoder_outs = []
        for idx in range(self.num_stages):
            x = self.patch_embed_stages[idx](x)  # (B, embed_dim[idx],H/(4*2^idx),W/(4*2^idx))
            B,C,H,W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.mhsa_stages[idx](x, H, W) if domain_label==None else self.mhsa_stages[idx](x, H, W,domain_label)
            x = rearrange(x, 'b (h w) c -> b c h w', w=W, h=H).contiguous()
            encoder_outs.append(x)
        
        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            if self.feature_dim == 15:
                return {'seg': None, 'feat': x, 'pred': self.proj_head(x)}
            if self.do_detach:
                x_d = x.detach()
            else: x_d = x
            x_d = self.proj_head(x_d)
            return {'seg': None, 'feat': x_d}

        # bridge
        out = self.bridge(encoder_outs[3])

        # decoding
        out = self.decoder1(out, encoder_outs[3]) if domain_label==None else self.decoder1(out, encoder_outs[3],domain_label) # (384,16,16)
        out = self.decoder2(out, encoder_outs[2]) if domain_label==None else self.decoder2(out, encoder_outs[2],domain_label) # (192,32,32)
        out = self.decoder3(out, encoder_outs[1]) if domain_label==None else self.decoder3(out, encoder_outs[1],domain_label) # (96,64,64)
        out = self.decoder4(out, encoder_outs[0]) if domain_label==None else self.decoder4(out, encoder_outs[0],domain_label) # (48,128,128)
        
        # upsample
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        out = self.finalconv(out)  # (1,512,512)            
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            if self.feature_dim == 15:
                return {'seg': out, 'feat': x, 'pred': self.proj_head(x)}
            if self.do_detach:
                x_d = x.detach()
            else: x_d = x
            x_d = self.proj_head(x_d)
            return {'seg': out, 'feat': x_d}
        else:
            return out



class FATNet_DSN(nn.Module):
    '''
    use domain-specific normalization
    A Conv Position encoding + Factorized attention Transformer
    use transformer encoder and decoder
    feature_dim is the 4th stage output dimension
    do_detach: ture means detach the feature from the last encoder, then pass into projection head
    Input: an image
    Output: a list contains features from each stage
    '''
    def __init__(
        self,
        img_size=512,
        in_chans=3,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        num_heads=[8, 8, 8, 8],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d,
        adapt_method=None,
        num_domains=4,
        feature_dim=512,
        do_detach = False,
        **kwargs,
    ):
        super(FATNet_DSN, self).__init__()
        self.num_stages = num_stages
        self.do_detach = do_detach

        # self.stem = nn.Sequential(
        #     Conv2d_BN_M(
        #         in_chans,
        #         embed_dims[0] // 2,
        #         kernel_size=3,
        #         stride=2,
        #         pad=1,
        #         act_layer=nn.Hardswish,
        #         num_domains=num_domains
        #     ),
        #     Conv2d_BN_M(
        #         embed_dims[0] // 2,
        #         embed_dims[0],
        #         kernel_size=3,
        #         stride=2,
        #         pad=1,
        #         act_layer=nn.Hardswish,
        #         num_domains=num_domains
        #     ),
        # )
        self.stem_1 = Conv2d_BN_M(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                num_domains=num_domains
            )
        self.stem_2 =  Conv2d_BN_M(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                num_domains=num_domains
            )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList([
            DWCPatchEmbed_M(
                in_chans=embed_dims[idx] if idx==0 else embed_dims[idx-1],
                embed_dim=embed_dims[idx],
                patch_size=3,
                stride=1 if idx==0 else 2, 
                conv_norm=conv_norm,
                num_domains=num_domains
            ) for idx in range(self.num_stages)
        ])

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhsa_stages = nn.ModuleList([
           MHSA_stage_adapt_M(
                (img_size//2**(idx+2))**2,
                embed_dims[idx],
                num_layers=num_layers[idx],
                num_heads=num_heads[idx], 
                mlp_ratio=mlp_ratios[idx], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer,
                adapt_method=adapt_method,
                num_domains=num_domains
            ) for idx in range(self.num_stages)
        ])


        # bridge 
        # self.bridge = nn.Sequential(
        #     nn.Conv2d(embed_dims[3],embed_dims[3],kernel_size=3,stride=1, padding=1),
        #     conv_norm(embed_dims[3]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(embed_dims[3],embed_dims[3]*2,kernel_size=3,stride=1, padding=1),
        #     conv_norm(embed_dims[3]*2),
        #     nn.ReLU(inplace=True)
        # )
        # bridge 
        self.bridge_conv1 = nn.Conv2d(embed_dims[3],embed_dims[3],kernel_size=3,stride=1, padding=1)
        self.bridge_norms1 = nn.ModuleList([conv_norm(embed_dims[3]) for _ in range(num_domains)])
        self.bridge_act1 = nn.ReLU(inplace=True)
        self.bridge_conv2 = nn.Conv2d(embed_dims[3],embed_dims[3]*2,kernel_size=3,stride=1, padding=1)
        self.bridge_norms2 = nn.ModuleList([conv_norm(embed_dims[3]*2) for _ in range(num_domains)])
        self.bridge_act2 = nn.ReLU(inplace=True)

        # decoder
        self.mhsa_list = []
        for idx in range(self.num_stages):
            self.mhsa_list.append(
                MHSA_stage_adapt_M(
                (img_size//2**(idx+2))**2,
                embed_dims[idx],
                num_layers=num_layers[idx],
                num_heads=num_heads[idx], 
                mlp_ratio=mlp_ratios[idx], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer,
                adapt_method=adapt_method,
                num_domains=num_domains
                )            
            )

        self.decoder1 = UnetDecodingBlockTransformer_M(embed_dims[3]*2,embed_dims[3],self.mhsa_list[3],conv_norm=conv_norm,num_domains=num_domains)  # 768,384
        self.decoder2 = UnetDecodingBlockTransformer_M(embed_dims[3],embed_dims[2],self.mhsa_list[2],conv_norm=conv_norm,num_domains=num_domains)  # 384,192
        self.decoder3 = UnetDecodingBlockTransformer_M(embed_dims[2],embed_dims[1],self.mhsa_list[1],conv_norm=conv_norm,num_domains=num_domains)   # 192,96
        self.decoder4 = UnetDecodingBlockTransformer_M(embed_dims[1],embed_dims[0],self.mhsa_list[0],conv_norm=conv_norm,num_domains=num_domains)    # 96,48
        self.finalconv = nn.Sequential(
            nn.Conv2d(embed_dims[0], 1, kernel_size=1)
        )    

        # projection head for the 4th stage output
        self.feature_dim = feature_dim
        if feature_dim == embed_dims[3]:
            self.proj_head = nn.Identity()
        elif feature_dim == 15:
            # use ce loss
            self.proj_head = nn.Linear(embed_dims[3], 15) 
        else:
            self.proj_head = nn.Sequential(
                 nn.Linear(embed_dims[3], 512),
                 nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, feature_dim),
            )

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x, domain_label=None, d=None, out_feat=False, out_seg=True):
        # out_feat if output mid features
        # out_seg  if output segmentation prediction
        # x (B,in_chans,H,W)
        img_size = x.size()[2:]
        # x = self.stem(x)  # (B,embed_dim[0],H/4,W/4)
        x = self.stem_1(x,d=d)
        x = self.stem_2(x,d=d)
        encoder_outs = []
        for idx in range(self.num_stages):
            x = self.patch_embed_stages[idx](x,d)  # (B, embed_dim[idx],H/(4*2^idx),W/(4*2^idx))
            B,C,H,W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.mhsa_stages[idx](x, H, W,d=d) if domain_label==None else self.mhsa_stages[idx](x, H, W,domain_label,d)
            x = rearrange(x, 'b (h w) c -> b c h w', w=W, h=H).contiguous()
            encoder_outs.append(x)
        
        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            if self.feature_dim == 15:
                return {'seg': None, 'feat': x, 'pred': self.proj_head(x)}
            if self.do_detach:
                x_d = x.detach()
            else: x_d = x
            x_d = self.proj_head(x_d)
            return {'seg': None, 'feat': x_d}

        # bridge
        # out = self.bridge(encoder_outs[3])
        d_int = int(d)
        out = self.bridge_conv1(encoder_outs[3])
        out = self.bridge_norms1[d_int](out)
        out = self.bridge_act1(out)
        out = self.bridge_conv2(out)
        out = self.bridge_norms2[d_int](out)
        out = self.bridge_act2(out)

        # decoding
        out = self.decoder1(out, encoder_outs[3],d=d) if domain_label==None else self.decoder1(out, encoder_outs[3],d,domain_label) # (384,16,16)
        out = self.decoder2(out, encoder_outs[2],d=d) if domain_label==None else self.decoder2(out, encoder_outs[2],d,domain_label) # (192,32,32)
        out = self.decoder3(out, encoder_outs[1],d=d) if domain_label==None else self.decoder3(out, encoder_outs[1],d,domain_label) # (96,64,64)
        out = self.decoder4(out, encoder_outs[0],d=d) if domain_label==None else self.decoder4(out, encoder_outs[0],d,domain_label) # (48,128,128)
        
        # upsample
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        out = self.finalconv(out)  # (1,512,512)            
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            if self.feature_dim == 15:
                return {'seg': out, 'feat': x, 'pred': self.proj_head(x)}
            if self.do_detach:
                x_d = x.detach()
            else: x_d = x
            x_d = self.proj_head(x_d)
            return {'seg': out, 'feat': x_d}
        else:
            return out





if __name__ == '__main__':
    x = torch.randn(5,3,512,512)
    domain_label = torch.randint(0,4,(5,))
  

    domain_label = torch.nn.functional.one_hot(domain_label, 4).float()

    model = FATNet_adapt(adapt_method='Sup',num_domains=4)
    # model = FATNet_DSN(adapt_method=Fal4se, num_domains=1)
    y = model(x, domain_label, out_feat=True)
    # y = model(x,d='0',out_feat=True)
    print(y['seg'].shape)
    print(y['feat'].shape)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    # flops = FlopCountAnalysis(model, x)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # acts = ActivationCountAnalysis(model, x)

    # print(f"total flops : {flops.total()/1e12} M")
    # print(f"total activations: {acts.total()/1e6} M")
    print(f"number of parameter: {param/1e6} M")

    count = 0
    for name, params in model.named_parameters():
        # print(name)
        if 'norm' in name:
            count += params.numel()
    print(f'number of params in Norm: {count/1e6} M')