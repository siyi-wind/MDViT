'''
Add selective patch mechanism to Transformer
DWC_patch_embed uses differnt kernel sizes separable depth-wise convolution, to get several feature maps and append them
MP_Attention 
'''
import torch
from torch import nn, einsum
from typing import Type, Any, Callable, Union, List, Optional
from einops import rearrange, repeat

import sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')
from Models.Transformer.Vit import pair, FeedForward


class PreNorm(nn.Module):
    def __init__(self, dim, fn, size=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.size = size
    def forward(self, x, **kwargs):
        if self.size:
            return self.fn(self.norm(x), self.size, **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class DWConv2d_BN(nn.Module):
    """
    Depthwise Separable Conv
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        act_layer=nn.Hardswish,
    ):
        super().__init__()
        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Patch_Embed_stage(nn.Module):
    '''
    input x, output a list, including feature maps using different cnn kernel sizes
    notice: input output are the same shapes
    '''
    def __init__(self, embed_dim, num_path = 4, is_pool=False):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList(
            [DWConv2d_BN(
                    in_ch=embed_dim,
                    out_ch=embed_dim,
                    kernel_size=2 if (is_pool and idx==0) else 1,
                    stride=1,  
                )
                for idx in range(num_path)])

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)
        assert att_inputs[0].shape == x.shape

        return att_inputs


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.
        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.  Like {3: 2, 5: 3, 7: 3}
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
                )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        return EV_hat_img # EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding class.
    dim: input dim
    head_dim: q k v dimension
    """
    def __init__(
        self,
        dim,
        head_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads*head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape
        
        # Generate Q, K, V.
        qkv = self.to_qkv(x).chunk(3, dim = -1) # [(b,n,inner_dim),same,same]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        
        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = (
            x.transpose(1, 2).contiguous().reshape(B, N, self.head_dim*self.num_heads).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FactorConv_Transformer(nn.Module):
    '''
    Factor attention + convolution relative positional encoding Transformer
    size: a turple
    dim: input x dim  (B,N,C)
    depth: num of transformer blocks
    '''
    def __init__(self, size, dim, depth, heads, head_dim, mlp_dim, dropout = 0., crpe_window={3: 2, 5: 3, 7: 3}):
        super().__init__()
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=head_dim, h=heads, window=crpe_window)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FactorAtt_ConvRelPosEnc(dim, head_dim = head_dim,
                 num_heads = heads, proj_drop = dropout, shared_crpe=self.crpe), size),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, size):
        for attn, ff in self.layers:
            if self.cpe:
                x = self.cpe(x, size)
            x = attn(x) + x
            x = ff(x) + x
        return x


if __name__ == '__main__':
    x = torch.randn(4,1024,128)
    net = FactorConv_Transformer(size=(32,32), dim=128,depth=6,heads=8,head_dim=128,mlp_dim=1024, dropout=0.3)
    total_trainable_params = sum(
                    p.numel() for p in net.parameters() if p.requires_grad)
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    y = net(x, (32,32))
    print(y.shape)