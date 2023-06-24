'''
Follow Swin Transformer and AIM
https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py 
https://github.com/taoyang1122/adapt-image-models/blob/main/mmaction/models/backbones/vit_clip.py 
'''
import math
from pyexpat import features
import torch
from torch import nn, einsum
from einops import rearrange
import sys
import timm
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
import collections
import torch.utils.model_zoo as model_zoo

sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')
from Models.Decoders import UnetDecodingBlock_M,ResidualDecodingBlock,MLPDecoder
from Models.CNN.ResNet import resnet34
from Utils._deeplab import ASPP

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x, size=None):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class AdapterDWCNN(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Sequential(
                nn.Conv2d(D_features,D_features,3,1,1,groups=D_features,bias=True),
                nn.Conv2d(D_features,D_hidden_features,1,1,0,bias=True),
        )
        self.D_fc2 = nn.Sequential(
                nn.Conv2d(D_hidden_features,D_hidden_features,3,1,1,groups=D_hidden_features,bias=True),
                nn.Conv2d(D_hidden_features,D_features,1,1,0,bias=True),
        )
        # self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        # self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x, size=None):
        H,W = size
        B,L,C = x.shape
        xs = rearrange(x, 'b (h w) c -> b c h w', h=H,w=W).contiguous()
        xs = self.D_fc1(xs)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        xs = rearrange(xs, 'b c h w -> b (h w) c')
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class AdapterCNN(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Conv2d(D_features,D_hidden_features,3,1,1,bias=True)
        self.D_fc2 = nn.Conv2d(D_hidden_features,D_features,3,1,1,bias=True)
        # self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        # self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x, size=None):
        H,W = size
        B,L,C = x.shape
        xs = rearrange(x, 'b (h w) c -> b c h w', h=H,w=W).contiguous()
        xs = self.D_fc1(xs)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        xs = rearrange(xs, 'b c h w -> b (h w) c')
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


def load_pretrain(model, pre_s_dict):
    ''' Load state_dict in pre_model to model
    Solve the problem that model and pre_model have some different keys'''
    s_dict = model.state_dict()
    # use new dict to store states, record missing keys
    missing_keys = []
    new_state_dict = collections.OrderedDict()
    for key in s_dict.keys():
        if key in pre_s_dict.keys():
            new_state_dict[key] = pre_s_dict[key]
        else:
            new_state_dict[key] = s_dict[key]
            missing_keys.append(key)
    print('{} keys are not in the pretrain model:'.format(len(missing_keys)), missing_keys)
    # load new s_dict
    model.load_state_dict(new_state_dict)
    return model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        try:
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        except:
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, 
                 adapt_method=False, num_domains=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_domains = num_domains
        self.adapt_method = adapt_method
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)]) if num_domains>1 else norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)]) if num_domains>1 else norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if adapt_method=='MLP':
            self.adapter1 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
            self.adapter2 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
        elif adapt_method=='DWCNN':
            self.adapter1 = nn.ModuleList([AdapterDWCNN(dim,skip_connect=True) for _ in range(num_domains)])
            self.adapter2 = nn.ModuleList([AdapterDWCNN(dim,skip_connect=True) for _ in range(num_domains)])
        elif adapt_method=='CNN1':
            if self.input_resolution[0] > 14:
                self.adapter1 = nn.ModuleList([AdapterCNN(dim,skip_connect=True) for _ in range(num_domains)])
                self.adapter2 = nn.ModuleList([AdapterCNN(dim,skip_connect=True) for _ in range(num_domains)])
            else:
                self.adapter1 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
                self.adapter2 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
        elif adapt_method=='CNN2':
            # if self.input_resolution[0] > 7:
            #     self.adapter1 = nn.ModuleList([AdapterCNN(dim,skip_connect=True) for _ in range(num_domains)])
            # else:
            #     self.adapter1 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
            if self.shift_size > 0:
                self.adapter1 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
            else:
                self.adapter1 = nn.ModuleList([AdapterDWCNN(dim,skip_connect=True) for _ in range(num_domains)])
            self.adapter2 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])

        # else:
        #     self.adapter1 = nn.ModuleList([nn.Identity()])
        #     self.adapter2 = nn.ModuleList([nn.Identity()])

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x, d):
        H, W = self.input_resolution
        B, L, C = x.shape
        int_d = int(d)
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1[int_d](x) if self.num_domains>1 else self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # # TODO adapter local window
        if self.adapt_method:
            attn_windows = self.adapter1[int_d](attn_windows,(self.window_size,self.window_size))


        # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)

        # TODO global adapter
        # if self.adapt_method:
        #     attn_windows = self.adapter1[int_d](x,(H,W))

        x = shortcut + self.drop_path(x)

        # TODO FFN adapter
        xs = self.mlp(self.norm2[int_d](x) if self.num_domains>1 else self.norm2(x) )
        if self.adapt_method:
            xs = self.adapter2[int_d](xs,(H,W))
        x = x+self.drop_path(xs)
        # x = x + self.drop_path(self.adapter2[int_d](self.mlp(self.norm2[int_d](x) if self.num_domains>1 else self.norm2(x) )))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, num_domains=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.num_domains = num_domains
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.ModuleList([norm_layer(4 * dim) for _ in range(num_domains)]) if num_domains>1 else norm_layer(4 * dim)

    def forward(self, x, d):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        int_d = int(d)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm[int_d](x) if self.num_domains>1 else self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False,adapt_method=False, num_domains=1):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 adapt_method=adapt_method,
                                 num_domains=num_domains)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, num_domains=num_domains)
        else:
            self.downsample = None

    def forward(self, x, d):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, d)
        if self.downsample is not None:
            down_x = self.downsample(x,d)
        else:
            down_x = x
        return [x,down_x]

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, num_domains=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.num_domains = num_domains

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = nn.ModuleList([norm_layer(embed_dim) for _ in range(num_domains)]) if num_domains>1 else norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, d):
        B, C, H, W = x.shape
        int_d = int(d)
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm[int_d](x) if self.num_domains>1 else self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer_adapt(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, pretrained=None, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 adapt_method=None, num_domains=1, 
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.pretrained = pretrained

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, num_domains=num_domains)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,
                               adapt_method=adapt_method,num_domains=num_domains)
            self.layers.append(layer)
        
        # self.norm = norm_layer(self.num_features)

        # self.norm = nn.ModuleList([norm_layer(self.num_features) for _ in range(num_domains)])
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)


    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, d):
        x = self.patch_embed(x, d)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # for layer in self.layers:
        #     x = layer(x)

        # x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        # return x
        output = []
        for i,layer in enumerate(self.layers):
            before_down, x = layer(x, d)
            # if i==len(self.layers)-1:
            #     before_down = self.norm(before_down)
            B,HW,C = before_down.shape
            H,W = int(math.sqrt(HW)),int(math.sqrt(HW))
            out = rearrange(before_down, 'b (h w) c -> b c h w',h=H,w=W)
            output.append(out)
        return output

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops







class SwinSeg_adapt(nn.Module):
    '''
    use Swin Transformer as the encoder, CNN as the decoder
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_swin_name='swin_base_patch4_window7_224_in22k',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, fused_window_process=False,
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super(SwinSeg_adapt, self).__init__()
        self.num_stages = len(depths)
        embed_dims = [embed_dim*(2**i) for i in range(self.num_stages)]
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method

        self.encoder = SwinTransformer_adapt(
                pretrained=pretrained, img_size=img_size, patch_size=patch_size, 
                in_chans=in_chans, num_classes=num_classes,embed_dim=embed_dim, 
                depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint, fused_window_process=fused_window_process,
                 adapt_method=adapt_method, num_domains=num_domains)
        
        # bridge 
        self.bridge_conv1 = nn.Conv2d(embed_dims[3],embed_dims[3],kernel_size=1,stride=1, padding=1)
        self.bridge_norm1 = nn.ModuleList([conv_norm(embed_dims[3]) for _ in range(num_domains)])
        self.bridge_act1 = nn.ReLU(inplace=True)

        # decoder
        self.decoder1 = UnetDecodingBlock_M(embed_dims[3],embed_dims[3],conv_norm=conv_norm,num_domains=num_domains)  # 768,384
        self.decoder2 = UnetDecodingBlock_M(embed_dims[3],embed_dims[2],conv_norm=conv_norm,num_domains=num_domains)  # 384,192
        self.decoder3 = UnetDecodingBlock_M(embed_dims[2],embed_dims[1],conv_norm=conv_norm,num_domains=num_domains)   # 192,96
        self.decoder4 = UnetDecodingBlock_M(embed_dims[1],embed_dims[0],conv_norm=conv_norm,num_domains=num_domains)    # 96,48
        # self.decoder1 = ResidualDecodingBlock(embed_dims[3],embed_dims[3])  # 768,384
        # self.decoder2 = ResidualDecodingBlock(embed_dims[3],embed_dims[2])  # 384,192
        # self.decoder3 = ResidualDecodingBlock(embed_dims[2],embed_dims[1])   # 192,96
        # self.decoder4 = ResidualDecodingBlock(embed_dims[1],embed_dims[0])    # 96,48
        self.finalconv = nn.Sequential(
            nn.Conv2d(embed_dims[0], 1, kernel_size=1)
        )
        self.init_weights(pretrained_swin_name,pretrained_folder)
    
    def init_weights(self,pretrained_name,pretrained_folder):
        def _init_weights(m):
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
        
        if self.pretrained:
            self.apply(_init_weights)
            # pretrained_encoder = timm.create_model(pretrained_name,pretrained=True)
            pretrained_encoder_sd = torch.load(pretrained_folder+'/pretrained/{}.pth'.format(pretrained_name))
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            del pretrained_encoder_sd
            torch.cuda.empty_cache()
            print('loaded pretrained {} successfully'.format(pretrained_name))
        else:
            self.apply(_init_weights)
        
        cnn_set = set(['CNN1','CNN2'])
        for n, m in self.encoder.named_modules():
            if self.adapt_method=='MLP' and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method=='DWCNN' and 'adapter' in n and 'D_fc2.1' in n:
                if isinstance(m,nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method in cnn_set and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            
    
    def forward(self,x,d=None, out_feat=False, out_seg=True):
        if d == None:
            d = '0'
            print('No domain ID input')
        if self.debug:
            print('domain ID: {}'.format(d))
        img_size = x.size()[2:]
        B = x.shape[0]
        int_d = int(d)
        encoder_outs = self.encoder.forward_features(x, d)

        # bridge
        out = self.bridge_conv1(encoder_outs[3])
        out = self.bridge_norm1[int_d](out)
        out = self.bridge_act1(out)

        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # decoding
        out = self.decoder1(out, encoder_outs[3], d)  # (384,16,16)
        out = self.decoder2(out, encoder_outs[2], d)  # (192,32,32)
        out = self.decoder3(out, encoder_outs[1], d)  # (96,64,64)
        out = self.decoder4(out, encoder_outs[0], d)  # (48,128,128)
        
        # upsample
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        out = self.finalconv(out)  # (1,512,512)

        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}





class SwinSeg_CNNprompt_adapt(nn.Module):
    '''
    use Swin Transformer as the encoder, CNN as the decoder
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_swin_name='swin_base_patch4_window7_224_in22k',
        pretrained_cnn_name='resnet34',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, fused_window_process=False,
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.num_stages = len(depths)
        embed_dims = [embed_dim*(2**i) for i in range(self.num_stages)]
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method

        self.encoder = SwinTransformer_adapt(
                pretrained=pretrained, img_size=img_size, patch_size=patch_size, 
                in_chans=in_chans, num_classes=num_classes,embed_dim=embed_dim, 
                depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint, fused_window_process=fused_window_process,
                 adapt_method=adapt_method, num_domains=num_domains)
        
        self.prompt_encoder = resnet34(pretrained=False, out_indices=[1])
        # bridge 
        self.bridge_conv1 = nn.Conv2d(embed_dims[3],embed_dims[3],kernel_size=1,stride=1, padding=1)
        self.bridge_norm1 = nn.ModuleList([conv_norm(embed_dims[3]) for _ in range(num_domains)])
        self.bridge_act1 = nn.ReLU(inplace=True)

        # decoder
        self.decoder1 = UnetDecodingBlock_M(embed_dims[3],embed_dims[3],conv_norm=conv_norm,num_domains=num_domains)  # 768,384
        self.decoder2 = UnetDecodingBlock_M(embed_dims[3],embed_dims[2],conv_norm=conv_norm,num_domains=num_domains)  # 384,192
        self.decoder3 = UnetDecodingBlock_M(embed_dims[2],embed_dims[1],conv_norm=conv_norm,num_domains=num_domains)   # 192,96
        self.decoder4 = UnetDecodingBlock_M(embed_dims[1],embed_dims[0],conv_norm=conv_norm,num_domains=num_domains)    # 96,48

        # self.finalconv = nn.Sequential(
        #     nn.Conv2d(embed_dims[0], 1, kernel_size=1)
        # )
        self.finalconv = nn.Sequential(
            nn.Conv2d(embed_dims[0]+64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),    
            nn.Conv2d(64,1,1,1,0)       
        )
        self.init_weights(pretrained_swin_name,pretrained_cnn_name,pretrained_folder)
    
    def init_weights(self,pretrained_name,pretrained_cnn_name,pretrained_folder):
        def _init_weights(m):
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
        
        if self.pretrained:
            self.apply(_init_weights)
            # pretrained_encoder = timm.create_model(pretrained_name,pretrained=True)
            pretrained_encoder_sd = torch.load(pretrained_folder+'/pretrained/{}.pth'.format(pretrained_name))
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load CNN
            pretrained_cnn_sd = model_zoo.load_url(model_urls[pretrained_cnn_name])
            self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
            del pretrained_cnn_sd
            del pretrained_encoder_sd
            torch.cuda.empty_cache()
            print('loaded pretrained {} and {} successfully'.format(pretrained_name,pretrained_cnn_name))
        else:
            self.apply(_init_weights)
        
        cnn_set = set(['CNN1','CNN2'])
        for n, m in self.encoder.named_modules():
            if self.adapt_method=='MLP' and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method=='DWCNN' and 'adapter' in n and 'D_fc2.1' in n:
                if isinstance(m,nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method in cnn_set and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            
    
    def forward(self,x,d=None, out_feat=False, out_seg=True):
        if d == None:
            d = '0'
            print('No domain ID input')
        if self.debug:
            print('domain ID: {}'.format(d))
        img_size = x.size()[2:]
        B = x.shape[0]
        int_d = int(d)
        encoder_outs = self.encoder.forward_features(x, d)

        # bridge
        out = self.bridge_conv1(encoder_outs[3])
        out = self.bridge_norm1[int_d](out)
        out = self.bridge_act1(out)
        # out = encoder_outs[3]

        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # decoding
        out = self.decoder1(out, encoder_outs[3], d)  # (384,16,16)
        out = self.decoder2(out, encoder_outs[2], d)  # (192,32,32)
        out = self.decoder3(out, encoder_outs[1], d)  # (96,64,64)
        out = self.decoder4(out, encoder_outs[0], d)  # (48,128,128)

        p0 = self.prompt_encoder(x)[0]
        out = torch.cat((p0,out),dim=1)
        
        # upsample
        out = self.finalconv(out)  # (1,512,512)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        

        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


class SwinSimpleSeg_adapt(nn.Module):
    '''
    use Swin Transformer as the encoder, ASPP+upsample as decoder
    use domain-specific adapters and norms
    depths=[2, 2, 18, 2]  num_heads=[4, 8, 16, 32]
    '''
    def __init__(
        self,pretrained=None, pretrained_swin_name='swin_base_patch4_window7_224_in22k',
        pretrained_cnn_name=None,
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 18], num_heads=[4, 8, 16],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, fused_window_process=False,
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.num_stages = len(depths)
        embed_dims = [embed_dim*(2**i) for i in range(self.num_stages)]
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method

        self.encoder = SwinTransformer_adapt(
                pretrained=pretrained, img_size=img_size, patch_size=patch_size, 
                in_chans=in_chans, num_classes=num_classes,embed_dim=embed_dim, 
                depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint, fused_window_process=fused_window_process,
                 adapt_method=adapt_method, num_domains=num_domains)
        
        self.aspp = ASPP(in_channels=embed_dims[-1],atrous_rates=[6,12,18])
 

        # self.finalconv = nn.Sequential(
        #     nn.Conv2d(embed_dims[0], 1, kernel_size=1)
        # )
        self.finalconv = nn.Sequential(
            nn.Conv2d(256,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),    
            nn.Conv2d(64,1,1,1,0)       
        )
        self.init_weights(pretrained_swin_name,pretrained_cnn_name,pretrained_folder)
    
    def init_weights(self,pretrained_name,pretrained_cnn_name,pretrained_folder):
        def _init_weights(m):
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
        
        if self.pretrained:
            self.apply(_init_weights)
            # pretrained_encoder = timm.create_model(pretrained_name,pretrained=True)
            pretrained_encoder_sd = torch.load(pretrained_folder+'/pretrained/{}.pth'.format(pretrained_name))
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load CNN
            if pretrained_cnn_name:
                pretrained_cnn_sd = model_zoo.load_url(model_urls[pretrained_cnn_name])
                self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
                del pretrained_cnn_sd
            del pretrained_encoder_sd
            torch.cuda.empty_cache()
            print('loaded pretrained {} and {} successfully'.format(pretrained_name,pretrained_cnn_name))
        else:
            self.apply(_init_weights)
        
        cnn_set = set(['CNN1','CNN2'])
        for n, m in self.encoder.named_modules():
            if self.adapt_method=='MLP' and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method=='DWCNN' and 'adapter' in n and 'D_fc2.1' in n:
                if isinstance(m,nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method in cnn_set and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            
    
    def forward(self,x,d=None, out_feat=False, out_seg=True):
        if d == None:
            d = '0'
            print('No domain ID input')
        if self.debug:
            print('domain ID: {}'.format(d))
        img_size = x.size()[2:]
        B = x.shape[0]
        int_d = int(d)
        encoder_outs = self.encoder.forward_features(x, d)


        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # decoding
        out = self.aspp(encoder_outs[-1])
        
        # upsample
        out = self.finalconv(out)  # (1,512,512)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        

        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


class SwinSimpleSeg_CNNprompt_adapt(nn.Module):
    '''
    use Swin Transformer as the encoder, ASPP+upsample as decoder
    use domain-specific adapters and norms
    depths=[2, 2, 18, 2]  num_heads=[4, 8, 16, 32]
    '''
    def __init__(
        self,pretrained=None, pretrained_swin_name='swin_base_patch4_window7_224_in22k',
        pretrained_cnn_name='resnet34',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 18], num_heads=[4, 8, 16],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, fused_window_process=False,
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.num_stages = len(depths)
        embed_dims = [embed_dim*(2**i) for i in range(self.num_stages)]
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method

        self.encoder = SwinTransformer_adapt(
                pretrained=pretrained, img_size=img_size, patch_size=patch_size, 
                in_chans=in_chans, num_classes=num_classes,embed_dim=embed_dim, 
                depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint, fused_window_process=fused_window_process,
                 adapt_method=adapt_method, num_domains=num_domains)
        
        self.prompt_encoder = resnet34(pretrained=False, out_indices=[1])
        self.aspp = ASPP(in_channels=embed_dims[-1],atrous_rates=[6,12,18])
 
        # self.finalconv = nn.Sequential(
        #     nn.Conv2d(embed_dims[0], 1, kernel_size=1)
        # )
        self.finalconv = nn.Sequential(
            nn.Conv2d(256+64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),    
            nn.Conv2d(64,1,1,1,0)       
        )
        self.init_weights(pretrained_swin_name,pretrained_cnn_name,pretrained_folder)
    
    def init_weights(self,pretrained_name,pretrained_cnn_name,pretrained_folder):
        def _init_weights(m):
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
        
        if self.pretrained:
            self.apply(_init_weights)
            # pretrained_encoder = timm.create_model(pretrained_name,pretrained=True)
            pretrained_encoder_sd = torch.load(pretrained_folder+'/pretrained/{}.pth'.format(pretrained_name))
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load CNN
            if pretrained_cnn_name:
                # pretrained_cnn_sd = model_zoo.load_url(model_urls[pretrained_cnn_name])
                pretrained_cnn_sd = torch.load(pretrained_folder+'/pretrained/resnet34-333f7ec4.pth')
                self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
                del pretrained_cnn_sd
            del pretrained_encoder_sd
            torch.cuda.empty_cache()
            print('loaded pretrained {} and {} successfully'.format(pretrained_name,pretrained_cnn_name))
        else:
            self.apply(_init_weights)
        
        cnn_set = set(['CNN1','CNN2'])
        for n, m in self.encoder.named_modules():
            if self.adapt_method=='MLP' and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method=='DWCNN' and 'adapter' in n and 'D_fc2.1' in n:
                if isinstance(m,nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method in cnn_set and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            
    
    def forward(self,x,d=None, out_feat=False, out_seg=True):
        if d == None:
            d = '0'
            print('No domain ID input')
        if self.debug:
            print('domain ID: {}'.format(d))
        img_size = x.size()[2:]
        B = x.shape[0]
        int_d = int(d)
        encoder_outs = self.encoder.forward_features(x, d)


        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # decoding
        out = self.aspp(encoder_outs[-1])
        p0 = self.prompt_encoder(x)[0]
        out = nn.functional.interpolate(out,size = p0.shape[2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((p0,out),dim=1)
        
        # upsample
        out = self.finalconv(out)  # (1,512,512)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        

        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


class SwinFormer_adapt(nn.Module):
    '''
    use Swin Transformer as the encoder, ASPP+upsample as decoder
    use domain-specific adapters and norms
    depths=[2, 2, 18, 2]  num_heads=[4, 8, 16, 32]
    '''
    def __init__(
        self,pretrained=None, pretrained_swin_name='swin_base_patch4_window7_224_in22k',
        pretrained_cnn_name=None,
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, fused_window_process=False,
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.num_stages = len(depths)
        embed_dims = [embed_dim*(2**i) for i in range(self.num_stages)]
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method

        self.encoder = SwinTransformer_adapt(
                pretrained=pretrained, img_size=img_size, patch_size=patch_size, 
                in_chans=in_chans, num_classes=num_classes,embed_dim=embed_dim, 
                depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint, fused_window_process=fused_window_process,
                 adapt_method=adapt_method, num_domains=num_domains)
        
        self.decoder = MLPDecoder(embed_dims,1,512)
 
        
        # self.finalconv = nn.Sequential(
        #     nn.Conv2d(256,64,3,1,1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64,64,3,1,1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),    
        #     nn.Conv2d(64,1,1,1,0)       
        # )
        self.init_weights(pretrained_swin_name,pretrained_cnn_name,pretrained_folder)
    
    def init_weights(self,pretrained_name,pretrained_cnn_name,pretrained_folder):
        def _init_weights(m):
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
        
        if self.pretrained:
            self.apply(_init_weights)
            # pretrained_encoder = timm.create_model(pretrained_name,pretrained=True)
            pretrained_encoder_sd = torch.load(pretrained_folder+'/pretrained/{}.pth'.format(pretrained_name))
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load CNN
            if pretrained_cnn_name:
                pretrained_cnn_sd = model_zoo.load_url(model_urls[pretrained_cnn_name])
                self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
                del pretrained_cnn_sd
            del pretrained_encoder_sd
            torch.cuda.empty_cache()
            print('loaded pretrained {} and {} successfully'.format(pretrained_name,pretrained_cnn_name))
        else:
            self.apply(_init_weights)
        
        cnn_set = set(['CNN1','CNN2'])
        for n, m in self.encoder.named_modules():
            if self.adapt_method=='MLP' and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method=='DWCNN' and 'adapter' in n and 'D_fc2.1' in n:
                if isinstance(m,nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method in cnn_set and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            
    
    def forward(self,x,d=None, out_feat=False, out_seg=True):
        if d == None:
            d = '0'
            print('No domain ID input')
        if self.debug:
            print('domain ID: {}'.format(d))
        img_size = x.size()[2:]
        B = x.shape[0]
        int_d = int(d)
        encoder_outs = self.encoder.forward_features(x, d)


        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # decoding
        out = self.decoder(encoder_outs,img_size)

        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


class SwinFormer_CNNprompt_adapt(nn.Module):
    '''
    use Swin Transformer as the encoder, ASPP+upsample as decoder
    use domain-specific adapters and norms
    depths=[2, 2, 18, 2]  num_heads=[4, 8, 16, 32]
    '''
    def __init__(
        self,pretrained=None, pretrained_swin_name='swin_base_patch4_window7_224_in22k',
        pretrained_cnn_name='resnet34',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, fused_window_process=False,
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.num_stages = len(depths)
        embed_dims = [embed_dim*(2**i) for i in range(self.num_stages)]
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method

        self.encoder = SwinTransformer_adapt(
                pretrained=pretrained, img_size=img_size, patch_size=patch_size, 
                in_chans=in_chans, num_classes=num_classes,embed_dim=embed_dim, 
                depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint, fused_window_process=fused_window_process,
                 adapt_method=adapt_method, num_domains=num_domains)
        
        self.prompt_encoder = resnet34(pretrained=False, out_indices=[1])
        self.decoder = MLPDecoder(embed_dims,512,512)
 
        # self.finalconv = nn.Sequential(
        #     nn.Conv2d(embed_dims[0], 1, kernel_size=1)
        # )
        self.finalconv = nn.Sequential(
            nn.Conv2d(512+64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),    
            nn.Conv2d(128,1,1,1,0)       
        )
        self.init_weights(pretrained_swin_name,pretrained_cnn_name,pretrained_folder)
    
    def init_weights(self,pretrained_name,pretrained_cnn_name,pretrained_folder):
        def _init_weights(m):
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
        
        if self.pretrained:
            self.apply(_init_weights)
            # pretrained_encoder = timm.create_model(pretrained_name,pretrained=True)
            pretrained_encoder_sd = torch.load(pretrained_folder+'/pretrained/{}.pth'.format(pretrained_name))
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load CNN
            if pretrained_cnn_name:
                pretrained_cnn_sd = model_zoo.load_url(model_urls[pretrained_cnn_name])
                self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
                del pretrained_cnn_sd
            del pretrained_encoder_sd
            torch.cuda.empty_cache()
            print('loaded pretrained {} and {} successfully'.format(pretrained_name,pretrained_cnn_name))
        else:
            self.apply(_init_weights)
        
        cnn_set = set(['CNN1','CNN2'])
        for n, m in self.encoder.named_modules():
            if self.adapt_method=='MLP' and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method=='DWCNN' and 'adapter' in n and 'D_fc2.1' in n:
                if isinstance(m,nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            elif self.adapt_method in cnn_set and 'adapter' in n and 'D_fc2' in n:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            
    
    def forward(self,x,d=None, out_feat=False, out_seg=True):
        if d == None:
            d = '0'
            print('No domain ID input')
        if self.debug:
            print('domain ID: {}'.format(d))
        img_size = x.size()[2:]
        B = x.shape[0]
        int_d = int(d)
        encoder_outs = self.encoder.forward_features(x, d)


        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # decoding
        
        p0 = self.prompt_encoder(x)[0]
        out = self.decoder(encoder_outs,p0.shape[2:])
        out = torch.cat((p0,out),dim=1)
        
        # upsample
        out = self.finalconv(out)  # (1,512,512)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        

        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


if __name__ == '__main__':
    model = SwinSimpleSeg_adapt(pretrained=True,adapt_method='MLP',num_domains=1)
    x = torch.randn(3,3,224,224)
    d = '0'
    outs = model.forward(x,d)
    print(outs['seg'].shape)

    # for n, m in model.encoder.named_modules():
    #     print(n)
    # for y in outs:
    #     print(y.shape)
    # print(model.patches_resolution)

    ## freeze some parameters
    for name, param in model.encoder.named_parameters():
        # print(name)
        if 'adapter' not in name and 'norm' not in name:
            param.requires_grad = False 
            # print(name)
        # else:
        #     print(name)
    

    # AdapterDWConv
    # model = AdapterCNN(96)
    # x = torch.randn(5,3136,96)
    # y = model(x,(56,56))
    # print(y.shape)

    param = sum(p.numel() for p in model.parameters())
    print(f"number of parameter: {param/1e6} M")
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable parameter: {param/1e6} M")