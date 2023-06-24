import torch
from torch import nn
from typing import Type, Any, Callable, Union, List, Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys
import math
from timm.models.layers import DropPath, trunc_normal_

sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')
from Models.Decoders import DeepLabV3Decoder


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, domain_label=None, **kwargs):
        return self.fn(self.norm(x), **kwargs) if domain_label==None else self.fn(self.norm(x),domain_label,**kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    '''
    input x, convert to q k v, do softmax(qk/\sqrt(k))v, output the same size as x
    dim: x dimension
    dim_head: q k v dimenstion
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # (dim_k)^{-0.5}

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1) # [(b,n,inner_dim),same,same]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b,h,n,n)

        attn = self.attend(dots)

        out = torch.matmul(attn, v) # (b,h,n,d)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class AttentionSup(nn.Module):
    '''
    input x, convert to q k v, do softmax(qk/\sqrt(k))v, output the same size as x
    dim: x dimension
    dim_head: q k v dimenstion
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., 
                num_domains=4, r=2):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        hidden_dim = max(int(dim_head/r),4)

        self.heads = heads
        self.scale = dim_head ** -0.5  # (dim_k)^{-0.5}

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # TODO for adapt method
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.transform = nn.Sequential(nn.Conv1d(dim_head, hidden_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            )
        self.fc_select = nn.Conv1d(hidden_dim,inner_dim,kernel_size=1,bias=False)

        self.domain_layer = nn.Sequential(
            nn.Linear(num_domains, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,inner_dim),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, domain_label):
        qkv = self.to_qkv(x).chunk(3, dim = -1) # [(b,n,inner_dim),same,same]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b,h,n,n)

        attn = self.attend(dots)

        out = torch.matmul(attn, v) # (b,h,n,k)
        
        # TODO for domain attention
        domain_att = self.domain_layer(domain_label).unsqueeze(2)  # (B,H*K,1)
        domain_att = rearrange(domain_att, 'b (h k) c -> b h c k', h=self.heads).contiguous()  # (b,h,1,k)
        domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        out = domain_att*out   # (B,H,N,dim)
        # domain_att = torch.sum(out,dim=1,keepdim=False)  # (b,n,k)
        # domain_att = rearrange(domain_att, 'b n k -> b k n')  # (b,k,n)
        # domain_att = self.average_pool(domain_att)   # (B,k,1)
        # domain_att = self.transform(domain_att)  # (B,hidden,1)
        # domain_label_up = self.domain_layer(domain_label).unsqueeze(2)  # (B,hidden,1)
        # domain_att = domain_att+domain_label_up   # (B,hidden,1)
        # domain_att = self.fc_select(domain_att)  # (B,h*k,1)
        # domain_att = rearrange(domain_att, 'b (h k) c -> b h c k', h=self.heads).contiguous()  # (b,h,1,k)
        # domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        # out = domain_att*out  # (B,H,N,dim)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class AttentionSupSK(nn.Module):
    '''
    input x, convert to q k v, do softmax(qk/\sqrt(k))v, output the same size as x
    dim: x dimension
    dim_head: q k v dimenstion
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., 
                num_domains=4, r=2):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        hidden_dim = max(int(dim_head/r),4)

        self.heads = heads
        self.scale = dim_head ** -0.5  # (dim_k)^{-0.5}

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # TODO for adapt method
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.transform = nn.Sequential(nn.Conv1d(dim_head, hidden_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            )
        self.fc_select = nn.Conv1d(hidden_dim,inner_dim,kernel_size=1,bias=False)

        self.domain_layer = nn.Sequential(
            nn.Linear(num_domains, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, domain_label):
        qkv = self.to_qkv(x).chunk(3, dim = -1) # [(b,n,inner_dim),same,same]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b,h,n,n)

        attn = self.attend(dots)

        out = torch.matmul(attn, v) # (b,h,n,k)
        
        # TODO for domain attention
        domain_att = torch.sum(out,dim=1,keepdim=False)  # (b,n,k)
        domain_att = rearrange(domain_att, 'b n k -> b k n')  # (b,k,n)
        domain_att = self.average_pool(domain_att)   # (B,k,1)
        domain_att = self.transform(domain_att)  # (B,hidden,1)
        domain_label_up = self.domain_layer(domain_label).unsqueeze(2)  # (B,hidden,1)
        domain_att = domain_att+domain_label_up   # (B,hidden,1)
        domain_att = self.fc_select(domain_att)  # (B,h*k,1)
        domain_att = rearrange(domain_att, 'b (h k) c -> b h c k', h=self.heads).contiguous()  # (b,h,1,k)
        domain_att = torch.softmax(domain_att, dim=1)   # (b,h,1,k)
        out = domain_att*out  # (B,H,N,dim)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    '''
    dim: input x dim
    depth: num of transformer blocks
    '''
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., adapt_method=None, num_domains=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if adapt_method == 'SupSK':
                attention_layer = AttentionSupSK(dim,heads=heads,dim_head=dim_head,dropout=dropout,num_domains=num_domains)
            elif adapt_method == 'Sup':
                attention_layer = AttentionSup(dim,heads,dim_head,dropout,num_domains)
            else:
                attention_layer = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attention_layer),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, domain_label=None):
        # print(domain_label)
        for attn, ff in self.layers:
            if domain_label==None:
                x = attn(x) + x 
            else: x = attn(x,domain_label)+x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    '''
    use: classification
    dim: transformer layer input dim
    mlp_dim: feed forward hidden dim
    '''
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



class ViTSeg(nn.Module):
    '''
    use: classification
    dim: transformer layer input dim
    mlp_dim: feed forward hidden dim
    '''
    def __init__(self, *, img_size=256, in_channel=3, out_channel=1, patch_size=16, dim=256, depth=8, heads=8, mlp_dim=1024, dim_head = 64, drop_rate = 0.1, emb_dropout = 0.):
        super(ViTSeg, self).__init__()
        image_height, image_width = pair(img_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        patch_dim = in_channel * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, drop_rate)

        self.trans_out_conv = nn.Conv2d(in_channels=dim,
                                        out_channels=512,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=True)
        
        self.decoder = DeepLabV3Decoder(in_channel=512, out_channel=out_channel)

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

    def forward(self, x):
        img_size = x.shape[-2:]
        x = self.to_patch_embedding(x)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=img_size[0]//self.patch_height,w=img_size[1]//self.patch_width).contiguous()

        out = self.trans_out_conv(x)
        out = self.decoder(out, img_size=img_size)
        # out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False)
        
        return out


class ViTSeg_adapt(nn.Module):
    '''
    use: classification
    dim: transformer layer input dim
    mlp_dim: feed forward hidden dim
    '''
    def __init__(self, *, img_size=256, in_channel=3, out_channel=1, patch_size=16, dim=256, depth=8, heads=8, mlp_dim=1024, dim_head = 64, drop_rate = 0.1, emb_dropout = 0.,
                adapt_method=None, num_domains=4):
        super(ViTSeg_adapt, self).__init__()
        image_height, image_width = pair(img_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        patch_dim = in_channel * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, drop_rate, adapt_method, num_domains)

        self.trans_out_conv = nn.Conv2d(in_channels=dim,
                                        out_channels=512,
                                        kernel_size=(3, 3),
                                        stride=(1, 1),
                                        padding=(1, 1),
                                        bias=True)
        
        self.decoder = DeepLabV3Decoder(in_channel=512, out_channel=out_channel)

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

    def forward(self, x, domain_label):
        img_size = x.shape[-2:]
        x = self.to_patch_embedding(x)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x, domain_label)
        x = rearrange(x, 'b (h w) c -> b c h w', h=img_size[0]//self.patch_height,w=img_size[1]//self.patch_width).contiguous()

        out = self.trans_out_conv(x)
        out = self.decoder(out, img_size=img_size)
        # out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False)
        
        return out


if __name__ == '__main__':
    x = torch.randn(6,3,256,256)
    domain_label = torch.randn(6,4)
    # model = ViTSeg(img_size=256, out_channel=1, patch_size=16, dim=256, depth=8, heads=8, mlp_dim=1024) # 2m
    model = ViTSeg_adapt(img_size=256, out_channel=1, patch_size=16, dim=256, depth=8, heads=8, mlp_dim=1024,
                adapt_method='Sup', num_domains=4)
    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    y = model(x,domain_label)
    # y = model(x)
    print(y.shape)
