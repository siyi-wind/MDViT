'''
built from https://github.com/ShoufaChen/AdaptFormer/tree/main/models
'''

import collections
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange,repeat
from functools import partial
import sys
import math
import torch.utils.model_zoo as model_zoo

sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')
from Utils._deeplab import ASPP
from Models.CNN.ResNet import resnet18,resnet34,resnet50
# from Models.Hybrid_models.TransFuseFolder.TransFuse import BiFusion_block,Up

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# class Adapter(nn.Module):
#     def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
#         super().__init__()
#         self.skip_connect = skip_connect
#         D_hidden_features = int(D_features * mlp_ratio)
#         self.act = act_layer()
#         self.D_fc1 = nn.Linear(D_features, D_hidden_features)
#         self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
#     def forward(self, x, size=None):
#         xs = self.D_fc1(x)
#         xs = self.act(xs)
#         xs = self.D_fc2(xs)
#         if self.skip_connect:
#             x = x + xs
#         else:
#             x = xs
#         return x

# Adapter(d_model=768,bottleneck=64,dropout=0.1,init_option='lora',
# adapter_scalar=0.1,adapter_layernorm_option='none')

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output




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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, k_dim, q_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = k_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.kv_proj = nn.Linear(k_dim,k_dim*2,bias=qkv_bias)
        self.q_proj = nn.Linear(q_dim,k_dim)
        self.proj = nn.Linear(k_dim, k_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # if self.with_qkv:
        #    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #    self.proj = nn.Linear(dim, dim)
        #    self.proj_drop = nn.Dropout(proj_drop)
        # self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k):
        B,N,K = k.shape
        kv = self.kv_proj(k).reshape(B,N,2,self.num_heads,K//self.num_heads).permute(2, 0, 3, 1, 4)  # 
        k,v = kv[0], kv[1]  # (B,H,N,C)
        q = self.q_proj(q).reshape(B,N,self.num_heads,K//self.num_heads).permute(0,2,1,3)  # (B,H,N,C)
        attn = (q @ k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, K)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class Block(nn.Module):
    def __init__(self, dim, num_frames, num_heads, mlp_ratio=4., scale=0.5, num_tadapter=1, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm,adapt_method=False,num_domains=1):
        super().__init__()
        self.scale = 0.5
        self.num_frames = num_frames
        self.num_tadapter = num_tadapter
        self.adapt_method = adapt_method
        self.num_domains = num_domains
        self.norm1 = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)]) if num_domains>1 else norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # self.MLP_Adapter = Adapter(dim, skip_connect=False)  # MLP-adapter, no skip connection
        # self.S_Adapter = Adapter(dim)  # with skip connection
        self.scale = scale
        # self.T_Adapter = Adapter(dim, skip_connect=False)  # no skip connection
        # if num_tadapter == 2:
        #     self.T_Adapter_in = Adapter(dim)

        if adapt_method == 'AdaptFormer':
            self.adapter =  Adapter(d_model=768,bottleneck=64,dropout=0.1,init_option='lora',
                                                adapter_scalar=0.1,adapter_layernorm_option='none') 

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)]) if num_domains>1 else norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, d, size=None):
        ## x in shape [BT, HW+1, D]
        B,N,D = x.shape
        int_d = int(d)
        xs = self.attn(self.norm1[int_d](x) if self.num_domains>1 else self.norm1(x))
        x = x+self.drop_path(xs)
        
        # xs = self.mlp(self.norm2[int_d](x) if self.num_domains>1 else self.norm2(x))
        # xs = self.norm2[int_d](x) if self.num_domains>1 else self.norm2(x)
        adapt_x = self.adapter(x,add_residual=False)
        residual = x 
        x = self.drop_path(self.mlp(self.norm2(x)))
        x = x+residual+adapt_x

        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT_ImageNet(nn.Module):
    def __init__(self, img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=None, adapt_method=False, num_domains=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.pretrained = pretrained
        self.depth = depth
        self.num_frames = num_frames
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=patch_embedding_bias)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_frames=num_frames, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                adapt_method=adapt_method, num_domains=num_domains)
            for i in range(self.depth)])
        # self.ln_post = nn.ModuleList([norm_layer(embed_dim) for _ in range(num_domains)]) if self.num_domains>1 else norm_layer(embed_dim)
        # self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

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
        return {'pos_embed', 'temporal_embedding'}

    def forward_features(self, x, d=None):
        B,C,H,W = x.shape
        int_d = int(d)
        x = self.patch_embed(x) # (B,HW,D)
        x = torch.cat([self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],dim=1)
        x = x + self.pos_embed.to(x.dtype)

        for blk in self.blocks:
            x = blk(x,d=d,size=(self.img_size//self.patch_size,self.img_size//self.patch_size))
        
        return x[:,1:,:]
    
    def forward(x):
        return


# vit_base_patch16_224_in21k
# img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
# depth=12, num_heads=12, mlp_ratio=4., 
# patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
# drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),



class AdaptFormer(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        # decoder
        self.aspp = ASPP(in_channels=embed_dim,atrous_rates=[6,12,18])
        self.final_conv = nn.Conv2d(256, 1, kernel_size=1)

        self.init_weights(pretrained_vit_name,pretrained_folder)
    
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
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            del pretrained_encoder_sd
            torch.cuda.empty_cache()
            print('loaded pretrained {} successfully'.format(pretrained_name))
        else:
            self.apply(_init_weights)
        
        for n,m in self.encoder.named_modules():
            if 'adapter' in n and 'down_proj' in n:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight,a=math.sqrt(5))
                    nn.init.zeros_(m.bias)
                    # print('1: ', n)
            elif 'adapter' in n and 'up_proj' in n:
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                    # print('2: ', n)
            
    
    def forward(self,x,d=None, out_feat=False, out_seg=True):
        if d == None:
            d = '0'
            print('No domain ID input')
        if self.debug:
            print('domain ID: {}'.format(d))
        img_size = x.size()[2:]
        B = x.shape[0]
        int_d = int(d)
        encoder_out = self.encoder.forward_features(x, d)
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)

        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # decoding
        out = self.aspp(encoder_out)  # (B,C,H/16,W/16)
        
        # upsample
        out = self.final_conv(out)  # (1,H/16,W/16)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv2d(2,1,7,1,padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,g=None):
        # g is the feature to generate attention
        if g == None:
            g = x
        attn = torch.cat((torch.max(g,1)[0].unsqueeze(1), torch.mean(g,1).unsqueeze(1)), dim=1)
        attn = self.sigmoid(self.spatial(attn))
        return x*attn


class ChannelAttention(nn.Module):
    def __init__(self,embed):
        super().__init__()
        self.proj = nn.Sequential(
                    nn.Conv2d(embed,embed//2,1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed//2,embed,1)
        )
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self,x,g=None):
        if g==None:
            g = x
        attn = self.sigmoid(self.proj(self.maxpool(g))+self.proj(self.avgpool(g)))
        return x*attn


class ChannelSpatialAttention(nn.Module):
    def __init__(self,embed):
        super().__init__()
        self.channel_attn = ChannelAttention(embed)
        self.spatial_attn = SpatialAttention()
    
    def forward(self,x):
        x = self.channel_attn(x)+x
        x = self.spatial_attn(x)+x 
        return x


class ViTSeg_CNNprompt_adapt(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_cnn_name='resnet18',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super(ViTSeg_CNNprompt_adapt, self).__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet18(pretrained=False, out_indices=[1,2,3])
        # self.prompt_attn = CrossAttention(
        #    k_dim=embed_dim,q_dim=256, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.combination = nn.Conv2d(256+embed_dim,embed_dim,1,1,0)

        # decoder
        self.aspp = ASPP(in_channels=embed_dim,atrous_rates=[6,12,18])
        # self.final_conv = nn.Conv2d(256+64, 1, kernel_size=1)
        # C54-C57
        # self.final_conv = nn.Sequential(
        #                     nn.Conv2d(256+64,256,3,1,1),
        #                     nn.BatchNorm2d(256),
        #                     nn.ReLU(inplace=True),
        #                     nn.Conv2d(256,1,1,1,0)
        # )

        self.final_conv = nn.Sequential(
                            nn.Conv2d(256+64,64,3,1,1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64,64,3,1,1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64,1,1,1,0)
        )

        self.init_weights(pretrained_vit_name,pretrained_cnn_name,pretrained_folder)
    
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
            # load ViT
            pretrained_encoder_sd = torch.load(pretrained_folder+'/pretrained/{}.pth'.format(pretrained_name))
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # cross attention
        # p = self.prompt_encoder(x)[0]
        # p = rearrange(p, 'b c h w -> b (h w) c')
        # encoder_out = self.prompt_attn(q=p,k=encoder_out)
        # encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)

        # concat combination
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        p0,_,p1 = self.prompt_encoder(x)
        encoder_out = torch.cat((encoder_out,p1),dim=1)
        encoder_out =  self.combination(encoder_out)

        # decoding
        out = self.aspp(encoder_out)  # (B,C,H/16,W/16)

        # concat
        out = nn.functional.interpolate(out,size=p0.shape[-2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((out,p0),dim=1)
        
        # upsample
        out = self.final_conv(out)  # (1,H/16,W/16)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


class Residual(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,1,1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch,out_ch,1)
        else:
            self.skip = nn.Identity()
    def forward(self,x):
        res = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x+res 




class ViTSeg_Bifusion_adapt(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',pretrained_cnn_name='resnet18',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet18(pretrained=False, out_indices=[1,2,3])
        # self.prompt_attn = CrossAttention(
        #    k_dim=embed_dim,q_dim=256, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        # self.combination = nn.Conv2d(256+embed_dim,embed_dim,1,1,0)

        # decoder
        drop_rate = 0.2
        self.up1 = Up(in_ch1=768,out_ch=128)
        self.up2 = Up(in_ch1=128,out_ch=64)

        self.up_c = BiFusion_block(ch_1=256,ch_2=768,r_2=4,ch_int=256,ch_out=256,drop_rate=drop_rate/2)
        
        self.up_c_1_1 = BiFusion_block(ch_1=128,ch_2=128,r_2=2,ch_int=128,ch_out=128,drop_rate=drop_rate/2)
        self.up_c_1_2 = Up(in_ch1=256,out_ch=128,in_ch2=128,attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=64,ch_2=64,r_2=1,ch_int=64,ch_out=64,drop_rate=drop_rate/2)
        self.up_c_2_2 = Up(128,64,64,attn=True)

        self.drop = nn.Dropout2d(drop_rate/2)

        self.final_conv = nn.Sequential(
                            nn.Conv2d(64,64,3,1,1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64,1,1,1),
        )



        self.init_weights(pretrained_vit_name,pretrained_cnn_name,pretrained_folder)
    
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
            # load ViT
            pretrained_encoder_sd = torch.load(pretrained_folder+'/pretrained/{}.pth'.format(pretrained_name))
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)

            # load ResNet
            pretrained_cnn_sd = model_zoo.load_url(model_urls['resnet34'])
            self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
            del pretrained_encoder_sd
            del pretrained_cnn_sd
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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}


        # concat 3rd stage
        x_t_0 = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        x_t_0 = self.drop(x_t_0)
        x_t_1 = self.up1(x_t_0)
        x_t_1 = self.drop(x_t_1)
        x_t_2 = self.up2(x_t_1)
        x_t_2 = self.drop(x_t_2)

        p2,p1,p0 = self.prompt_encoder(x)

        x_c = self.up_c(p0,x_t_0)

        x_c_1_1 = self.up_c_1_1(x_t_1,p1)
        x_c_1 = self.up_c_1_2(x_c,x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_t_2,p2)
        x_c_2 = self.up_c_2_2(x_c_1,x_c_2_1)

        out = self.final_conv(x_c_2)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False)
    
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}



class ViTSeg_CNNprompt_adapt2(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet18(pretrained=True, out_indices=[1,2,3])
        # self.prompt_attn = CrossAttention(
        #    k_dim=embed_dim,q_dim=256, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        # self.combination = nn.Conv2d(256+embed_dim,embed_dim,1,1,0)

        # decoder
        self.aspp = ASPP(in_channels=embed_dim+256,atrous_rates=[6,12,18])
        self.combination_1 = Residual(384,128)
        self.combination_0 = Residual(192,64)
        self.final_conv = nn.Sequential(
                            nn.Conv2d(64,1,3,1,1),
        )

        # # attention
        # self.csattn_0 = ChannelSpatialAttention(embed_dim)
        # self.csattn_1 = ChannelSpatialAttention(256)
        # self.csattn_2 = ChannelSpatialAttention(64)


        self.init_weights(pretrained_vit_name,pretrained_folder)
    
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
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}


        # concat 3rd stage
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        p0,p1,p2 = self.prompt_encoder(x)
        encoder_out = torch.cat((encoder_out,p2),dim=1)
        out = self.aspp(encoder_out)  # (B,C,H/16,W/16)
        out = nn.functional.interpolate(out,size=p1.shape[-2:],mode = 'bilinear', align_corners=False)  # (B,C,H/8,W/8)

        # concat 2rd stage 
        out = torch.cat((out,p1),dim=1)
        out = self.combination_1(out)
        out = nn.functional.interpolate(out,size=p0.shape[-2:],mode = 'bilinear', align_corners=False) # (B,C,H/4,W/4)

        # concat 1st stage
        out = torch.cat((out,p0),dim=1)
        out = self.combination_0(out)  

        
        # upsample
        out = self.final_conv(out)  # (1,H/4,W/4)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}



class ViTSeg_CNNprompt_adapt3(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_cnn_name = 'resnet18',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet18(pretrained=False, out_indices=[1,2,3])

        # decoder
        self.aspp = ASPP(in_channels=embed_dim,atrous_rates=[6,12,18])
        self.combination_1 = Residual(512,256)
        self.down_channel_1 = nn.Conv2d(256,128,1)
        self.combination_0 = Residual(256,128)
        self.down_channel_0 = nn.Conv2d(128,64,1)
        self.final_conv = nn.Sequential(
                            nn.Conv2d(128,64,3,1,1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64,1,1,1),
        )


        self.init_weights(pretrained_vit_name,pretrained_cnn_name,pretrained_folder)
    
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
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load ResNet
            pretrained_cnn_sd = model_zoo.load_url(model_urls['resnet18'])
            self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
            del pretrained_encoder_sd
            del pretrained_cnn_sd
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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}


        # concat 3rd stage
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        out = self.aspp(encoder_out)
        p0,p1,p2 = self.prompt_encoder(x)
        out = torch.cat((out,p2),dim=1)
        out = nn.functional.interpolate(out,size=p1.shape[-2:],mode = 'bilinear', align_corners=False)  # (B,C,H/8,W/8)
        out = self.combination_1(out)  # (B,256,H/8,W/8)

        # concat 2rd stage 
        out = self.down_channel_1(out)
        out = torch.cat((out,p1),dim=1)  # (B,256,H/8,W/8)
        out = nn.functional.interpolate(out,size=p0.shape[-2:],mode = 'bilinear', align_corners=False) # (B,128,H/4,W/4)
        out = self.combination_0(out)  # (B,128,H/4,W/4)

        # concat 1st stage
        out = self.down_channel_0(out)
        out = torch.cat((out,p0),dim=1)  # (B,128,H/4,W/4)

        
        # upsample
        out = self.final_conv(out)  # (1,H/4,W/4)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}



class ViTSeg_CNNprompt_adapt4(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_cnn_name = 'resnet34',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet34(pretrained=False, out_indices=[1])

        # decoder
        # self.proj = nn.Conv2d(embed_dim,768,1,1,0)
        self.aspp = ASPP(in_channels=embed_dim,atrous_rates=[6,12,18])

        # for resnet34
        self.final_conv = nn.Sequential(
                            nn.Conv2d(256+64,64,3,1,1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64,64,3,1,1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64,1,1,1),
        )

        # self.proj_1 = nn.Conv2d(256,128,3,1,1)

        # self.final_conv = nn.Sequential(
        #                     nn.Conv2d(256+256,128,3,1,1),
        #                     nn.BatchNorm2d(128),
        #                     nn.ReLU(inplace=True),
        #                     nn.Conv2d(128,128,3,1,1),
        #                     nn.BatchNorm2d(128),
        #                     nn.ReLU(inplace=True),
        #                     nn.Conv2d(128,1,1,1),
        # )


        self.init_weights(pretrained_vit_name,pretrained_cnn_name,pretrained_folder)
    
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
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load ResNet
            # pretrained_cnn_sd = model_zoo.load_url(model_urls[pretrained_cnn_name])
            pretrained_cnn_sd = torch.load(pretrained_folder+'/pretrained/resnet34-333f7ec4.pth')
            self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
            del pretrained_encoder_sd
            del pretrained_cnn_sd
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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}


        # concat 3rd stage
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        out = self.aspp(encoder_out)
        p0 = self.prompt_encoder(x)[0]
        # p0 = self.proj_1(p0)
        out = nn.functional.interpolate(out,size=p0.size()[2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((p0,out),dim=1)


        
        # upsample
        out = self.final_conv(out)  # (1,H/4,W/4)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}



class ViTSeg_CNNprompt_adapt5(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_cnn_name = 'resnet34',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet34(pretrained=False, out_indices=[1,2,3])

        # decoder
        self.aspp = ASPP(in_channels=embed_dim,atrous_rates=[6,12,18])

        self.combination = nn.Sequential(
                            nn.Conv2d(512+128,128,3,1,1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128,128,3,1,1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
                            nn.Conv2d(128+64,64,3,1,1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64,64,3,1,1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64,1,1,1),
        )


        self.init_weights(pretrained_vit_name,pretrained_cnn_name,pretrained_folder)
    
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
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load ResNet
            pretrained_cnn_sd = model_zoo.load_url(model_urls['resnet34'])
            self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
            del pretrained_encoder_sd
            del pretrained_cnn_sd
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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}


        # concat 3rd stage
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        out = self.aspp(encoder_out)
        p0,p1 = self.prompt_encoder(x)
        out = nn.functional.interpolate(out,size=p1.size()[2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((p1,out),dim=1)

        out = self.combination(out)
        out = nn.functional.interpolate(out,size=p0.size()[2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((p0,out),dim=1)


        
        # upsample
        out = self.final_conv(out)  # (1,H/4,W/4)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


class ViTSeg_CNNprompt_adapt6(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_cnn_name = 'resnet34',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet34(pretrained=False, out_indices=[1,2])

        # decoder
        self.proj_1 = nn.Conv2d(768,512,1,1,0)
        

        self.combination_1 = Residual(512+128,256)

        self.combination_2 = Residual(256+64,128)

        self.final_conv = nn.Conv2d(128,1,1,1,0)


        self.init_weights(pretrained_vit_name,pretrained_cnn_name,pretrained_folder)
    
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
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load ResNet
            pretrained_cnn_sd = model_zoo.load_url(model_urls['resnet34'])
            self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
            del pretrained_encoder_sd
            del pretrained_cnn_sd
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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}


        # concat 3rd stage
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        out = self.proj_1(encoder_out)
        p0,p1 = self.prompt_encoder(x)
        out = nn.functional.interpolate(out,size=p1.size()[2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((p1,out),dim=1)

        out = self.combination_1(out)
        out = nn.functional.interpolate(out,size=p0.size()[2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((p0,out),dim=1)

        out = self.combination_2(out)

        # upsample
        out = self.final_conv(out)  # (1,H/4,W/4)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


class ViTSeg_CNNprompt_adapt7(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_cnn_name = 'resnet18',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet18(pretrained=False, out_indices=[1,2,3])

        # decoder
        self.proj_1 = nn.Conv2d(768,512,1,1,0)
        
        self.combination_1 = Residual(512+256,256)

        self.combination_2 = Residual(256+128,128)

        self.combination_3 = Residual(128+64,64)

        self.final_conv = nn.Conv2d(64,1,1,1,0)


        self.init_weights(pretrained_vit_name,pretrained_cnn_name,pretrained_folder)
    
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
            if pretrained_name == 'mae_pretrain_vit_base':
                pretrained_encoder_sd = pretrained_encoder_sd['model']
            self.encoder = load_pretrain(self.encoder,pretrained_encoder_sd)
            # load ResNet
            # pretrained_cnn_sd = model_zoo.load_url(model_urls[pretrained_cnn_name])
            pretrained_cnn_sd = torch.load(pretrained_folder+'/pretrained/resnet34-333f7ec4.pth')
            self.prompt_encoder = load_pretrain(self.prompt_encoder, pretrained_cnn_sd)
            del pretrained_encoder_sd
            del pretrained_cnn_sd
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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}


        # concat 3rd stage
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        out = self.proj_1(encoder_out)
        p0,p1,p2 = self.prompt_encoder(x)

        out = torch.cat((p2,out),dim=1)
        out = self.combination_1(out)

        out = nn.functional.interpolate(out,size=p1.size()[2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((p1,out),dim=1)

        out = self.combination_2(out)
        out = nn.functional.interpolate(out,size=p0.size()[2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((p0,out),dim=1)

        out = self.combination_3(out)

        # upsample
        out = self.final_conv(out)  # (1,H/4,W/4)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


if __name__ == '__main__':
    model = AdaptFormer(adapt_method='AdaptFormer',num_domains=1,pretrained=True)
    # model = resnet34(pretrained=False,out_indices=[1,2])
    # import timm
    # model_pre = timm.create_model('vit_base_patch16_224_in21k',pretrained=True)
    # model = load_pretrain(model,model_pre.state_dict())
    x = torch.randn(4,3,224,224)
    y = model(x,'0')
    print(y['seg'].shape)

    for name, param in model.encoder.named_parameters():
        # print(name)
        if 'adapter' not in name:
            param.requires_grad = False 
    # for name, param in model.prompt_encoder.named_parameters():
    #     # print(name)
    #     if 'norm' not in name:
    #         param.requires_grad = False 

    # model = ChannelAttention(256)
    # x = torch.randn(5,256,14,14)
    # g = torch.randn(5,256,14,14)
    # y = model(x=x)
    # print(y.shape)
    param = sum(p.numel() for p in model.parameters())
    print(f"number of parameter: {param/1e6} M")
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable parameter: {param/1e6} M")

    from thop import profile
    x = torch.randn(1,3,224,224)
    # model.eval()
    flops, params = profile(model, (x,))
    print(f"total flops : {flops/1e6} M")
