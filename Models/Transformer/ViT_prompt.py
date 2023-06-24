'''
built from https://github.com/taoyang1122/adapt-image-models/blob/main/mmaction/models/backbones/vit_imagenet.py
'''

import collections
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange,repeat
from functools import partial,reduce
from operator import mul
import sys
import math

sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')
from Utils._deeplab import ASPP
# from Models.CNN.ResNet import resnet18

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
        class_token = x[:,0,:].unsqueeze(1)
        xs = x[:,1:,:]
        xs = rearrange(xs, 'b (h w) c -> b c h w', h=H,w=W).contiguous()
        xs = self.D_fc1(xs)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        xs = rearrange(xs, 'b c h w -> b (h w) c')
        xs = torch.cat((class_token,xs),dim=1)
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


        # B, N, C = x.shape
        # if self.with_qkv:
        #    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #    q, k, v = qkv[0], qkv[1], qkv[2]
        # else:
        #    qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #    q, k, v  = qkv, qkv, qkv

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # if self.with_qkv:
        #    x = self.proj(x)
        #    x = self.proj_drop(x)
        # return x


class Block(nn.Module):
    def __init__(self, dim, num_frames, num_heads, mlp_ratio=4., scale=0.5, num_tadapter=1, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm,adapt_method=False,num_domains=1):
        super().__init__()
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

        if adapt_method == 'MLP':
            self.adapter1 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
            self.adapter2 = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
        elif adapt_method == 'DWCNN':
            self.adapter1 = nn.ModuleList([AdapterDWCNN(dim,skip_connect=True) for _ in range(num_domains)])
            self.adapter2 = nn.ModuleList([AdapterDWCNN(dim,skip_connect=True) for _ in range(num_domains)])

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
        if self.adapt_method:
            xs = self.adapter1[int_d](xs,size)
        x = x+self.drop_path(xs)
        
        xs = self.mlp(self.norm2[int_d](x) if self.num_domains>1 else self.norm2(x))
        if self.adapt_method:
            xs = self.adapter2[int_d](xs,size)
        x = x+self.drop_path(xs)
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


class ViT_ImageNet_prompt(nn.Module):
    def __init__(self, img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=None, adapt_method=False, num_domains=1,
                 prompt_len=5,prompt_drop_rate=0.1,):
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
        self.pos_drop = nn.Dropout(p=drop_rate)
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

        ## ------------------- prompt -----------------------
        # initialize prompt for every block
        self.prompt_len = prompt_len
        self.prompt_drop = DropPath(prompt_drop_rate)
        val = math.sqrt(6. / float(3 * reduce(mul, (patch_size,patch_size), 1) + embed_dim))
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
            self.depth, prompt_len, embed_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)


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

    def forward_features(self, x, d):
        B,C,H,W = x.shape
        int_d = int(d)
        x = self.patch_embed(x) # (B,HW,D)
        x = torch.cat([self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],dim=1)
        x = x + self.pos_embed.to(x.dtype)

        # insert prompt
        prompt = self.prompt_drop(self.deep_prompt_embeddings[0].expand(B,-1,-1))
        x = torch.cat((x[:,:1,:],prompt,x[:,1:,:]),dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x,d=d,size=(self.img_size//self.patch_size,self.img_size//self.patch_size))
            p_id = i+1
            if p_id < self.depth:
                prompt = self.prompt_drop(self.deep_prompt_embeddings[p_id].expand(B,-1,-1))
                x = torch.cat((x[:,:1,:],prompt,x[:,(1+self.prompt_len):,:]),dim=1)
        
        return x[:,(1+self.prompt_len):,:]
    
    def forward(x):
        return


# vit_base_patch16_224_in21k
# img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
# depth=12, num_heads=12, mlp_ratio=4., 
# patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
# drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),



class ViTSeg_prompt(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    follow VPT to use visual prompt
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        prompt_len=5,prompt_drop_rate=0.1,
        **kwargs,):
        super(ViTSeg_prompt, self).__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet_prompt(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains,
                    prompt_len=prompt_len,prompt_drop_rate=prompt_drop_rate)
        
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









if __name__ == '__main__':
    model = ViTSeg_prompt(adapt_method=False,num_domains=1,pretrained=True)
    # import timm
    # model_pre = timm.create_model('vit_base_patch16_224_in21k',pretrained=True)
    # model = load_pretrain(model,model_pre.state_dict())
    x = torch.randn(4,3,224,224)
    y = model(x,'0')
    print(y['seg'].shape)

    for name, param in model.encoder.named_parameters():
        param.requires_grad = False 
        # print(name)
        # if 'prompt' not in name:
        #     param.requires_grad = False 
        # else:
        #     print(name)
    # for name, param in model.prompt_encoder.named_parameters():
    #     # print(name)
    #     if 'norm' not in name:
    #         param.requires_grad = False 

    # model = CrossAttention(q_dim=256, k_dim=768)
    # k = torch.randn(5,196,768)
    # q = torch.randn(5,196,256)
    # y = model(q=q,k=k)
    # print(y.shape)
    param = sum(p.numel() for p in model.parameters())
    print(f"number of parameter: {param/1e6} M")
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable parameter: {param/1e6} M")