'''
UNet architecture: Factorized attention Transformer encoder, CNN decoder
Encoder is from MPViT
This file implements AIM adapters
'''

import math
from pyexpat import features
from sklearn.metrics import f1_score
import torch
from torch import nn, einsum
from einops import rearrange
import sys
from typing import Tuple
from functools import partial
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')
from Models.Decoders import UnetDecodingBlockTransformer_M
from Models.Transformer.mpvit import FactorAtt_ConvRelPosEnc, ConvRelPosEnc, ConvPosEnc, Mlp, Conv2d_BN


class Conv2d_BN_M(nn.Module):
    """Convolution with BN module.
    different domains use different norm"""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=None,
        num_domains=1,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        # self.bn = norm_layer(out_ch)
        self.bns = nn.ModuleList([norm_layer(out_ch) for _ in range(num_domains)])
        for bn in self.bns:
            torch.nn.init.constant_(bn.weight, bn_weight_init)
            torch.nn.init.constant_(bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x, d=None):
        """foward function"""
        d = int(d)
        x = self.conv(x)
        x = self.bns[d](x)
        x = self.act_layer(x)

        return x



class DWConv2d_BN_M(nn.Module):
    """Depthwise Separable Convolution with BN module.
    Modify on MPViT DWConv2d_BN, this is for input output are different channel dim
    different domains use different BN"""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
        num_domains = 1,
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
        self.bns = nn.ModuleList([norm_layer(out_ch) for _ in range(num_domains)])
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for bn in self.bns:
            torch.nn.init.constant_(bn.weight, bn_weight_init)
            torch.nn.init.constant_(bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, d=None):
        """
        foward function
        """
        d = int(d)
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bns[d](x)
        x = self.act(x)

        return x



class DWCPatchEmbed_M(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding. The same as the module in MPViT
    different domains use different norm"""
    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 conv_norm=nn.BatchNorm2d,
                 act_layer=nn.Hardswish,
                 num_domains=1):
        super().__init__()

        self.patch_conv = DWConv2d_BN_M(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            norm_layer=conv_norm,
            act_layer=act_layer,
            num_domains=num_domains,)

    def forward(self, x, d=None):
        """foward function"""
        x = self.patch_conv(x, d)
        return x


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
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
        self.norm1s = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)])
        self.adapt_method = adapt_method

        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2s = nn.ModuleList([norm_layer(dim) for _ in range(num_domains)])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # adapter
        self.adapter1s = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])
        self.adapter2s = nn.ModuleList([Adapter(dim,skip_connect=True) for _ in range(num_domains)])


    def forward(self, x, size: Tuple[int, int], domain_label=None, d=None):
        # Conv-Attention.
        d = int(d)
        x = self.cpe(x, size)
        cur = self.norm1s[d](x)
        cur = self.factoratt_crpe(cur, size)
        # 1st adaptation
        cur = self.adapter1s[d](cur)
        x = x + self.drop_path(cur) 

        # MLP. 
        cur = self.norm2s[d](x)
        cur = self.mlp(cur)
        # 2rd adaptation
        cur = self.adapter2s[d](cur)
        x = x + self.drop_path(cur)

        return x



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
            input = blk(input, size=(H,W),d=d) if domain_label==None else blk(input, (H,W), domain_label,d=d)
        return input




# use different norms for different domains
class FATNet_adapt_M(nn.Module):
    '''
    A Conv Position encoding + Factorized attention Transformer
    use transformer encoder and decoder
    use domain-specific adapters and norms
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
        do_detach = False,
        decoder_name = 'MLP',
        **kwargs,
    ):
        super(FATNet_adapt_M, self).__init__()
        self.num_stages = num_stages
        self.do_detach = do_detach
        self.decoder_name = decoder_name


        self.stem_1 = Conv2d_BN_M(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                num_domains=num_domains)
        self.stem_2 = Conv2d_BN_M(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                num_domains=num_domains)

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList([
            DWCPatchEmbed_M(
                in_chans=embed_dims[idx] if idx==0 else embed_dims[idx-1],
                embed_dim=embed_dims[idx],
                patch_size=3,
                stride=1 if idx==0 else 2, 
                conv_norm=conv_norm,
                num_domains=num_domains
            ) for idx in range(self.num_stages)])

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
            ) for idx in range(self.num_stages)])


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
                ))

        self.decoder1 = UnetDecodingBlockTransformer_M(embed_dims[3]*2,embed_dims[3],self.mhsa_list[3],conv_norm=conv_norm,num_domains=num_domains)  # 768,384
        self.decoder2 = UnetDecodingBlockTransformer_M(embed_dims[3],embed_dims[2],self.mhsa_list[2],conv_norm=conv_norm,num_domains=num_domains)  # 384,192
        self.decoder3 = UnetDecodingBlockTransformer_M(embed_dims[2],embed_dims[1],self.mhsa_list[1],conv_norm=conv_norm,num_domains=num_domains)   # 192,96
        self.decoder4 = UnetDecodingBlockTransformer_M(embed_dims[1],embed_dims[0],self.mhsa_list[0],conv_norm=conv_norm,num_domains=num_domains)    # 96,48
        self.finalconv = nn.Sequential(nn.Conv2d(embed_dims[0], 1, kernel_size=1))    

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
            return {'seg': None, 'feat': x}

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
        decoder_outs = []
        # decoder_outs.append(out.detach())
        decoder_outs.append(out)
        
        # upsample
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        out = self.finalconv(out)  # (1,512,512)     

        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}




if __name__ == '__main__':
    # x = torch.randn(5,3,256,256)
    model = FATNet_adapt_M(num_domains=4)

    # y = model(x, d='1', out_feat=False) # d='2', out_feat=True
    # print(y['seg'].shape)

    # # from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    # # # flops = FlopCountAnalysis(model, x)
    # param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of parameter: {param/1e6} M")
    # # acts = ActivationCountAnalysis(model, x)

    # # print(f"total flops : {flops.total()/1e12} M")
    # # print(f"total activations: {acts.total()/1e6} M")
    # 
    
    count = 0
    # for name, params in model.named_parameters():
        # print(name)
    #     if 'debranch' not in name:
    #         count += params.numel()
    # print(f'number of params in debranches: {count/1e6} M')



