'''
UNet architecture: Factorized attention Transformer encoder, CNN decoder
Encoder is from MPViT
implement different adapters from other papers
(DASE) Domain-attentive SE adapters https://github.com/frank-xwang/towards-universal-object-detection/tree/cf40aed4c79b86b3e8e08e4adf94f43742693111
'''

import math
import torch
import torch.nn as nn
from einops import rearrange
import sys
from typing import Tuple
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')

from Models.Transformer.mpvit import FactorAtt_ConvRelPosEnc, ConvRelPosEnc, ConvPosEnc, Mlp, Conv2d_BN
from Models.Decoders import UnetDecodingBlock, UnetDecodingBlockTransformer, MLPDecoder
from Models.Sota_adapters.domain_attention_module import DomainAttention
from Models.Sota_adapters.se_module_vector import SELayer




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


class SerialBlock(nn.Module):
    """ Serial block class. For UFAT
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. 
        input: x (B,N,C), (H,W)  output: out (B,N,C)"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpe=None, shared_crpe=None):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe

        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Tuple[int, int]):
        # Conv-Attention.
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur) 

        # MLP. 
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x


class MHSA_stage(nn.Module):
    '''
    Multi-head self attention
    (B, N, C) --> (B, N, C)
    Combine several Serial blocks for a stage
    '''
    def __init__(self, dim, num_layers, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
        norm_layer=nn.LayerNorm, crpe_window={3:2, 5:3, 7:3}):
        super(MHSA_stage, self).__init__()

        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim//num_heads, h=num_heads, window=crpe_window)

        self.mhca_blks = nn.ModuleList(
            [SerialBlock(
                dim, num_heads, mlp_ratio, qkv_bias, qk_scale, 
                drop_rate, attn_drop_rate, drop_path_rate,
                nn.GELU, norm_layer, self.cpe, self.crpe
            ) for _ in range(num_layers)]
        )

    def forward(self, input, H, W):
        for blk in self.mhca_blks:
            input = blk(input, size=(H,W))
        return input



class FAT_Transformer(nn.Module):
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
        **kwargs,
    ):
        super(FAT_Transformer, self).__init__()
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
            MHSA_stage(
                embed_dims[idx],
                num_layers=num_layers[idx],
                num_heads=num_heads[idx], 
                mlp_ratio=mlp_ratios[idx], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer
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

    def forward(self, x):
        # x (B,in_chans,H,W)
        x = self.stem(x)  # (B,embed_dim[0],H/4,W/4)
        out = []
        for idx in range(self.num_stages):
            x = self.patch_embed_stages[idx](x)  # (B, embed_dim[idx],H/(4*2^idx),W/(4*2^idx))
            B,C,H,W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.mhsa_stages[idx](x, H, W)
            x = rearrange(x, 'b (h w) c -> b c h w', w=W, h=H).contiguous()
            out.append(x)
        
        return out


class UFAT(nn.Module):
    '''
    Unet architecture Factorized Transformer, used for segmentation
    tran_dim: dim between attention and mlp in transformer layer
    dim_head: dim in the attention
    '''
    def __init__(self, 
        image_size=512,
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
        ):
        super(UFAT, self).__init__()
        # encoder
        #    0         1        2        3
        # [(48,128),(96,64),(192,32),(384,16)]
        self.encoder = FAT_Transformer(image_size,in_chans,num_stages,num_layers,embed_dims,mlp_ratios,
                                        num_heads,qkv_bias,qk_scale,
                                        drop_rate,attn_drop_rate,drop_path_rate,norm_layer,nn.BatchNorm2d)

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


    def forward(self,x):
        # encoding
        #    0         1        2        3
        # [(48,128),(96,64),(192,32),(384,16)]
        encoder_outs = self.encoder(x) 

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

        return out



class FATSegmenter(nn.Module):
    '''
    Unet architecture Factorized Transformer as the encoder, use MLP or other decoders
    tran_dim: dim between attention and mlp in transformer layer
    dim_head: dim in the attention
    '''
    def __init__(self, 
        image_size=512,
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
        decoder_name = 'MLP',
        ):
        super(FATSegmenter, self).__init__()
        # encoder
        #    0         1        2        3
        # [(48,128),(96,64),(192,32),(384,16)]
        self.encoder = FAT_Transformer(image_size,in_chans,num_stages,num_layers,embed_dims,mlp_ratios,
                                        num_heads,qkv_bias,qk_scale,
                                        drop_rate,attn_drop_rate,drop_path_rate,norm_layer,nn.BatchNorm2d)
        
        self.decoder = MLPDecoder(embed_dims, 1, 512)

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


    def forward(self,x,out_feat=False,out_seg=True):
        img_size = x.size()[2:]
        B,C,H,W = x.shape
        # encoding
        #    0         1        2        3
        # [(48,128),(96,64),(192,32),(384,16)]
        encoder_outs = self.encoder(x) 

        if out_seg == False:
            return {'seg': None, 'feat': nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)}

        out = self.decoder(encoder_outs, img_size=img_size)

        if out_feat:
            return {'seg': out, 'feat': nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)}
        else:
            return out






class FATNet_DASE(nn.Module):
    '''
    implement sota adapters: (DASE) Domain-attentive SE adapter
    https://github.com/frank-xwang/towards-universal-object-detection/tree/cf40aed4c79b86b3e8e08e4adf94f43742693111
    A Conv Position encoding + Factorized attention Transformer
    use transformer encoder and decoder
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
        **kwargs,
    ):
        super(FATNet_DASE, self).__init__()
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
            MHSA_stage(
                embed_dims[idx],
                num_layers=num_layers[idx],
                num_heads=num_heads[idx], 
                mlp_ratio=mlp_ratios[idx], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer
            ) for idx in range(self.num_stages)
        ])
        
        self.encoder_adapters = nn.ModuleList([
            DomainAttention(embed_dims[idx],reduction=16) for idx in range(self.num_stages)
        ])
        self.decoder_adapters = nn.ModuleList([
            DomainAttention(embed_dims[-idx-1],reduction=16) for idx in range(self.num_stages)
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
                MHSA_stage(
                    embed_dims[idx],
                    num_layers=num_layers[idx],
                    num_heads=num_heads[idx], 
                    mlp_ratio=mlp_ratios[idx], 
                    qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                    norm_layer=norm_layer
                )            
            )

        self.decoder1 = UnetDecodingBlockTransformer(embed_dims[3]*2,embed_dims[3],self.mhsa_list[3],conv_norm=conv_norm)  # 768,384
        self.decoder2 = UnetDecodingBlockTransformer(embed_dims[3],embed_dims[2],self.mhsa_list[2],conv_norm=conv_norm)  # 384,192
        self.decoder3 = UnetDecodingBlockTransformer(embed_dims[2],embed_dims[1],self.mhsa_list[1],conv_norm=conv_norm)   # 192,96
        self.decoder4 = UnetDecodingBlockTransformer(embed_dims[1],embed_dims[0],self.mhsa_list[0],conv_norm=conv_norm)    # 96,48
        self.finalconv = nn.Sequential(
            nn.Conv2d(embed_dims[0], 1, kernel_size=1)
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

    def forward(self, x, out_feat=False, out_seg=True):
        # x (B,in_chans,H,W)
        img_size = x.size()[2:]
        x = self.stem(x)  # (B,embed_dim[0],H/4,W/4)
        encoder_outs = []
        for idx in range(self.num_stages):
            x = self.patch_embed_stages[idx](x)  # (B, embed_dim[idx],H/(4*2^idx),W/(4*2^idx))
            B,C,H,W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.mhsa_stages[idx](x, H, W)
            x = rearrange(x, 'b (h w) c -> b c h w', w=W, h=H).contiguous()
            # TODO adapter
            x = self.encoder_adapters[idx](x)
            encoder_outs.append(x)

        if out_seg == False:
            return {'seg': None, 'feat': nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)}
  
        # bridge
        out = self.bridge(encoder_outs[3])

        # decoding
        out = self.decoder1(out, encoder_outs[3])  # (384,16,16)
        out = self.decoder_adapters[0](out)
        out = self.decoder2(out, encoder_outs[2])  # (192,32,32)
        out = self.decoder_adapters[1](out)
        out = self.decoder3(out, encoder_outs[1])  # (96,64,64)
        out = self.decoder_adapters[2](out)
        out = self.decoder4(out, encoder_outs[0])  # (48,128,128)
        out = self.decoder_adapters[3](out)
        
        # upsample
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        out = self.finalconv(out)  # (1,512,512)            
        
        if out_feat:
            return {'seg': out, 'feat': nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)}
        else:
            return out


class FATNet_USE(nn.Module):
    '''
    implement USE-Net on FATNet backbone
    https://github.com/frank-xwang/towards-universal-object-detection/tree/cf40aed4c79b86b3e8e08e4adf94f43742693111
    A Conv Position encoding + Factorized attention Transformer
    use transformer encoder and decoder
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
        **kwargs,
    ):
        super(FATNet_USE, self).__init__()
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
            MHSA_stage(
                embed_dims[idx],
                num_layers=num_layers[idx],
                num_heads=num_heads[idx], 
                mlp_ratio=mlp_ratios[idx], 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer
            ) for idx in range(self.num_stages)
        ])
        
        self.encoder_adapters = nn.ModuleList([
            SEBlock(embed_dims[idx],reduction=8) for idx in range(self.num_stages)
        ])
        self.bridge_adapter = SEBlock(embed_dims[3]*2,reduction=8)
        self.decoder_adapters = nn.ModuleList([
            SEBlock(embed_dims[-idx-1],reduction=8) for idx in range(self.num_stages)
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
                MHSA_stage(
                    embed_dims[idx],
                    num_layers=num_layers[idx],
                    num_heads=num_heads[idx], 
                    mlp_ratio=mlp_ratios[idx], 
                    qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                    norm_layer=norm_layer
                )            
            )

        self.decoder1 = UnetDecodingBlockTransformer(embed_dims[3]*2,embed_dims[3],self.mhsa_list[3],conv_norm=conv_norm)  # 768,384
        self.decoder2 = UnetDecodingBlockTransformer(embed_dims[3],embed_dims[2],self.mhsa_list[2],conv_norm=conv_norm)  # 384,192
        self.decoder3 = UnetDecodingBlockTransformer(embed_dims[2],embed_dims[1],self.mhsa_list[1],conv_norm=conv_norm)   # 192,96
        self.decoder4 = UnetDecodingBlockTransformer(embed_dims[1],embed_dims[0],self.mhsa_list[0],conv_norm=conv_norm)    # 96,48
        self.finalconv = nn.Sequential(
            nn.Conv2d(embed_dims[0], 1, kernel_size=1)
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

    def forward(self, x, out_feat=False, out_seg=True):
        # x (B,in_chans,H,W)
        img_size = x.size()[2:]
        x = self.stem(x)  # (B,embed_dim[0],H/4,W/4)
        encoder_outs = []
        for idx in range(self.num_stages):
            x = self.patch_embed_stages[idx](x)  # (B, embed_dim[idx],H/(4*2^idx),W/(4*2^idx))
            B,C,H,W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.mhsa_stages[idx](x, H, W)
            x = rearrange(x, 'b (h w) c -> b c h w', w=W, h=H).contiguous()
            # TODO adapter on the skip connection
            encoder_outs.append(self.encoder_adapters[idx](x))

        if out_seg == False:
            return {'seg': None, 'feat': nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)}
  
        # bridge
        out = self.bridge(encoder_outs[3])
        out = self.bridge_adapter(out)

        # decoding
        out = self.decoder1(out, encoder_outs[3])  # (384,16,16)
        out = self.decoder_adapters[0](out)
        out = self.decoder2(out, encoder_outs[2])  # (192,32,32)
        out = self.decoder_adapters[1](out)
        out = self.decoder3(out, encoder_outs[1])  # (96,64,64)
        out = self.decoder_adapters[2](out)
        out = self.decoder4(out, encoder_outs[0])  # (48,128,128)
        out = self.decoder_adapters[3](out)
        
        # upsample
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (48,512,512)
        out = self.finalconv(out)  # (1,512,512)            
        
        if out_feat:
            return {'seg': out, 'feat': nn.functional.adaptive_avg_pool2d(encoder_outs[3],1).reshape(B, -1)}
        else:
            return out


class SEBlock(nn.Module):
    '''for USE-Net'''
    def __init__(self, channel, reduction):
        super(SEBlock, self).__init__()
        self.se_layer = SELayer(channel, reduction, with_sigmoid=True)
    
    def forward(self, x):
        se = self.se_layer(x)
        se = se*x+x
        return se



if __name__ == '__main__':
    x = torch.randn(2,3,256,256)
    model = FATNet_USE()
    y = model(x)
    print(y.shape)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    # flops = FlopCountAnalysis(model, x)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # acts = ActivationCountAnalysis(model, x)

    # print(f"total flops : {flops.total()/1e12} M")
    # print(f"total activations: {acts.total()/1e6} M")
    print(f"number of parameter: {param/1e6} M")