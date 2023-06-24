'''
Store different decoders for segmentation model
'''

from turtle import forward
import torch
import torch.nn as nn
import math
from einops import rearrange
import sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')
from Utils._deeplab import ASPP


class DWConv2d_BN(nn.Module):
    """Depthwise Separable Convolution with BN module."""
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
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
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

    def forward(self, x):
        """
        foward function
        """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWConv2d_BN_M(nn.Module):
    """Depthwise Separable Convolution with BN module."""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
        num_domains=1,
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
        # self.bn = norm_layer(out_ch)
        self.bns = nn.ModuleList([norm_layer(out_ch) for _ in range(num_domains)])
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

    def forward(self, x, d='0'):
        """
        foward function
        """
        d = int(d)
        x = self.dwconv(x)
        x = self.pwconv(x)
        # x = self.bn(x)
        x = self.bns[d](x)
        x = self.act(x)

        return x



class UnetDecodingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_res=False, conv_norm=nn.BatchNorm2d):
        '''
        upsample and conv input, concat with skip from encoder
        then conv this combination
        use_res: True means to use residual block for conv_after
        '''
        super(UnetDecodingBlock, self).__init__()
        self.use_res = use_res
        self.conv_before = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        # conv after cat
        if out_channel>512:
            kernel_size,padding = 1,0
        else:
            kernel_size,padding = 3,1
        self.conv_after = nn.Sequential(
            nn.Conv2d(out_channel*2,out_channel,kernel_size=kernel_size,stride=1,padding=padding),
            conv_norm(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,kernel_size=kernel_size,stride=1,padding=padding),
            conv_norm(out_channel),
            nn.ReLU(inplace=True)
        )

        if self.use_res:
            self.res_conv = nn.Sequential(
                nn.Conv2d(out_channel*2, out_channel, kernel_size=1, stride=1),
                conv_norm(out_channel)
            )
    
    def forward(self, input, skip):
        skip_size = skip.size()[2:]
        out = nn.functional.interpolate(input,size = skip_size,mode = 'bilinear', align_corners=False)
        out = self.conv_before(out)
        out = torch.cat((skip,out),dim=1)
        if self.use_res:
            return self.res_conv(out) + self.conv_after(out)
        else:
            return self.conv_after(out)


class UnetDecodingBlock_M(nn.Module):
    def __init__(self, in_channel, out_channel, use_res=False, conv_norm=nn.BatchNorm2d, num_domains=1):
        '''
        upsample and conv input, concat with skip from encoder
        then conv this combination
        use_res: True means to use residual block for conv_after
        '''
        super(UnetDecodingBlock_M, self).__init__()
        self.use_res = use_res
        self.conv_before = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        # conv after cat
        if out_channel>512:
            kernel_size,padding = 1,0
        else:
            kernel_size,padding = 3,1

        self.conv_after_conv1 = nn.Conv2d(out_channel*2,out_channel,kernel_size=kernel_size,stride=1,padding=padding)
        self.conv_after_norm1 = nn.ModuleList([conv_norm(out_channel) for _ in range(num_domains)])
        self.conv_after_act1 = nn.ReLU(inplace=True)
        self.conv_after_conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=kernel_size,stride=1,padding=padding)
        self.conv_after_norm2 = nn.ModuleList([conv_norm(out_channel) for _ in range(num_domains)])
        self.conv_after_act2 = nn.ReLU(inplace=True)

        if self.use_res:
            self.res_conv_conv1 = nn.Conv2d(out_channel*2, out_channel, kernel_size=1, stride=1),
            self.res_conv_norm1 = nn.ModuleList([conv_norm(out_channel) for _ in range(num_domains)])
    
    def forward(self, input, skip, d):
        skip_size = skip.size()[2:]
        int_d = int(d)
        out = nn.functional.interpolate(input,size = skip_size,mode = 'bilinear', align_corners=False)
        out = self.conv_before(out)
        out = torch.cat((skip,out),dim=1)
        x = self.conv_after_conv1(out)
        x = self.conv_after_norm1[int_d](x)
        x = self.conv_after_act1(x)
        x = self.conv_after_conv2(x)
        x = self.conv_after_norm2[int_d](x)
        x = self.conv_after_act2(x)
        if self.use_res:
            return self.res_conv_norm1[int_d](self.res_conv_conv1(out)) + x
        else:
            return x


class ResidualDecodingBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.before_conv = nn.Conv2d(in_channels,out_channels,1,1)
        self.conv_after = nn.Sequential(
                        nn.Conv2d(out_channels*2,out_channels//2,1,1),
                        nn.BatchNorm2d(out_channels//2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//2,out_channels//2,3,1,1),
                        nn.BatchNorm2d(out_channels//2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//2,out_channels,1,1,),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
        )
        self.skip = nn.Conv2d(out_channels*2,out_channels,1,1)
    
    def forward(self,input,skip,d=None):
        skip_size = skip.size()[2:]
        out = nn.functional.interpolate(input,size = skip_size,mode = 'bilinear', align_corners=False)
        out = self.before_conv(out)
        out = torch.cat((skip,out),dim=1)
        return self.conv_after(out)+self.skip(out)
        


class UnetDecodingBlockTransformer_M(nn.Module):
    def __init__(self, in_channel, out_channel, mhsa_block, use_res=False, conv_norm=nn.BatchNorm2d,num_domains=1):
        '''
        upsample and conv input, concat with skip from encoder
        then conv this combination
        use_res: True means to use residual block for conv_after
        '''
        super(UnetDecodingBlockTransformer_M, self).__init__()
        self.use_res = use_res
        self.conv_before = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.conv_after = DWConv2d_BN_M(out_channel*2, out_channel, kernel_size=3, stride=1,num_domains=num_domains)
        # conv after cat
        self.mhsa_block = mhsa_block

        if self.use_res:
            self.res_conv = nn.Sequential(
                nn.Conv2d(out_channel*2, out_channel, kernel_size=1, stride=1),
                conv_norm(out_channel)
            )
    
    def forward(self, input, skip, d=None, domain_label=None):
        skip_size = skip.size()[2:]
        out = nn.functional.interpolate(input,size = skip_size,mode = 'bilinear', align_corners=False)
        out = self.conv_before(out)
        out = torch.cat((skip,out),dim=1)
        d = int(d)
        if self.use_res:
            out = self.conv_after(out,d)
            res = self.res_conv(out)
            out = rearrange(out, 'b c h w -> b (h w) c')
            try:
                out = self.mhsa_block(out, skip_size[0], skip_size[1],d=d) if \
                    domain_label==None else self.mhsa_block(out, skip_size[0], skip_size[1],domain_label,d)
            except:
                out = self.mhsa_block(out, skip_size[0], skip_size[1]) if \
                    domain_label==None else self.mhsa_block(out, skip_size[0], skip_size[1],domain_label)
            out = rearrange(out, 'b (h w) c -> b c h w', h=skip_size[0], w=skip_size[1]).contiguous()
            return res + out
        else:
            out = self.conv_after(out,d)
            out = rearrange(out, 'b c h w -> b (h w) c')
            try:
                out = self.mhsa_block(out, skip_size[0], skip_size[1], d=d) if \
                    domain_label==None else self.mhsa_block(out, skip_size[0], skip_size[1],domain_label, d)
            except:
                out = self.mhsa_block(out, skip_size[0], skip_size[1]) if \
                    domain_label==None else self.mhsa_block(out, skip_size[0], skip_size[1],domain_label)
            # out = self.mhsa_block(out)
            out = rearrange(out, 'b (h w) c -> b c h w', h=skip_size[0], w=skip_size[1]).contiguous()
            return out


class UnetDecodingBlockTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, mhsa_block, use_res=False, conv_norm=nn.BatchNorm2d):
        '''
        upsample and conv input, concat with skip from encoder
        then conv this combination
        use_res: True means to use residual block for conv_after
        '''
        super(UnetDecodingBlockTransformer, self).__init__()
        self.use_res = use_res
        self.conv_before = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.conv_after = DWConv2d_BN(out_channel*2, out_channel, kernel_size=3, stride=1)
        # conv after cat
        self.mhsa_block = mhsa_block

        if self.use_res:
            self.res_conv = nn.Sequential(
                nn.Conv2d(out_channel*2, out_channel, kernel_size=1, stride=1),
                conv_norm(out_channel)
            )
    
    def forward(self, input, skip, domain_label=None):
        skip_size = skip.size()[2:]
        out = nn.functional.interpolate(input,size = skip_size,mode = 'bilinear', align_corners=False)
        out = self.conv_before(out)
        out = torch.cat((skip,out),dim=1)
        if self.use_res:
            out = self.conv_after(out)
            res = self.res_conv(out)
            out = rearrange(out, 'b c h w -> b (h w) c')
            out = self.mhsa_block(out, skip_size[0], skip_size[1]) if \
                domain_label==None else self.mhsa_block(out, skip_size[0], skip_size[1],domain_label)
            out = rearrange(out, 'b (h w) c -> b c h w', h=skip_size[0], w=skip_size[1]).contiguous()
            return res + out
        else:
            out = self.conv_after(out)
            out = rearrange(out, 'b c h w -> b (h w) c')
            out = self.mhsa_block(out, skip_size[0], skip_size[1]) if \
                domain_label==None else self.mhsa_block(out, skip_size[0], skip_size[1],domain_label)
            # out = self.mhsa_block(out)
            out = rearrange(out, 'b (h w) c -> b c h w', h=skip_size[0], w=skip_size[1]).contiguous()
            return out



class DeepLabV3Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, aspp_dilate=[6, 12, 18], conv_norm=nn.BatchNorm2d,):
        super(DeepLabV3Decoder, self).__init__()
        self.classifier = nn.Sequential(
            ASPP(in_channel, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            conv_norm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channel, 1)
        )

    def forward(self, feature, img_size):
        if isinstance(feature, list):
            feature = feature[-1]
        out = self.classifier(feature)
        # print(out.shape)
        out = nn.functional.interpolate(out,size = img_size, mode = 'bilinear', align_corners=False) # (B,hC,H,W)
        return out


# 4 encoder features
class MLPDecoder(nn.Module):
    '''
    Imitate SegFormer decoder
    '''
    def __init__(self, in_channels, out_channel, hidden_channel=256, dropout_ratio=0.1, conv_norm=nn.BatchNorm2d,):
        super(MLPDecoder, self).__init__()
        self.linear1 = nn.Conv2d(in_channels[0], hidden_channel, 1)  # H/4
        self.linear2 = nn.Conv2d(in_channels[1], hidden_channel, 1)  # H/8
        self.linear3 = nn.Conv2d(in_channels[2], hidden_channel, 1)  # H/16
        self.linear4 = nn.Conv2d(in_channels[3], hidden_channel, 1)  # H/32

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(hidden_channel*4, hidden_channel, 1),
            conv_norm(hidden_channel),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        if hidden_channel == out_channel:
            self.linear_out = nn.Identity()
        else:
            self.linear_out = nn.Conv2d(hidden_channel, out_channel, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        


    def forward(self, features, img_size, out_feat=False):
        x1, x2, x3, x4 = features   # (B,C,H,W)  H/4, H/8, H/16, H/32
        h, w = x1.shape[2:]
        x1 = self.linear1(x1)  
        x1 = nn.functional.interpolate(x1,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

        x2 = self.linear2(x2)
        x2 = nn.functional.interpolate(x2,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

        x3 = self.linear3(x3)
        x3 = nn.functional.interpolate(x3,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

        x4 = self.linear4(x4)
        x4 = nn.functional.interpolate(x4,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

        out = torch.cat([x1, x2, x3, x4], dim=1)  # (B,4*hC,H/4,W/4)
        out = self.linear_fuse(out)  # (B,hC,H/4,W/4)
        if out_feat == True:
            feat = self.avg_pool(out)  # (B,hc,1)
        out = self.dropout(out)

        out = nn.functional.interpolate(out,size = img_size, mode = 'bilinear', align_corners=False) # (B,hC,H,W)
        out = self.linear_out(out)
   
        return {'seg':out, 'feat':feat} if out_feat else out


# 5 encoder features
# class MLPDecoder(nn.Module):
#     '''
#     Imitate SegFormer decoder
#     '''
#     def __init__(self, in_channels, out_channel, hidden_channel=256, dropout_ratio=0.1, conv_norm=nn.BatchNorm2d,):
#         super(MLPDecoder, self).__init__()
#         self.linear1 = nn.Conv2d(in_channels[0], hidden_channel, 1)  # H/4
#         self.linear2 = nn.Conv2d(in_channels[1], hidden_channel, 1)  # H/8
#         self.linear3 = nn.Conv2d(in_channels[2], hidden_channel, 1)  # H/16
#         self.linear4 = nn.Conv2d(in_channels[3], hidden_channel, 1)  # H/32
#         self.linear5 = nn.Conv2d(in_channels[4], hidden_channel, 1)  # H/32

#         self.linear_fuse = nn.Sequential(
#             nn.Conv2d(hidden_channel*5, hidden_channel, 1),
#             conv_norm(hidden_channel),
#             nn.ReLU(inplace=True),
#         )
#         self.dropout = nn.Dropout2d(dropout_ratio)
#         self.linear_out = nn.Conv2d(hidden_channel, out_channel, 1)
        


#     def forward(self, features, img_size):
#         x1, x2, x3, x4, x5 = features   # (B,C,H,W)  H/4, H/8, H/16, H/32
#         h, w = x1.shape[2:]
#         x1 = self.linear1(x1)  
#         x1 = nn.functional.interpolate(x1,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

#         x2 = self.linear2(x2)
#         x2 = nn.functional.interpolate(x2,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

#         x3 = self.linear3(x3)
#         x3 = nn.functional.interpolate(x3,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

#         x4 = self.linear4(x4)
#         x4 = nn.functional.interpolate(x4,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

#         x5 = self.linear5(x5)
#         x5 = nn.functional.interpolate(x5,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

#         out = torch.cat([x1, x2, x3, x4, x5], dim=1)  # (B,4*hC,H/4,W/4)
#         out = self.linear_fuse(out)  # (B,hC,H/4,W/4)
#         out = self.dropout(out)

#         out = nn.functional.interpolate(out,size = img_size, mode = 'bilinear', align_corners=False) # (B,hC,H,W)
#         out = self.linear_out(out)

#         return out


class MLPDecoderFM(nn.Module):
    '''
    Imitate SegFormer decoder
    add a feature from uni decoder
    '''
    def __init__(self, in_channels, out_channel, hidden_channel=256, outfeature_channel=64, dropout_ratio=0.1, conv_norm=nn.BatchNorm2d,):
        '''
        outfeature_channel is the dimension of features from outside
        '''
        super(MLPDecoderFM, self).__init__()
        self.linear1 = nn.Conv2d(in_channels[0], hidden_channel, 1)  # H/4
        self.linear2 = nn.Conv2d(in_channels[1], hidden_channel, 1)  # H/8
        self.linear3 = nn.Conv2d(in_channels[2], hidden_channel, 1)  # H/16
        self.linear4 = nn.Conv2d(in_channels[3], hidden_channel, 1)  # H/32

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(hidden_channel*4+outfeature_channel, hidden_channel, 1),
            conv_norm(hidden_channel),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_out = nn.Conv2d(hidden_channel, out_channel, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, features, img_size, out_feat=False):
        x1, x2, x3, x4, x5 = features   # (B,C,H,W)  H/4, H/8, H/16, H/32
        h, w = x1.shape[2:]
        x1 = self.linear1(x1)  
        x1 = nn.functional.interpolate(x1,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

        x2 = self.linear2(x2)
        x2 = nn.functional.interpolate(x2,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

        x3 = self.linear3(x3)
        x3 = nn.functional.interpolate(x3,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

        x4 = self.linear4(x4)
        x4 = nn.functional.interpolate(x4,size = (h,w), mode = 'bilinear', align_corners=False) # (B,hC,H/4,W/4)

        out = torch.cat([x1, x2, x3, x4, x5], dim=1)  # (B,4*hC,H/4,W/4)
        out = self.linear_fuse(out)  # (B,hC,H/4,W/4)
        if out_feat == True:
            feat = self.avg_pool(out)  # (B,hc,1)
        out = self.dropout(out)

        out = nn.functional.interpolate(out,size = img_size, mode = 'bilinear', align_corners=False) # (B,hC,H,W)
        out = self.linear_out(out)

        return {'seg':out, 'feat':feat} if out_feat else out


if __name__ == '__main__':
    # mhsa_block = nn.Identity()
    # net = UnetDecodingBlockTransformer(640, 256, mhsa_block, use_res=False, conv_norm=nn.InstanceNorm2d)
    # x, skip = torch.randn(5, 640, 16, 16), torch.randn(5, 256, 32, 32)
    # y = net(x, skip)
    # print(y.shape)

    # net = MLPDecoderFM([64,128,320,512],1,hidden_channel=512)
    # x1 = torch.randn(5,512,8,8)
    # x2 = torch.randn(5,320,16,16)
    # x3 = torch.randn(5,128,32,32)
    # x4 = torch.randn(5,64,64,64)
    # x5 = torch.randn(5,64,64,64)
    # y = net([x4,x3,x2,x1,x5], (256,256))

    net = ResidualDecodingBlock(512,256)
    x = torch.randn(5, 512, 8, 8)
    skip = torch.randn(5,256,16,16)
    y = net(x, skip)

    print(y.shape)
    total_trainable_params = sum(
                    p.numel() for p in net.parameters() if p.requires_grad)
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    pass
