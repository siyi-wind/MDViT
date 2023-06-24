'''
MS-Net reproduction https://ieeexplore.ieee.org/abstract/document/9000851?casa_token=GxRRMSQQrAkAAAAA:ScnXvN_2P_EpiyatJM2iw3LfxiZxrH2u8WAZr3eWwyfNt8rJkQ9aZkaIsn9MrzjqIOoB-cKyqOI 
Residual Unet + Specific Batch Norm + Auxiliary decoders
Follow codes
https://github.com/liuquande/MS-Net/tree/99c17044d2ece468216310b68fc9e86d6c4dbd78
https://github.com/med-air/Contrastive-COVIDNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')


class MSNet(nn.Module):
    '''
    Residual conv, use domain specific norm
    '''
    def __init__(self, in_chans=3, out_chans=1, embed_dims=[32, 64, 128, 256, 512], num_domains=4):
        super(MSNet, self).__init__()
        
        # encoding
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dims[0], kernel_size=3, stride=1, padding=1)
        self.res_block1 = DSResConv(embed_dims[0], embed_dims[0], num_domains=num_domains)
        self.pool1 = nn.MaxPool2d(2,2)

        self.res_block2 = DSResConv(embed_dims[0], embed_dims[1], num_domains=num_domains)
        self.pool2 = nn.MaxPool2d(2,2)

        self.res_block3 = DSResConv(embed_dims[1], embed_dims[2], num_domains=num_domains)
        self.pool3 = nn.MaxPool2d(2,2)

        self.res_block4 = DSResConv(embed_dims[2], embed_dims[3], num_domains=num_domains)
        self.pool4 = nn.MaxPool2d(2,2)

        self.res_block5_1 = DSResConv(embed_dims[3], embed_dims[4], num_domains=num_domains)
        self.res_block5_2 = DSResConv(embed_dims[4], embed_dims[4], num_domains=num_domains)


        # decoding
        self.uni_decoder = MSNetDecoder(in_chans,out_chans,embed_dims,num_domains)
        self.debranch1 = MSNetDecoder(in_chans,out_chans,embed_dims,1)
        self.debranch2 = MSNetDecoder(in_chans,out_chans,embed_dims,1)
        self.debranch3 = MSNetDecoder(in_chans,out_chans,embed_dims,1)
        self.debranch4 = MSNetDecoder(in_chans,out_chans,embed_dims,1)
         

    def forward(self, x, d, out_feat=False, out_seg=True):
        # encoding
        B,C,H,W = x.shape
        out = self.conv(x)
        en_res1 = self.res_block1(out,d)
        out = self.pool1(en_res1)

        en_res2 = self.res_block2(out,d)
        out = self.pool2(en_res2)

        en_res3 = self.res_block3(out,d)
        out = self.pool3(en_res3)

        en_res4 = self.res_block4(out,d)
        out = self.pool4(en_res4)

        out = self.res_block5_1(out,d)
        out = self.res_block5_2(out,d)

        encoder_outs = [en_res1,en_res2,en_res3,en_res4,out]

        if out_seg == False:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[4],1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # decoding
        uni_out = self.uni_decoder(encoder_outs,d,img_size=(H,W))
        if d == '0':
            aux_out = self.debranch1(encoder_outs, '0', img_size=(H,W))
        elif d == '1':
            aux_out = self.debranch2(encoder_outs, '0', img_size=(H,W))
        elif d == '2':
            aux_out = self.debranch3(encoder_outs, '0', img_size=(H,W))
        elif d == '3':
            aux_out = self.debranch4(encoder_outs, '0', img_size=(H,W))
        else:
            aux_out = None

        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[4],1).reshape(B, -1)
            return {'seg': [uni_out, aux_out], 'feat': x}
        else:
            return [uni_out, aux_out]



class MSNetDecoder(nn.Module):
    '''
    Residual deconv, use domain specific norm
    '''
    def __init__(self, in_chans, out_chans, embed_dims=[32, 64, 128, 256, 512], num_domains=4, kernel=3, pad=1):
        super(MSNetDecoder, self).__init__()
        self.conv1 = nn.Conv2d(embed_dims[4],embed_dims[3],kernel,1,pad)
        self.res_block1 = DSResConv(embed_dims[3]*2,embed_dims[3],num_domains=num_domains)
        self.conv2 = nn.Conv2d(embed_dims[3],embed_dims[2],kernel,1,pad)
        self.res_block2 = DSResConv(embed_dims[2]*2,embed_dims[2],num_domains=num_domains)
        self.conv3 = nn.Conv2d(embed_dims[2],embed_dims[1],kernel,1,pad)
        self.res_block3 = DSResConv(embed_dims[1]*2,embed_dims[1],num_domains=num_domains)
        self.conv4 = nn.Conv2d(embed_dims[1],embed_dims[0],kernel,1,pad)
        self.res_block4 = DSResConv(embed_dims[0]*2,embed_dims[0],num_domains=num_domains)

        self.out_conv = nn.Conv2d(embed_dims[0],1,1,1)

    def forward(self, features, d='0', img_size=None):
        # features 512 H, 256 H/2, 128 H/4, 64 H/8, 32 H/16
        out = self.conv1(features[4])  # (256,H/16,W/16)
        out = F.interpolate(out,size = features[3].shape[2:], mode = 'bilinear', align_corners=False) # (256,H/8,W/8)
        out = torch.cat((features[3],out),dim=1)  # (512,H/8,W/8)
        out = self.res_block1(out,d)  # (256,H/8,W/8)

        out = self.conv2(out)  # (128,H/8,W/8)
        out = F.interpolate(out,size = features[2].shape[2:], mode = 'bilinear', align_corners=False) # (128,H/4,W/4)
        out = torch.cat((features[2],out),dim=1)  # (256,H/4,W/4)
        out = self.res_block2(out,d)  # (128,H/4,W/4)

        out = self.conv3(out)  # (64,H/4,W/4)
        out = F.interpolate(out,size = features[1].shape[2:], mode = 'bilinear', align_corners=False) # (64,H/2,W/2)
        out = torch.cat((features[1],out),dim=1)  # (128,H/2,W/2)
        out = self.res_block3(out,d)  # (64,H/2,W/2)

        out = self.conv4(out)  # (32,H/2,W/2)
        out = F.interpolate(out,size = features[0].shape[2:], mode = 'bilinear', align_corners=False) # (32,H,W)
        out = torch.cat((features[0],out),dim=1)  # (64,H,W)
        out = self.res_block4(out,d)  # (32,H,W)

        out = F.interpolate(out,size = img_size, mode = 'bilinear', align_corners=False)
        out = self.out_conv(out)  # (1,H,W)

        return out



class DSResConv(nn.Module):
    '''
    Residual conv, use domain specific norm
    '''
    def __init__(self, in_c, out_c, kernel=3, stride=1, pad=1, num_domains=4):
        super(DSResConv, self).__init__()
        self.conv1 = nn.Conv2d(in_c,out_c,kernel,stride,pad)
        self.norms1 = nn.ModuleList([nn.BatchNorm2d(out_c) for i in range(num_domains)])
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c,out_c,kernel,stride,pad)
        self.norms2 = nn.ModuleList([nn.BatchNorm2d(out_c) for i in range(num_domains)])
        self.relu2 = nn.ReLU(inplace=True)

        self.res = nn.Identity() if in_c==out_c else nn.Conv2d(in_c, out_c, 3, 1, 1)
        # self.res = nn.Identity() if in_c==out_c else nn.Conv2d(in_c, out_c, 1)

    def forward(self, x, d):
        '''d is domain number'''
        d = int(d)
        out = self.conv1(x)
        out = self.norms1[d](out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norms2[d](out)
        out = self.relu2(out)
        out = out+self.res(x)
        return out


if __name__ == '__main__':
    x = torch.randn(5,3,256,256).cuda()
    domain_label = torch.randint(0,4,(5,))

    domain_label = torch.nn.functional.one_hot(domain_label, 4).float()

    model = MSNet()
    # d = '0'
    # y = model(x, d, out_feat=True)
    # print(y['seg'][0].shape)
    # print(y['feat'].shape)

    # model = MSNetDecoder(3,1, num_domains=1)  # [64,128,320,512,512]


    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    # flops = FlopCountAnalysis(model, x)
    param = sum(p.numel() for p in model.debranch1.parameters() if p.requires_grad)
    # acts = ActivationCountAnalysis(model, x)

    # print(f"total flops : {flops.total()/1e12} M")
    # print(f"total activations: {acts.total()/1e6} M")
    print(f"number of parameter: {param/1e6} M")