'''
ResUnet as the backbone, adapters: series_adapters, parallel_adapters, DASE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')
from Models.Sota_adapters.residual_adapter_module import BasicBlock


class ResUnet_adapt(nn.Module):
    '''
    Residual conv, use domain specific norm
    '''
    def __init__(self, in_chans=3, out_chans=1, embed_dims=[32, 64, 128, 256, 512], num_domains=4, adapt_method='series_adapters'):
        super(ResUnet_adapt, self).__init__()
        
        # encoding
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dims[0], kernel_size=3, stride=1, padding=1)
        self.res_block1 = BasicBlock(embed_dims[0], embed_dims[0], nb_tasks=num_domains, adapt_method=adapt_method)
        self.pool1 = nn.MaxPool2d(2,2)

        self.res_block2 = BasicBlock(embed_dims[0], embed_dims[1], nb_tasks=num_domains, adapt_method=adapt_method)
        self.pool2 = nn.MaxPool2d(2,2)       

        self.res_block3 = BasicBlock(embed_dims[1], embed_dims[2], nb_tasks=num_domains, adapt_method=adapt_method)
        self.pool3 = nn.MaxPool2d(2,2)

        self.res_block4 = BasicBlock(embed_dims[2], embed_dims[3], nb_tasks=num_domains, adapt_method=adapt_method)
        self.pool4 = nn.MaxPool2d(2,2)

        self.res_block5_1 = BasicBlock(embed_dims[3], embed_dims[4], nb_tasks=num_domains, adapt_method=adapt_method)
        self.res_block5_2 = BasicBlock(embed_dims[4], embed_dims[4], nb_tasks=num_domains, adapt_method=adapt_method)


        # decoding
        self.decoder = ResUnetDecoder(in_chans,out_chans,embed_dims,num_domains,adapt_method=adapt_method)
         

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
        out = self.decoder(encoder_outs,d,img_size=(H,W))

        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_outs[4],1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return out



class ResUnetDecoder(nn.Module):
    '''
    Residual deconv, use domain specific norm
    '''
    def __init__(self, in_chans, out_chans, embed_dims=[32, 64, 128, 256, 512], num_domains=4, kernel=3, pad=1, adapt_method='series_adapts'):
        super(ResUnetDecoder, self).__init__()
        self.conv1 = nn.Conv2d(embed_dims[4],embed_dims[3],kernel,1,pad)
        self.res_block1 = BasicBlock(embed_dims[3]*2,embed_dims[3],nb_tasks=num_domains, adapt_method=adapt_method)
        self.conv2 = nn.Conv2d(embed_dims[3],embed_dims[2],kernel,1,pad)
        self.res_block2 = BasicBlock(embed_dims[2]*2,embed_dims[2],nb_tasks=num_domains, adapt_method=adapt_method)
        self.conv3 = nn.Conv2d(embed_dims[2],embed_dims[1],kernel,1,pad)
        self.res_block3 = BasicBlock(embed_dims[1]*2,embed_dims[1],nb_tasks=num_domains, adapt_method=adapt_method)
        self.conv4 = nn.Conv2d(embed_dims[1],embed_dims[0],kernel,1,pad)
        self.res_block4 = BasicBlock(embed_dims[0]*2,embed_dims[0],nb_tasks=num_domains, adapt_method=adapt_method)

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



if __name__ == '__main__':
    x = torch.randn(5,3,256,256)
    domain_label = torch.randint(0,4,(5,))

    domain_label = torch.nn.functional.one_hot(domain_label, 4).float()

    model = ResUnet_adapt(num_domains=4, embed_dims=[32,64,128,256,512],adapt_method='series_adapters')
    d = '1'
    y = model(x, d, out_feat=True)
    print(y['seg'].shape)
    print(y['feat'].shape)

    # model = MSNetDecoder(3,1, num_domains=1)  # [64,128,320,512,512]


    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    # flops = FlopCountAnalysis(model, x)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # acts = ActivationCountAnalysis(model, x)

    # print(f"total flops : {flops.total()/1e12} M")
    # print(f"total activations: {acts.total()/1e6} M")
    print(f"number of parameter: {param/1e6} M")
