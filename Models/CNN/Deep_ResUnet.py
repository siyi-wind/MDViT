'''
Resnet+Unet
'''
import torch
import torch.nn as nn
import os, sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')

from Models.CNN.ResNet import resnet18, resnet34, resnet50, resnet101

resnet_list = [resnet18, resnet34, resnet50, resnet101]

class DecodingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        '''
        upsample and conv low_input, concat with cur_input
        then conv this combination
        '''
        super(DecodingBlock, self).__init__()
        self.conv_before = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        # conv after cat
        self.conv_after = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, low_input, cur_input):
        cur_size = cur_input.size()[2:]
        out = nn.functional.interpolate(low_input,size = cur_size,mode = 'bilinear',align_corners=False)
        out = self.conv_before(out)
        out = torch.cat((cur_input,out),dim=1)
        return self.conv_after(out)


class DeepResUnet(nn.Module):
    def __init__(self, pretrained, encoder_id=0):
        '''
        encoder_id chooses between resnet[18,34,50,101]
        '''

        super(DeepResUnet, self).__init__()
        self.encoder = resnet_list[encoder_id](pretrained=pretrained, out_indices=[1,2,3,4])
        self.center = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # decoder
        self.decoder1 = DecodingBlock(1024,512)
        self.decoder2 = DecodingBlock(512,256)
        self.decoder3 = DecodingBlock(256,128)
        self.decoder4 = DecodingBlock(128,64)
        self.finalconv = nn.Conv2d(64,1,kernel_size=1)
    
    def forward(self,x):
        #    0         1        2        3
        # [(64,128),(128,64),(256,32),(512,16)]
        encoder_outs = self.encoder(x) 
        # for i in range(len(encoder_outs)):
        #     print('encoder outs {}'.format(i), encoder_outs[i].shape)
        center = self.center(encoder_outs[3]) # (1024,16,16)

        # decode
        out = self.decoder1(center, encoder_outs[3])  # (512,16,16)
        out = self.decoder2(out, encoder_outs[2])  # (256,32,32)
        out = self.decoder3(out, encoder_outs[1])  # (128,64,64)
        out = self.decoder4(out, encoder_outs[0])  # (64,128,128)

        # upsample
        # out = self.finalconv(out)  # (1,128,128)
        out = nn.functional.interpolate(out,size = x.size()[2:],mode = 'bilinear',align_corners=False) # (1,512,512)
        out = self.finalconv(out)  # (1,128,128)

        return out



if __name__ == '__main__':
    model = DeepResUnet(pretrained=False, encoder_id=0)
    total_trainable_params = sum(
                    p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    x = torch.randn(16,3,512,512)
    y = model(x)
    print(y.shape)

    # from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    # flops = FlopCountAnalysis(model, x)
    param = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    # acts = ActivationCountAnalysis(model, x)

    # print(f"total flops : {flops.total()/1e12} M")
    # print(f"total activations: {acts.total()/1e6} M")
    print(f"number of parameter: {param/1e6} M")
