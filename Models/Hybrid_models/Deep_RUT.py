'''
DeepRUT: DeepResUnetTransformer
'''
import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')

from Models.CNN.ResNet import resnet18, resnet34, resnet50, resnet101
# from ResNet import resnet18, resnet34, resnet50, resnet101
from Models.Transformer.Vit import Transformer

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
        out = nn.functional.interpolate(low_input,size = cur_size,mode = 'bilinear')
        out = self.conv_before(out)
        out = torch.cat((cur_input,out),dim=1)
        return self.conv_after(out)


class DeepRUT(nn.Module):
    '''
    DeepResUnetTransformer, used for segmentation
    encoder_id chooses between resnet[18,34,50,101]
    tran_dim: dim between attention and mlp in transformer layer
    dim_head: dim in the attention
    '''
    def __init__(self, pretrained, encoder_id=0, image_size=512, tran_dim=128, depth=6, heads=8, head_dim=128, mlp_dim=1024, dropout=0.1):
        super(DeepRUT, self).__init__()
        #    0         1        2        3
        # [(64,128),(128,64),(256,32),(512,16)]
        self.encoder = resnet_list[encoder_id](pretrained=pretrained, out_indices=[1,2,3,4])

        # ------------------------------------- transformer ----------------------------
        num_patches = (image_size//32)**2
        self.conv_before_tran = nn.Conv2d(512, tran_dim, kernel_size=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, tran_dim))
        self.dropout = nn.Dropout(dropout)
        # transformer layers
        self.transformer = Transformer(tran_dim, depth, heads, head_dim, mlp_dim, dropout) # don't do project in attention
        self.conv_after_tran = nn.Sequential(
            nn.Conv2d(128,512,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1024,kernel_size=3,stride=1, padding=1),
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

        # transformer
        b,c,w,h = encoder_outs[3].shape
        out = self.conv_before_tran(encoder_outs[3])  # (128,16,16)
        out = rearrange(out, 'b c w h -> b (w h) c')  # (n, 16*16, 128)
        out = out+self.pos_embedding
        out = self.dropout(out)
        out = self.transformer(out)  # (n, 16*16, 128)
        out = rearrange(out, 'b (w h) c -> b c w h', w=w, h=h) # (128,16,16)
        out = self.conv_after_tran(out)  # (1024,16,16)

        # decode
        out = self.decoder1(out, encoder_outs[3])  # (512,16,16)
        out = self.decoder2(out, encoder_outs[2])  # (256,32,32)
        out = self.decoder3(out, encoder_outs[1])  # (128,64,64)
        out = self.decoder4(out, encoder_outs[0])  # (64,128,128)

        # upsample
        out = self.finalconv(out)  # (1,128,128)
        out = nn.functional.interpolate(out,size = x.size()[2:],mode = 'bilinear') # (1,512,512)

        return out


if __name__ == '__main__':
    x = torch.randn(4,3,512,512)
    model = DeepRUT(pretrained=False)
    y = model(x)
    print(y.shape)