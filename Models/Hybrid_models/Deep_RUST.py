'''
DeepRUST: DeepResUnet-SelectiveTransformer
'''
from audioop import bias
from turtle import forward
import torch
import torch.nn as nn
from einops import rearrange, repeat
import sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer/')

from Models.CNN.ResNet import resnet18, resnet34, resnet50, resnet101
# from ResNet import resnet18, resnet34, resnet50, resnet101
from Models.Transformer.Selective_Transformer import Patch_Embed_stage, FactorConv_Transformer


resnet_list = [resnet18, resnet34, resnet50, resnet101]

class DecodingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        '''
        upsample and conv low_input, concat with cur_input
        then conv this combination
        '''
        super(DecodingBlock, self).__init__()
        self.conv_before = nn.Conv2d(in_channel, out_channel, kernel_size=1,bias=False)
        # conv after cat
        self.conv_after = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, low_input, cur_input):
        cur_size = cur_input.size()[2:]
        out = nn.functional.interpolate(low_input,size = cur_size,mode = 'bilinear')
        out = self.conv_before(out)
        out = torch.cat((cur_input,out),dim=1)
        return self.conv_after(out)


class DeepRUST(nn.Module):
    '''
    DeepResUnetSelectiveTransformer, used for segmentation
    encoder_id chooses between resnet[18,34,50,101]
    tran_dim: dim between attention and mlp in transformer layer
    dim_head: dim in the attention
    num_paths: parallel Transformer branch
    select_patch: True means using selective patch mechanism
    '''
    def __init__(self, pretrained, encoder_id=0, image_size=512, tran_dim=128, 
    depth=6, heads=8, head_dim=64, mlp_dim=1024, dropout=0.1, num_paths=3, select_patch = True):
        super(DeepRUST, self).__init__()
        #    0         1        2       
        # [(64,128),(128,64),(256,32)]
        self.select_patch = select_patch
        self.size = (32,32)
        self.encoder = resnet_list[encoder_id](pretrained=pretrained, out_indices=[1,2,3])
        self.layer4 = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1,bias=False)

        # ------------------------------------- transformer ----------------------------
        self.conv_before_tran = nn.Conv2d(512, tran_dim, kernel_size=1,bias=False)
        self.patch_embed = Patch_Embed_stage(embed_dim=tran_dim, num_path=num_paths)
        self.dropout = nn.Dropout(dropout)

        # transformer layers
        self.parallel_transformers = nn.ModuleList([
            FactorConv_Transformer(size=self.size,dim=tran_dim,depth=depth,heads=heads,
            head_dim=head_dim,mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(num_paths)
        ])

        if self.select_patch:
            self.build_att = SelectivePatchAtt(channels=32*32,num_paths=num_paths)
            self.conv_after_tran = nn.Sequential(
                nn.Conv2d(128,512,kernel_size=3,stride=1, padding=1,bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_after_tran = nn.Sequential(
                nn.Conv2d(128*num_paths,128,kernel_size=3,stride=1, padding=1,bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128,512,kernel_size=3,stride=1, padding=1,bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        
        # decoder
        # self.decoder1 = nn.MaxPool2d(2,2) # 
        self.decoder1 = nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1,bias=False)
        self.decoder2 = DecodingBlock(512,256)
        self.decoder3 = DecodingBlock(256,128)
        self.decoder4 = DecodingBlock(128,64)
        self.finalconv = nn.Conv2d(64,1,kernel_size=1,bias=False)
    

    def forward(self,x):
        #    0         1        2        3 
        # [(64,128),(128,64),(256,32),(512,32)]
        encoder_outs = self.encoder(x) 
        encoder_outs.append(self.layer4(encoder_outs[2])) 

        # transformer
        b,c,w,h = encoder_outs[3].shape
        out = self.conv_before_tran(encoder_outs[3])  # (128,32,32)
        att_inputs = self.patch_embed(out)  # a list, each (b,128,32,32)
        out_list = []

        for m, transformer in zip(att_inputs, self.parallel_transformers):
            m = rearrange(m, 'b c w h -> b (w h) c')  # (b,n,dim)
            m = self.dropout(m)
            m = transformer(m, self.size)  # (b,h*w,dim)
            m = rearrange(m, 'b (w h) c -> b c w h', w=w,h=h) # (n,128,32,32)
            out_list.append(m)
        
        if self.select_patch:
            out = torch.stack(out_list, dim=1)  # (n,3,128,32,32)
            out = rearrange(out, 'b p c w h -> b p (w h) c')  # (n,3,1024,128)
            out = out*self.build_att(out)
            out = torch.sum(out,dim=1)
            out = rearrange(out, 'b (w h) c -> b c w h',w=w,h=h)
            out = self.conv_after_tran(out)
        else:
            out = torch.cat(out_list, dim=1)   # (b,3*c,w,h)
            out = self.conv_after_tran(out)    #  (512,32,32)

        # decode
        out = out+encoder_outs[3]
        out = self.decoder1(out)  # (512,16,16)
        out = self.decoder2(out, encoder_outs[2])  # (256,32,32)
        out = self.decoder3(out, encoder_outs[1])  # (128,64,64)
        out = self.decoder4(out, encoder_outs[0])  # (64,128,128)

        # upsample
        out = self.finalconv(out)  # (1,128,128)
        out = nn.functional.interpolate(out,size = x.size()[2:],mode = 'bilinear') # (1,512,512)

        return out


class SelectivePatchAtt(nn.Module):
    '''
    Use Selective Kernel Attention to choose useful patches
    input x  (b,num_paths,channels,dim)
    output   (b,num_paths,channels,1)
    r: ratio, max(32,n//r) is the hidden size
    '''
    def __init__(self,channels, num_paths=3,r=16):
        super(SelectivePatchAtt,self).__init__()
        self.num_paths = num_paths
        hidden_dim = max(channels//r,32)

        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.transform = nn.Sequential(nn.Conv1d(channels, hidden_dim,kernel_size=1,bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            )
        self.fc_select = nn.Conv1d(hidden_dim,num_paths*channels,kernel_size=1,bias=False)

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = self.average_pool(x.sum(1))  # (b,n,1)
        x = self.transform(x)  # (b,hidden,1)
        x = self.fc_select(x)  # (b,num_paths*channels,1)
        x = rearrange(x, 'b (p n) c -> b p n c', p=self.num_paths)
        x = torch.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    x = torch.randn(4,3,512,512)
    net = DeepRUST(pretrained=True)
    total_trainable_params = sum(
                    p.numel() for p in net.parameters() if p.requires_grad)
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    y = net(x)
    print(y.shape)