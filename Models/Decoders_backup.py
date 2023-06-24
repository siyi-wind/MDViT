'''
4.17
C1-C17 decoder 15.2M
'''
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