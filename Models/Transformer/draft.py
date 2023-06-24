'''
draft for ViTSeg CNN adapt
'''

# DMF
# 23.05.02 12:00 0.9191  0.8524
# 23.05.02 16:34 0.9197  0.8536
# 23.05.02 17:54 0.9199

# 23.05.02 12:00  0.9191  0.8524
class ViTSeg_CNNprompt_adapt2(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet18(pretrained=True, out_indices=[1,3])
        # self.prompt_attn = CrossAttention(
        #    k_dim=embed_dim,q_dim=256, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.combination = nn.Conv2d(256+embed_dim,embed_dim,1,1,0)

        # decoder
        self.aspp = ASPP(in_channels=embed_dim,atrous_rates=[6,12,18])
        # self.final_conv = nn.Conv2d(256+64, 1, kernel_size=1)
        self.final_conv = nn.Sequential(
                            nn.Conv2d(256+64,256,3,1,1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256,1,1,1,0)
        )

        # # attention
        self.csattn_0 = ChannelSpatialAttention(embed_dim)
        self.csattn_1 = ChannelSpatialAttention(256)
        self.csattn_2 = ChannelSpatialAttention(64)


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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}

        # cross attention
        # p = self.prompt_encoder(x)[0]
        # p = rearrange(p, 'b c h w -> b (h w) c')
        # encoder_out = self.prompt_attn(q=p,k=encoder_out)
        # encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)

        # concat combination
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        p0,p1 = self.prompt_encoder(x)
        encoder_out = torch.cat((self.csattn_0(encoder_out),self.csattn_1(p1)),dim=1)
        encoder_out =  self.combination(encoder_out)

        # decoding
        out = self.aspp(encoder_out)  # (B,C,H/16,W/16)

        # concat
        out = nn.functional.interpolate(out,size=p0.shape[-2:],mode = 'bilinear', align_corners=False)
        out = torch.cat((out,self.csattn_2(p0)),dim=1)
        
        # upsample
        out = self.final_conv(out)  # (1,H/16,W/16)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}

# 23.05.02 16:34 0.9197  0.8536
class ViTSeg_CNNprompt_adapt2(nn.Module):
    '''
    This is not the MDViT paper's ViTSeg
    Encoder is ViT, decoder is CNN
    use domain-specific adapters and norms
    '''
    def __init__(
        self,pretrained=None, pretrained_vit_name='vit_base_patch16_224_in21k',
        pretrained_folder='/bigdata/siyiplace/data/skin_lesion',
        img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, 
        depth=12, num_heads=12, mlp_ratio=4., 
        patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_norm=nn.BatchNorm2d, debug=False, adapt_method=False, num_domains=1,
        **kwargs,):
        super().__init__()
        self.pretrained = pretrained
        self.debug = debug
        self.adapt_method = adapt_method
        self.patch_size=patch_size

        self.encoder = ViT_ImageNet(
                    img_size=img_size,num_frames=num_frames,patch_size=patch_size,in_chans=in_chans,
                    embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,
                    patch_embedding_bias=patch_embedding_bias,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,pretrained=pretrained,
                    adapt_method=adapt_method,num_domains=num_domains)
        
        self.prompt_encoder = resnet18(pretrained=True, out_indices=[1,2,3])
        # self.prompt_attn = CrossAttention(
        #    k_dim=embed_dim,q_dim=256, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        # self.combination = nn.Conv2d(256+embed_dim,embed_dim,1,1,0)

        # decoder
        self.aspp = ASPP(in_channels=embed_dim+256,atrous_rates=[6,12,18])
        self.combination_1 = Residual(384,128)
        self.combination_0 = Residual(192,64)
        self.final_conv = nn.Sequential(
                            nn.Conv2d(64,1,3,1,1),
        )

        # # attention
        # self.csattn_0 = ChannelSpatialAttention(embed_dim)
        # self.csattn_1 = ChannelSpatialAttention(256)
        # self.csattn_2 = ChannelSpatialAttention(64)


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
        encoder_out = self.encoder.forward_features(x, d) # (B,N,C)

        if out_seg == False:
            encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': None, 'feat': x}


        # concat 3rd stage
        encoder_out = rearrange(encoder_out,'b (h w) c -> b c h w', h=img_size[0]//self.patch_size,w=img_size[1]//self.patch_size)
        p0,p1,p2 = self.prompt_encoder(x)
        encoder_out = torch.cat((encoder_out,p2),dim=1)
        out = self.aspp(encoder_out)  # (B,C,H/16,W/16)
        out = nn.functional.interpolate(out,size=p1.shape[-2:],mode = 'bilinear', align_corners=False)  # (B,C,H/8,W/8)

        # concat 2rd stage 
        out = torch.cat((out,p1),dim=1)
        out = self.combination_1(out)
        out = nn.functional.interpolate(out,size=p0.shape[-2:],mode = 'bilinear', align_corners=False) # (B,C,H/4,W/4)

        # concat 1st stage
        out = torch.cat((out,p0),dim=1)
        out = self.combination_0(out)  

        
        # upsample
        out = self.final_conv(out)  # (1,H/4,W/4)
        out = nn.functional.interpolate(out,size = img_size,mode = 'bilinear', align_corners=False) # (B,1,H,W)
        
        if out_feat:
            x = nn.functional.adaptive_avg_pool2d(encoder_out,1).reshape(B, -1)
            return {'seg': out, 'feat': x}
        else:
            return {'seg':out}


