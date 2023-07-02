# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import collections

# from .vision_transformer import VisionTransformer, _cfg, VisionTransformer_adapt
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/MDViT/')
from Models.Hybrid_models.TransFuseFolder.vision_transformer import VisionTransformer, _cfg, VisionTransformer_adapt


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        pe = self.pos_embed

        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x



class DeiT_adapt(VisionTransformer_adapt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

    def forward(self, x, domain_label):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        pe = self.pos_embed

        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, domain_label)

        x = self.norm(x)
        return x


def load_pretrain(model, pre_s_dict):
    ''' Load state_dict in pre_model to model
    Solve the problem that model and pre_model have some different keys'''
    s_dict = model.state_dict()
    # use new dict to store states, record missing keys
    missing_keys = []
    new_state_dict = collections.OrderedDict()
    for key in s_dict.keys():
        if key in pre_s_dict.keys():
            new_state_dict[key] = pre_s_dict[key]
        else:
            new_state_dict[key] = s_dict[key]
            missing_keys.append(key)
    print('{} keys are not in the pretrain model:'.format(len(missing_keys)), missing_keys)
    # load new s_dict
    model.load_state_dict(new_state_dict)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, pretrained_folder=None, **kwargs):
    model = DeiT(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load(pretrained_folder+'/pretrained/deit_small_patch16_224-cd65a155.pth')
        # model = load_pretrain(model, ckpt)
        model.load_state_dict(ckpt['model'], strict=False)
    
    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    # pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)
    pe = F.interpolate(pe, size=(14, 14), mode='bilinear', align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model


@register_model
def deit_small_patch16_224_adapt(pretrained=False, pretrained_folder=None, num_domains=4, **kwargs):
    model = DeiT_adapt(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_domains=num_domains, **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     ckpt = torch.load(pretrained_folder+'/pretrained/deit_small_patch16_224-cd65a155.pth')
    #     model.load_state_dict(ckpt['model'], strict=False)
    if pretrained:
        ckpt = torch.load(pretrained_folder+'/pretrained/deit_small_patch16_224-cd65a155.pth')
        model = load_pretrain(model, ckpt)
    
    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    # pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)
    # pe = F.interpolate(pe, size=(14, 14), mode='bilinear', align_corners=True)
    pe = F.interpolate(pe, size=(16, 16), mode='bilinear', align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model


@register_model
def deit_base_patch16_224(pretrained=False, pretrained_folder=None,**kwargs):
    model = DeiT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     ckpt = torch.load('pretrained/deit_base_patch16_224-b5f2ef4d.pth')
    #     model.load_state_dict(ckpt['model'], strict=False)
    if pretrained:
        ckpt = torch.load(pretrained_folder+'/pretrained/deit_base_patch16_224-b5f2ef4d.pth')['model']
        model = load_pretrain(model, ckpt)

    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    # pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)
    # pe = F.interpolate(pe, size=(14, 14), mode='bilinear', align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = DeiT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(ckpt["model"])

    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    pe = F.interpolate(pe, size=(24, 32), mode='bilinear', align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model



if __name__ == '__main__':
    x = torch.randn(5,3,256,256)
    domain_label = torch.randint(0,4,(5,))
    domain_label = torch.nn.functional.one_hot(domain_label, 4).float()
    net = deit_small_patch16_224_adapt(pretrained=False, num_domains=4, pretrained_folder='/bigdata/siyiplace/data/skin_lesion')
    # net = deit_small_patch16_224(pretrained=True, pretrained_folder='/bigdata/siyiplace/data/skin_lesion')
    # print(net.state_dict().keys())
    # net = deit_small_patch16_224(pretrained=True, pretrained_folder='/bigdata/siyiplace/data/skin_lesion')
    y = net(x, domain_label)
    print(y.shape)
    param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of parameter: {param/1e6} M")