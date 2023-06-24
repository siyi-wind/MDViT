import sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer')
from Datasets.create_dataset import norm01, Dataset_wrap, Dataset_wrap_csv
import torch
import numpy as np
import pandas as pd
import os
import random

# model files
import yaml
from tqdm import tqdm
from Models.Transformer.UFAT import FATNet
from Models.Transformer.UFAT_for_adapt_KT import FATNet_KT_adapt
from monai.networks.nets import SwinUNETR
from Models.Hybrid_models.TransFuseFolder.TransFuse import TransFuse_S
from Models.Hybrid_models.UTNetFolder.UTNet import UTNet
from Models.Transformer.SwinUnet import SwinUnet
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/courses_firstyear/CPSC533R/BA-Transformer-main')
from BAT_simple.BAT_main import BAT
from Utils.pieces import dice_per_img


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

loaders = {}
for dataset_name in ['isic2018','DMF','SKD','PH2']:
    datas = Dataset_wrap_csv(k_fold='4', use_old_split=True, img_size=256, 
        dataset_name = dataset_name, split_ratio=[0.8,0.2], 
        train_aug=True)
    train_data, val_data, test_data = datas['train'], datas['test'], datas['test']
    loaders[dataset_name] = torch.utils.data.DataLoader(test_data,
                                                batch_size=16,
                                                shuffle=False,
                                                num_workers=2,
                                                pin_memory=True,
                                                drop_last=False)


model_name_list = ['BASE', 'BAT', 'SwinUNTER', 'TransFuse', 'UTNet', 'SwinUnet', 'MDViT']                  
model_path_dic = yaml.load(open('Visualization/model_path.yml'), Loader=yaml.FullLoader)
store_folder = '/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/V_results'

# different models
for model_name in model_name_list:
    if model_name == 'BASE':
        model = FATNet(drop_rate=0.1, drop_path_rate=0.1, conv_norm=torch.nn.BatchNorm2d)
    elif model_name == 'BAT':
        model = BAT(1, 34, 1, 6)
    elif model_name == 'SwinUNTER': 
        model = SwinUNETR(img_size=(256,256), in_channels=3, out_channels=1, feature_size=48, use_checkpoint=False, spatial_dims=2)
    elif model_name == 'TransFuse':
        model = TransFuse_S(pretrained=False)
    elif model_name == 'UTNet':
        model = UTNet(in_chan=3,base_chan=32,num_classes=1,reduce_size=8,block_list='1234',num_blocks=[1,1,1,1],
                        num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    elif model_name == 'SwinUnet':
        model = SwinUnet(img_size=256)
    elif model_name == 'MDViT':
        model = FATNet_KT_adapt(img_size=256, drop_rate=0.1, drop_path_rate=0.1,
                        conv_norm=torch.nn.BatchNorm2d, adapt_method='Sup', num_domains=4, do_detach=False, decoder_name='MLPFM')
    
    
    # for different datasets 'isic2018','PH2','DMF','SKD'
    for dataset_name in ['isic2018','PH2','DMF','SKD']:
        loader = loaders[dataset_name]
        # different training paradigms
        for setting in ['ST', 'JT']:
            value = dataset_name if setting=='ST' else 'JT'
            model.load_state_dict(torch.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
            'V_results/{}/best.pth'.format(model_path_dic[value][model_name])))
            model.cuda()
            model.eval()
            id_list = []
            dataset_list = []
            dice_list = []
            data_len = 0
            for batch in tqdm(loader):
                id = batch['ID']
                img = batch['image'].float().cuda()
                label = batch['label'].float().cpu().numpy()
                domain_label = batch['set_id']
                domain_label_oh = torch.nn.functional.one_hot(domain_label, 4).float().cuda()
                data_len += img.shape[0]
                with torch.no_grad():
                        if model_name == 'MDViT':
                            output = model(img, domain_label_oh)
                            output = output[0]
                        elif model_name == 'TransFuse':
                            _, _, output = model(img) 
                        elif model_name == 'BAT':
                            output, _ = model(img)
                        else:
                            output = model(img)
                        output = torch.sigmoid(output).detach().cpu().numpy() > 0.5
                        dice =  dice_per_img(label, output)
                        dice_list.extend(dice)
                        id_list.extend(id)
                        dataset_list.extend(batch['set_name'])
                        # break
            df_path = store_folder+'/{}.csv'.format(dataset_name)
            if os.path.exists(df_path):
                df = pd.read_csv(df_path, dtype={'ID': str})
                df['{}_{}'.format(model_name,setting)]=dice_list
                # df.to_csv(df_path, index=True)
            else:
                df = pd.DataFrame({
                    'index': list(range(data_len)),
                    'ID': id_list,
                    'dataset': dataset_list,
                    '{}_{}'.format(model_name,setting): dice_list,
                })
            df.to_csv(df_path, index=False)
            # break
        # break
    # print(id_list)
    # break

