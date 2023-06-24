'''
extract image features using resnet  get 2048 features 
'''
import sys
sys.path.append('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/skin-lesion-segmentation-transformer')
from Datasets.create_dataset import norm01, SkinClasDataset
import torch
import numpy as np
import pandas as pd
import os
import torchvision
import random
import h5py
from tqdm import tqdm
import timm


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


loaders = {}
model_name = 'FATNet_KT_adapt'
for dataset_name in ['isic2018','PH2','DMF','SKD']:
    dataset = SkinClasDataset(dataset_name, 224)
    loaders[dataset_name] = torch.utils.data.DataLoader(dataset,
                                                batch_size=16,
                                                shuffle=False,
                                                num_workers=2,
                                                pin_memory=True,
                                                drop_last=False)
exp_name = 'V4'

# resnet
if model_name == 'resnet':
    network = torchvision.models.resnet101(pretrained=True)
    network = torch.nn.Sequential(*list(network.children())[:-1])
    # coat
elif model_name == 'coat':
    network = timm.create_model('coat_lite_small', pretrained=True) 
elif model_name == 'vit':
    network = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
elif model_name == 'FATNet':
    from Models.Transformer.UFAT import FATNet  
    network = FATNet(drop_rate=0.1, drop_path_rate=0.1, conv_norm=torch.nn.BatchNorm2d)
    network.load_state_dict(torch.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
    'V_results/{}/ca_four_fold4_FATNet_20221111_1917/best.pth'.format(exp_name)))
elif model_name == 'FATNet_adapt':
    from Models.Transformer.UFAT_for_adapt import FATNet_adapt
    network = FATNet_adapt(img_size=256, drop_rate=0.1, drop_path_rate=0.1,
        conv_norm=torch.nn.BatchNorm2d, adapt_method='Sup', num_domains=4, 
        feature_dim=512, do_detach=False)
    network.load_state_dict(torch.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
    'N_results/{}/ca_SupDo_fold4_FATNet_adapt_20221208_1858/best.pth'.format(exp_name))) 
    # dynamic_np = np.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
    # 'N_results/{}/ca_SupDo_fold4_FATNet_adapt_20221208_1858/dynamic_info.npy'.format(exp_name))
elif model_name == 'FATNet_dynamic':
    from Models.Transformer.UFAT_for_adapt import FATNet_adapt
    network = FATNet_adapt(img_size=256, drop_rate=0.1, drop_path_rate=0.1,
        conv_norm=torch.nn.BatchNorm2d, adapt_method='Sup', num_domains=4, 
        feature_dim=512, do_detach=False)
    network.load_state_dict(torch.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
    'N_results/{}/ca_knn_dyn701_oh_SupDo_FATNet_adapt_20221208_1901/best.pth'.format(exp_name))) 
    dynamic_np = np.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
    'N_results/{}/ca_knn_dyn701_oh_SupDo_FATNet_adapt_20221208_1901/dynamic_info.npy'.format(exp_name))
elif model_name == 'FATNet_KT_adapt':
    from Models.Transformer.UFAT_for_adapt_KT import FATNet_KT_adapt
    network = FATNet_KT_adapt(img_size=256, drop_rate=0.1, drop_path_rate=0.1,
    conv_norm=torch.nn.BatchNorm2d, adapt_method='Sup', num_domains=4, do_detach=False, decoder_name='MLPFM')
    # network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
    'V_results/{}/ca_4detKT_nodetachMLPFM_SupDo_fold_FATNet_KT_adapt_20230101_1241/best.pth'.format(exp_name)))
elif model_name == 'MSNet':
    from Models.CNN.MS_Net import MSNet
    network = MSNet(num_domains=4)
    network.load_state_dict(torch.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
    'N_results/{}/ca_4_fold_MSNet_20221222_1050/best.pth'.format(exp_name)))
elif model_name == 'FATNet_KT_adapt_M':
    from Models.Transformer.UFAT_for_adapt_KT import FATNet_KT_adapt_M
    network = FATNet_KT_adapt_M(img_size=256, drop_rate=0.1, drop_path_rate=0.1,
    conv_norm=torch.nn.BatchNorm2d, adapt_method='Sup', num_domains=4, do_detach=False, decoder_name='MLP')  
    network.load_state_dict(torch.load('/ubc/ece/home/ra/grads/siyi/Research/skin_lesion_segmentation/results/'+
    'N_results/{}/ca_4detktlossKT_MLPFM_SupDo_fold_FATNet_KT_adapt_20230104_1352/best.pth'.format(exp_name)))
        
elif model_name == 'FATNet_ENKT_adapt':
    from Models.Transformer.UFAT_for_adapt_KT import FATNet_ENKT_adapt
    model = FATNet_ENKT_adapt(img_size=256, drop_rate=0.1, drop_path_rate=0.1,
    conv_norm=torch.nn.BatchNorm2d, adapt_method='Sup', num_domains=4, do_detach=False, 
    decoder_name='MLPFM',ensemble_method='att')


network.cuda()
network.eval()


data_folder = '/bigdata/siyiplace/data/skin_lesion'
for dataset_name in ['isic2018','PH2','DMF','SKD']:
    loader = loaders[dataset_name]
    features = []
    diagnosis_list = []
    d_label_list = []
    dataset_id_list = []
    DC_id_list = []
    for batch in tqdm(loader):
        img = batch['image'].cuda().float()
        diagnosis = batch['diagnosis_id']
        dataset_id = batch['set_id']
        DC_id = batch['DC_id']
        with torch.no_grad():
            # resnet
            if model_name == 'resnet':
                output = network(img)
                output = output.view(output.size(0), -1).cpu().numpy()
            # coat
            elif model_name in ['coat', 'vit']:
                output = network.forward_features(img)
                output = output[:,0].cpu().numpy()
            elif model_name == 'FATNet':
                output = network(img,out_feat=True,out_seg=False)
                output = output['feat'].cpu().numpy()
            elif model_name == 'FATNet_adapt':
                domain_label = batch['set_id']
                domain_label_oh = torch.nn.functional.one_hot(domain_label, 4).float().cuda()
                output = network(img,domain_label_oh,out_feat=True,out_seg=False)
                output = output['feat'].cpu().numpy()
                four_id = batch['four_id'].cpu().numpy()
                # d_label = dynamic_np[four_id,0].astype('int64')
                # d_label_list.extend(d_label)
            elif model_name == 'FATNet_dynamic':
                four_id = batch['four_id'].cpu().numpy()
                d_label = dynamic_np[four_id,0].astype('int64')
                d_label_oh = torch.nn.functional.one_hot(torch.from_numpy(d_label),4).float().cuda()
                output = network(img,d_label_oh,out_feat=True,out_seg=False)
                output = output['feat'].cpu().numpy()
                d_label_list.extend(d_label)
            elif model_name == 'FATNet_KT_adapt' or model_name == 'FATNet_KT_adapt_M':
                domain_label = batch['set_id']
                domain_label_oh = torch.nn.functional.one_hot(domain_label, 4).float().cuda()
                output = network(img,domain_label_oh,out_feat=True,out_seg=False)
                output = output['feat'].cpu().numpy()
            elif model_name == 'MSNet':
                domain_label = batch['set_id']
                d = str(domain_label[0].item())
                output = network(img,d,out_feat=True,out_seg=False)
                output = output['feat'].cpu().numpy()
            # print(output.shape)
            # print(d_label_list)
            features.append(output)
            diagnosis_list.extend(diagnosis)
            dataset_id_list.extend(dataset_id)
            DC_id_list.extend(DC_id)
        # break
    features = np.concatenate(features, axis=0)
    print('features shape:', features.shape)
    hdf_path = data_folder+'/{}/{}_{}_feature_{}.h5'.format(dataset_name,exp_name, model_name,dataset_name)
    f = h5py.File(hdf_path, 'w')
    f['feature'] = features
    f['diagnosis_id'] = diagnosis_list
    f['d_label'] = d_label_list
    f['dataset_id'] = dataset_id_list
    f['DC_id'] = DC_id_list
    f.close()

    # break
    