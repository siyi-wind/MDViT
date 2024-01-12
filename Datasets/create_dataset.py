'''
Split dataset as train, test, val  6:2:2
use function dataset_wrap, return {train:, val:, test:} torch dataset

datasets names: isic2018, PH2, DMF, SKD
'''

import os
import json
import torch
import random
import numpy as np
from torchvision import transforms
import albumentations as A
import pandas as pd
import pandas as pd

dataset_indices = {
    'isic2018': 0,
    'PH2': 1,
    'DMF': 2,
    'SKD': 3,
}

def norm01(x):
    return np.clip(x, 0, 255) / 255


def Dataset_wrap_csv(k_fold='No', use_old_split=True, img_size=384, dataset_name='isic2018', split_ratio=[0.8, 0.2], train_aug=False,
    data_folder='/bigdata/siyiplace/data/skin_lesion'):
    '''
    use train val test csv to load the whole datasets in order to include domain (dataset) label
    if k_fold is a number, means we use k-fold to do experiments, load k_fold index data. default 5 folders
    if use_old_split, load existing train, test paths
    dataset_name: choose which dataset to load
        random split train val test set by split_ratio
        save train test id
    return train val test in a dic
    '''
    data_dic = {}
    data_path = '{}/{}/'.format(data_folder, dataset_name)
    # do k fold loading
    if k_fold != 'No':
        if use_old_split:
            try:
                train_df = pd.read_csv(data_path+'train_meta_kfold_{}.csv'.format(k_fold), dtype={'ID': str})
                test_df = pd.read_csv(data_path+'test_meta_kfold_{}.csv'.format(k_fold), dtype={'ID': str})
                data_dic['train'] = SkinDataset_csv(dataset_name, img_size, train_df, use_aug=train_aug, data_path=data_path)
                data_dic['test'] = SkinDataset_csv(dataset_name, img_size, test_df, use_aug=False, data_path=data_path)
                data_size = len(data_dic['train'])+len(data_dic['test'])
                print('{} has {} samples, {} are used to train, {} are used to test. \n 5 Folder -- Use {}'
                .format(dataset_name, data_size, len(data_dic['train']), len(data_dic['test']), k_fold))
                return data_dic
            except:
                print('No existing k_folder files, start creating new splitting....')
        
        print('use new split')
        df = pd.read_csv(data_path+'meta_{}.csv'.format(dataset_name),dtype={'ID': str})
        data_size = len(df)
        # # random split train test based on train_ratio
        index_list = list(range(data_size))
        random.Random(42).shuffle(index_list)
        split_size = int(data_size/5.0+0.5)  # one split size, 5 splits
        split_ids = [0,split_size,split_size*2,split_size*3,split_size*4,len(index_list)]
        for i in range(5):
            train_df = df.iloc[index_list[:split_ids[i]]+index_list[split_ids[i+1]:]]
            test_df = df.iloc[index_list[split_ids[i]:split_ids[i+1]]]
            # save train, test csv
            train_df.to_csv(data_path+'train_meta_kfold_{}.csv'.format(i), header=df.columns, index=False)
            test_df.to_csv(data_path+'test_meta_kfold_{}.csv'.format(i), header=df.columns, index=False)

        train_df = pd.read_csv(data_path+'train_meta_kfold_{}.csv'.format(k_fold), dtype={'ID': str})
        test_df = pd.read_csv(data_path+'test_meta_kfold_{}.csv'.format(k_fold), dtype={'ID': str})
        data_dic['train'] = SkinDataset_csv(dataset_name, img_size, train_df, use_aug=train_aug, data_path=data_path)
        data_dic['test'] = SkinDataset_csv(dataset_name, img_size, test_df, use_aug=False, data_path=data_path)
        assert data_size == len(data_dic['train'])+len(data_dic['test'])
        print('Finish creating new 5 folders. {} has {} samples, {} are used to train, {} are used to test. \n 5 Folder -- Use {}'
        .format(dataset_name, data_size, len(train_df), len(test_df), k_fold))
        return data_dic

    # do no k fold
    if use_old_split:
        # in case these files are not exist
        try: 
            train_df = pd.read_csv(data_path+'train_meta_{}.csv'.format(int(split_ratio[0]*100)), dtype={'ID': str})
            test_df = pd.read_csv(data_path+'test_meta_{}.csv'.format(int(split_ratio[1]*100)), dtype={'ID': str})
            data_dic['train'] = SkinDataset_csv(dataset_name, img_size, train_df, use_aug=train_aug, data_path=data_path)
            data_dic['test'] = SkinDataset_csv(dataset_name, img_size, test_df, use_aug=False, data_path=data_path)
            data_size = len(data_dic['train'])+len(data_dic['test'])
            print('{} has {} samples, {} are used to train, {} are used to test. \n The split ratio is {}'
            .format(dataset_name, data_size, len(data_dic['train']), len(data_dic['test']), split_ratio))
            return data_dic
        except:
            print('No existing split files, start creating new splitting....')
    
    print('use new split')
    df = pd.read_csv(data_path+'meta_{}.csv'.format(dataset_name),dtype={'ID': str})
    data_size = len(df)

    # # random split train test based on train_ratio
    index_list = list(range(data_size))
    random.Random(42).shuffle(index_list)
    train_df = df.iloc[index_list[: int(data_size*split_ratio[0])]]
    test_df = df.iloc[index_list[int(data_size*split_ratio[0]) : ]]

    print('{} has {} samples, {} are used to train, {} are used to test. \n The split ratio is {}'
    .format(dataset_name, data_size, len(train_df), len(test_df), split_ratio))

    # save train, test csv
    train_df.to_csv(data_path+'train_meta_{}.csv'.format(int(split_ratio[0]*100)), header=df.columns, index=False)
    test_df.to_csv(data_path+'test_meta_{}.csv'.format(int(split_ratio[1]*100)), header=df.columns, index=False)

    data_dic['train'] = SkinDataset_csv(dataset_name, img_size, train_df, use_aug=train_aug, data_path=data_path)
    data_dic['test'] = SkinDataset_csv(dataset_name, img_size, test_df, use_aug=False, data_path=data_path)

    return data_dic


class SkinDataset_csv(torch.utils.data.Dataset):
    def __init__(self, dataset_name, img_size, df, use_aug=False,  
        data_path='/bigdata/siyiplace/data/skin_lesion/isic2018/'):
        super(SkinDataset_csv, self).__init__()
        
        self.dataset_name = dataset_name
        self.root_dir = data_path
        self.df = df
        self.use_aug = use_aug

        self.num_samples = len(self.df)

        p = 0.5
        self.aug_transf = A.Compose([
            A.Resize(img_size, img_size),
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            A.RandomBrightnessContrast(p=p),
        ])
        self.transf = A.Compose([
            A.Resize(img_size, img_size),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.df.loc[self.df.index[index]]
        img_path = self.root_dir + 'Image/{}.npy'.format(row['ID'])
        label_path = self.root_dir + 'Label/{}.npy'.format(row['ID'])
        diagnosis = row['diagnosis']
        diagnosis_id = row['diagnosis_id']

        img_data = np.load(img_path)
        label_data = np.load(label_path) > 0.5

        if self.use_aug:
            tsf = self.aug_transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        else:
            tsf = self.transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        img_data, label_data = tsf['image'], tsf['mask']
        
        img_data = norm01(img_data)
        label_data = np.expand_dims(label_data, 0)

        img_data = torch.from_numpy(img_data).float()
        label_data = torch.from_numpy(label_data).float()

        img_data = img_data.permute(2, 0, 1)
        img_data = self.normalize(img_data)


        return{
            'ID': row['ID'],
            'set_name': self.dataset_name, 
            'set_id': dataset_indices[self.dataset_name],
            'image_path': img_path,
            'label_path': label_path,
            'diagnosis': diagnosis,
            'diagnosis_id': diagnosis_id,       # i.e., class label
            'image': img_data,
            'label': label_data,
        }


    def __len__(self):
        return self.num_samples



# ==================================================================================================================================
# Load the whole dataset and don't split train test sets
class SkinClasDataset(torch.utils.data.Dataset):
    '''
    Use csv file to load the whole dataset. Have diagnosis labels
    used for generate tsne
    '''
    def __init__(self, dataset_name, img_size,  
        data_folder='/bigdata/siyiplace/data/skin_lesion'):
        super(SkinClasDataset, self).__init__()
        
        self.dataset_name = dataset_name
        self.root_dir = '{}/{}/'.format(data_folder, dataset_name)
        self.df = pd.read_csv(self.root_dir+'meta_{}.csv'.format(dataset_name), dtype={'ID': str})

        self.num_samples = len(self.df)

        self.transf = A.Compose([
            A.Resize(img_size, img_size),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    
    def __getitem__(self, index):
        img_path = self.root_dir + 'Image/{}.npy'.format(self.df.loc[self.df.index[index], 'ID'])
        label_path = self.root_dir + 'Label/{}.npy'.format(self.df.loc[self.df.index[index], 'ID'])
        diagnosis = self.df.loc[self.df.index[index], 'diagnosis']
        diagnosis_id = self.df.loc[self.df.index[index], 'diagnosis_id']
        img_data = np.load(img_path)
        label_data = np.load(label_path) > 0.5

        tsf = self.transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        img_data, label_data = tsf['image'], tsf['mask']
        
        img_data = norm01(img_data)
        label_data = np.expand_dims(label_data, 0)

        img_data = torch.from_numpy(img_data).float()
        label_data = torch.from_numpy(label_data).float()

        img_data = img_data.permute(2, 0, 1)
        img_data = self.normalize(img_data)

        return{
            'set_name': self.dataset_name, 
            'set_id': dataset_indices[self.dataset_name],
            'image_path': img_path,
            'label_path': label_path,
            'diagnosis': diagnosis,
            'diagnosis_id': diagnosis_id,
            'image': img_data,
            'label': label_data,
        }

    def __len__(self):
        return self.num_samples



if __name__ == '__main__':
    # test
    datasets = Dataset_wrap_csv(k_fold='No', use_old_split=False, dataset_name='SKD',dynamic=True)
    for key in datasets.keys():
        print(len(datasets[key]))
    dataloader = torch.utils.data.DataLoader(datasets['train'],
                                            batch_size=3,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True,
                                            drop_last=True)
    batch = next(iter(dataloader))
    print(batch['four_id'])


    pass


