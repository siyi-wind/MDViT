'''
Input: downloaded datasets
Process: resize, change jpg to npy, store images and labels to Image/, Label/
From https://github.com/jcwang123/BA-Transformer
'''

import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def process_isic2018(
    isic2018_origin_folder = '/bigdata/siyiplace/data/skin_lesion/2018_raw_data',
    dim=(512, 512), isic2018_proceeded_folder='/bigdata/siyiplace/data/skin_lesion/isic2018'): # '/raid/wjc/data/skin_lesion/isic2018/')
    image_dir_path = isic2018_origin_folder+'/ISIC2018_Task1-2_Training_Input/'    
    mask_dir_path =  isic2018_origin_folder+'/ISIC2018_Task1_Training_GroundTruth/'  
    # '/raid/wl/2018_raw_data/ISIC2018_Task1_Training_GroundTruth/'

    image_path_list = os.listdir(image_dir_path)
    mask_path_list = os.listdir(mask_dir_path)

    image_path_list = list(filter(lambda x: x[-3:] == 'jpg', image_path_list))
    mask_path_list = list(filter(lambda x: x[-3:] == 'png', mask_path_list))
    
    # align masks and inputs
    image_path_list.sort()
    mask_path_list.sort()

    print('number of images: {}, number of masks: {}'.format(len(image_path_list), len(mask_path_list)))

    # ISBI Dataset
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        if image_path[-3:] == 'jpg':
            print(image_path)
            assert os.path.basename(image_path)[:-4].split(
                '_')[1] == os.path.basename(mask_path)[:-4].split('_')[1]
            _id = os.path.basename(image_path)[:-4].split('_')[1]
            image_path = os.path.join(image_dir_path, image_path)
            mask_path = os.path.join(mask_dir_path, mask_path)
            image = plt.imread(image_path)
            mask = plt.imread(mask_path)

            image_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)

            save_dir_path = isic2018_proceeded_folder + '/Image'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), image_new)

            save_dir_path = isic2018_proceeded_folder + '/Label'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), mask_new)
            


def process_PH2(
    PH2_origin_folder = '/bigdata/siyiplace/data/skin_lesion/PH2_rawdata',
    PH2_proceeded_folder = '/bigdata/siyiplace/data/skin_lesion/PH2'):
    
    PH2_images_path = os.path.join(PH2_origin_folder,'/PH2Dataset/PH2_Dataset_images')
    path_list = os.listdir(PH2_images_path)
    path_list.sort()

    for path in path_list:
        image_path = os.path.join(PH2_images_path, path,
                                  path + '_Dermoscopic_Image', path + '.bmp')
        label_path = os.path.join(PH2_images_path, path, path + '_lesion',
                                  path + '_lesion.bmp')
        image = plt.imread(image_path)
        label = plt.imread(label_path)
        label = label[:, :, 0]

        dim = (512, 512)
        image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        label_new = cv2.resize(label, dim, interpolation=cv2.INTER_AREA)

        image_save_path = os.path.join(
            PH2_proceeded_folder,'/Image',
            path + '.npy') #  '/data2/cf_data/skinlesion_segment/PH2_rawdata/PH2/Image'
        label_save_path = os.path.join(
            PH2_proceeded_folder,'/Label',
            path + '.npy') # /data2/cf_data/skinlesion_segment/PH2_rawdata/PH2/Label

        np.save(image_save_path, image_new)
        np.save(label_save_path, label_new)


def process_SKD(
    SKD_images_folder = '/bigdata/siyiplace/data/skin_lesion/skin_cancer_detection',
    SKD_proceeded_folder = '/bigdata/siyiplace/data/skin_lesion/SKD'):
    '''
    SKin Cancer Detection dataset
    '''
    
    SKD_images_path1 = '{}/skin_image_data_set-1/Skin Image Data Set-1/skin_data/melanoma/'.format(SKD_images_folder)
    SKD_images_path2 = '{}/skin_image_data_set-2/Skin Image Data Set-2/skin_data/notmelanoma/'.format(SKD_images_folder)

    
    for images_path in [SKD_images_path1, SKD_images_path2]:
        for dataset_name in ['dermis', 'dermquest']:
            path_list = os.listdir('{}{}'.format(images_path, dataset_name))

            for path in path_list:
                if path[-4:] == '.jpg':
                    image_path = os.path.join('{}{}'.format(images_path, dataset_name), path)
                    label_path = os.path.join('{}{}'.format(images_path, dataset_name), path[:-8]+'contour.png')
                else: continue

                image = plt.imread(image_path)
                label = plt.imread(label_path)
                dim = (512, 512)
                image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                label_new = cv2.resize(label, dim, interpolation=cv2.INTER_AREA)

                image_save_path = os.path.join(
                    SKD_proceeded_folder,'/Image',
                    dataset_name+'_'+path[:-4] + '.npy')
                label_save_path = os.path.join(
                    SKD_proceeded_folder,'/Label',
                    dataset_name+'_'+path[:-4] + '.npy') 
                np.save(image_save_path, image_new)
                np.save(label_save_path, label_new)
                

def process_DMF(
    DMF_images_folder = '/bigdata/siyiplace/data/skin_lesion/DMF_origin',
    DMF_proceeded_folder = '/bigdata/siyiplace/data/skin_lesion/DMF'):
    '''
    Dermofit (DMF) dataset
    '''
    
    DMF_images_path = '{}/images'.format(DMF_images_folder)

    path_list = os.listdir(DMF_images_path)
    path_list.sort()

    for path in tqdm(path_list):
        image_path = os.path.join(DMF_images_path, path,
                                  path + '.png')
        label_path = os.path.join(DMF_images_path, path, path + 'mask.png')
        image = plt.imread(image_path)
        label = plt.imread(label_path)

        dim = (512, 512)
        image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image_new = np.clip(image_new*255, 0, 255).astype(np.uint8) if image_new.max() < 1.2 else image_new
        label_new = cv2.resize(label, dim, interpolation=cv2.INTER_AREA)

        image_save_path = os.path.join(
            DMF_proceeded_folder,'/Image',
            path + '.npy') 
        label_save_path = os.path.join(
            DMF_proceeded_folder,'/Label',
            path + '.npy') 

        np.save(image_save_path, image_new)
        np.save(label_save_path, label_new)


if __name__ == '__main__':
    process_isic2018()
    process_PH2()
    process_SKD()
    process_DMF()
