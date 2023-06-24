MDViT: Multi-domain Vision Transformer for Small Medical Image Segmentation Datasets
======================================

Data
---------------------
1. Data Preparation
* [ISIC 2018 (ISIC)][1]
* [Dermofit Image Library (DMF)][2]
* [Skin Cancer Detection]{3}
* [PH2][4]

2. Preprocessing
Please run the following command to resize original images into the same dimension (512,512) and convert and store them as .npy files.
```sh
python Datasets/process_resize.py
```
