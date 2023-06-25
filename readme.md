# MDViT: Multi-domain Vision Transformer for Small Medical Image Segmentation Datasets

## Data
### Data Preparation

* [ISIC 2018 (ISIC)][1]
* [Dermofit Image Library (DMF)][2]
* [Skin Cancer Detection][3]
* [PH2][4]

### Preprocessing

Please run the following command to resize original images into the same dimension (512,512) and then convert and store them as .npy files.
```sh
python Datasets/process_resize.py
```

Use [Datasets/create_meta.ipynb][] to create the csv files for each dataset.

### Training


[1]: https://challenge.isic-archive.com/data/#2018
[2]: https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library
[3]: https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection
[4]: https://www.fc.up.pt/addi/ph2%20database.html
[5]: 
