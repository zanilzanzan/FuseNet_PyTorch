# FuseNet

This repository contains PyTorch implementation of FuseNet architecture from the paper
[FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture](https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf). 
Initial model's capability has been extended to perform joint scene classification and 
semantic segmentation. Potential effects of scene classification, as an auxiliary task, 
on the overall semantic segmentation quality (and vice versa) are investigated. 

Other implementations of FuseNet:
[[Caffe]](https://github.com/tum-vision/fusenet) 
[[PyTorch]](https://github.com/MehmetAygun/fusenet-pytorch)

<img src="images/framework_class.jpg" width="800px"/>

### Dependencies
- python 3.6
- [PyTorch 0.4.0 CUDA 9.1 version](https://pytorch.org/get-started/previous-versions/)

### Datasets 

#### NYU

1. Simply download the processed .h5py dataset with 40 annotations and 10 classes here: [train + test set](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/nyu_class_10_db.h5)

2. Scene mapping can be found [here](utils/text/nyu_scene_mapping.txt)


#### SUNRGBD

1. Download the dataset [here](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/sun/sn_class_10_db.h5)

2. Class weights are set in [FuseNet.py](FuseNet.py#L306). In case of an update on the training set, please adjust the weights, accordingly.

### Training
- To train Fusenet run `Train_FuseNet.py`. Dataset choice is manually implemented in the script for now. The dataset is taken and prepared by `utils/data_utils_class.py`, therefore make sure to give the correct path in the script.

- Note: VGG weights are downloaded automatically at the beginning of the training process. Depth layers weights will also be initialized with their vgg16 equivalent layers. However, for 'conv1_1' the weights will be averaged to fit one channel depth input (3, 3, 3, 64) -> (3, 3, 1, 64)

### Evaluation
- To evaluate Fusenet results, locate the trained model file and use `FuseNet_Class_Plots.ipynb`.
- NYU models can be downloaded from
    - [checkpoint25](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/checkpoint25.pth.tar)
    - [model_best19](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/model_best19.pth.tar)
    - [checkpoint_class_27](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/checkpoint_class_27.pth.tar)
    - [model_best_class_24](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/model_best_class_24.pth.tar)

### Citing FuseNet
Caner Hazirbas, Lingni Ma, Csaba Domokos and Daniel Cremers, _"FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture"_, in proceedings of the 13th Asian Conference on Computer Vision, 2016. ([pdf](https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf))

    @inproceedings{fusenet2016accv,
     author    = "C. Hazirbas and L. Ma and C. Domokos and D. Cremers",
     title     = "FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture",
     booktitle = "Asian Conference on Computer Vision",
     year      = "2016",
     month     = "November",
    }
