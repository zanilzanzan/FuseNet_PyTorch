# FuseNet

This repository contains PyTorch implementation of FuseNet architecture from the paper
[FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture](https://pdfs.semanticscholar.org/9360/ce51ec055c05fd0384343792c58363383952.pdf). 
Initial model's capability has been extended to perform joint scene classification and 
semantic segmentation. Potential effects of scene classification, as an auxiliary task, 
on the overall semantic segmentation quality (and vice versa) are investigated. 

Other implementations of FuseNet:
[[Caffe]](https://github.com/tum-vision/fusenet) 
[[PyTorch]](https://github.com/MehmetAygun/fusenet-pytorch)

<p><img src="images/framework_class.jpg" width="700px" style="margin:0px 45px"/></p>

## Prerequisites
- python 3.6
- [PyTorch 0.4.0 CUDA 9.1 version](https://pytorch.org/get-started/previous-versions/)

## Datasets 

### [NYU-Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
1. Simply download the preprocessed dataset, in HDF5 format, with 40 semantic-segmentation and 10 scene classes here: [train + test set](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/nyu_class_10_db.h5)
2. Preprocessed dataset contains 1449 (train: 795, test: 654) RGB-D images with 320x240 resolution, their semantic-segmentation and scene-type annotations.
3. Originially, NYU-Depth V2 dataset has 27 distinct scene types associated with the images. The number has been redcued to 10 classes (9 most common categories and the rest) 
based on the mapping [Gupta et al.](http://saurabhg.web.illinois.edu/pdfs/gupta2013perceptual.pdf) used. Scene mapping text file can be accessed [here](utils/text/nyu_scene_mapping.txt).

### [SUNRGBD](http://rgbd.cs.princeton.edu/)
This section will be updated soon.

## Training
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
