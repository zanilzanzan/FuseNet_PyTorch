# FuseNet
## [[Caffe]](https://github.com/tum-vision/fusenet)

Joint scene classification and semantic segmentation using FuseNet architecture from [FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture](https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf). Potential effects of additional scene classification loss on the overall semantic segmentation quality are tested.

<img src="images/framework.png" width="800px"/>

### Dependencies
- python 2.7
- [PyTorch 0.1.12 CUDA 8.0 version](http://pytorch.org/previous-versions/)
- In addition, please `pip install -r requirements.txt` to install the following packages:
    - TODO

### Datasets 

#### NYU

1.Simply download the processed .h5py dataset with 40 annotations and 10 classes here: [train + test set](https://www.dropbox.com/s/n664gpdt7rkuejj/nyu_class_10_db.h5?dl=0). (TODO: When running Train_FuseNet.py download the dataset automatically if not found)


#### SUNRGBD

1. TODO: Download the dataset [here](link)

2. TODO: Download class weight file [here](link) .

### Training
- To train Fusenet run `Train_FuseNet.py`. Dataset choice is manually implemented in the script for now. The dataset is taken and prepared by `utils/data_utils_class.py`, therefore make sure to give the correct path in the script. (TODO: Pass the arguments instead of manually entering in the script) 

- Note: VGG weights are downloaded automatically at the beginning of the training process. Depth layers weights will also be initialized with their vgg16 equivalent layers. However, for 'conv1_1' the weights will be averaged to fit one channel depth input (3, 3, 3, 64) -> (3, 3, 1, 64)

### Evaluation
- To evaluate Fusenet results, locate the trained model file and use `FuseNet_Class_Plots.ipynb`. 

### To-Do
- [ ] Modularize the code 

### Citing FuseNet
Caner Hazirbas, Lingni Ma, Csaba Domokos and Daniel Cremers, _"FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture"_, in proceedings of the 13th Asian Conference on Computer Vision, 2016. ([pdf](https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf))

    @inproceedings{fusenet2016accv,
     author    = "C. Hazirbas and L. Ma and C. Domokos and D. Cremers",
     title     = "FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture",
     booktitle = "Asian Conference on Computer Vision",
     year      = "2016",
     month     = "November",
    }
