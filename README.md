# FuseNet

This repository contains PyTorch implementation of FuseNet architecture from the paper
[FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture](https://pdfs.semanticscholar.org/9360/ce51ec055c05fd0384343792c58363383952.pdf). 
Initial model's capability has been extended to perform joint scene classification and semantic segmentation. Potential effects of scene classification, as an auxiliary task, 
on overall semantic segmentation quality (and vice versa) are investigated with this project. 

Other implementations of FuseNet:
[[Caffe]](https://github.com/tum-vision/fusenet) 
[[PyTorch]](https://github.com/MehmetAygun/fusenet-pytorch)

<p align="center"><img src="images/framework_class.jpg" width="700px"/></p>

## Installation
Prerequisites:
- python 3.6
- Nvidia GPU + CUDA cuDNN

Clone the repository and install the required packages:
```bash
git clone https://github.com/zanilzanzan/FuseNet_PyTorch
cd FuseNet_PyTorch
pip install -r requirements.txt
```
## Datasets 

### [NYU-Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- Simply, create a directory named datasets in the main project directory and in datasets directory download the preprocessed dataset, in HDF5 format, with 40 semantic-segmentation and 10 scene classes here: [train + test set](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/nyu_class_10_db.h5)
- Preprocessed dataset contains 1449 (train: 795, test: 654) RGB-D images with 320x240 resolution, their semantic-segmentation and scene-type annotations.
- Depth image values have been normalized so that they fall into 0-255 range. 
- Originially, NYU-Depth V2 dataset has 27 distinct scene types associated with the images. The number has been redcued to 10 classes (9 most common categories and the rest) 
based on the mapping [Gupta et al.](http://saurabhg.web.illinois.edu/pdfs/gupta2013perceptual.pdf) used. Scene mapping text file can be accessed [here](utils/text/nyu_scene_mapping.txt).

### [SUNRGBD](http://rgbd.cs.princeton.edu/)
This section will be updated soon.

## Training
- To train FuseNet, run `fusenet_train.py` by providing the path of the dataset. 
- If you would like to train a FuseNet model with the classification head, provide `--use_class True`   
- Note: VGG weights are downloaded automatically at the beginning of the training process. Depth layers weights will also be initialized with their vgg16 equivalent layers. However, for 'conv1_1' the weights will be averaged to fit one channel depth input (3, 3, 3, 64) -> (3, 3, 1, 64)
- Example training commnads can be found below.

### Training from scratch
w/o classification head:
```bash
python fusenet_train.py --dataroot ./datasets/nyu_class_10_db.h5 --batch_size 8 --lr 0.005
```

w/ classification head:
```bash
python fusenet_train.py --dataroot ./datasets/nyu_class_10_db.h5 --batch_size 8 --lr 0.005 \
                        --use_class True --name experiment_1
```

### Resuming training from a checkpoint
w/o classification head:
```bash
python fusenet_train.py --dataroot ./datasets/nyu_class_10_db.h5 --resume_train True --batch_size 8 \
                        --load_checkpoint ./checkpoints/experiment/nyu/best_model.pth.tar --lr 0.01
```

w classification head:
```bash
python fusenet_train.py --dataroot ./datasets/nyu_class_10_db.h5 --resume_train True --use_class True --batch_size 8 \
                        --load_checkpoint ./checkpoints/experiment/nyu/best_model_class_0_00040.pth.tar --lr 0.01 \
                        --lambda_class_range 0.004 0.01 5
```

Note: When training a model that contains the classification head, by default the lambda value, [which is the coefficient of the classification loss](/images/loss.PNG), is set to
0.001. In order to train the model for multiple sessions with multiple lambda values, following option should be added to the run command: ```--lambda_class_range start_value,
end_value, steps_between```. To train the model with only one session with one lambda value, set the start_value and the end_value the same, and the step_size to 1. 

## Inference
- To evaluate FuseNet results, run `fusenet_test.py`. Do not forget to include the 'class' word in the checkpoint file name when loading a model that contains 
the classification head.
- Model's semantic segmentation performance on the given dataset will be evaluated in three accuracy measures: global pixel-wise classification accuracy, 
intersection over union, and mean accuracy.
- Example run command:
```bash
python fusenet_test.py --dataroot ./datasets/nyu_class_10_db.h5 --load_checkpoint ./checkpoints/nyu/best_model.pth.tar
```

Note: To visualize the resulting images within the testing process, add `--vis_results True` option.  

- Pretrained FuseNet models will be uploaded soon.

## Plotting loss and accuracy history graphs
- To plot the loss and accuracy history of a model, use `fusenet_plots.ipynb` notebook.

## Result Visualization
- To visualize segmentation predictions separately, run `fusenet_visualize.py` script. Do not forget to include the 'class' word in the checkpoint file name when loading a model
that contains the classification head.
- Example run command:
```bash
python fusenet_visualize.py --dataroot ./datasets/nyu_class_10_db.h5 \
                            --load_checkpoint ./checkpoints/experiment/nyu/best_model_class_0_00010.pth.tar
```
Sample output images on NYU v2 (RGB - Ground Truth - Prediction):

<p float="left"><img src="images/sample_visuals/7.png" width="400px" style="margin:0px 45px"/>
<img src="images/sample_visuals/9.png" width="400px" style="margin:0px 45px"/></p>
<p float="left"><img src="images/sample_visuals/8.png" width="400px" style="margin:0px 45px"/>
<img src="images/sample_visuals/10.png" width="400px" style="margin:0px 45px"/></p>
<p float="left"><img src="images/sample_visuals/6.png" width="400px" style="margin:0px 45px"/>
<img src="images/sample_visuals/11.png" width="400px" style="margin:0px 45px"/></p>

## Citing FuseNet
Caner Hazirbas, Lingni Ma, Csaba Domokos and Daniel Cremers, _"FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture"_, in proceedings of the 13th Asian Conference on Computer Vision, 2016.

    @inproceedings{fusenet2016accv,
     author    = "C. Hazirbas and L. Ma and C. Domokos and D. Cremers",
     title     = "FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture",
     booktitle = "Asian Conference on Computer Vision",
     year      = "2016",
     month     = "November",
    }
