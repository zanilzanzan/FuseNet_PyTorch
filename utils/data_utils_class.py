import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
from scipy import io
from PIL import Image
import h5py
import torch
import torch.utils.data as data
from torchvision import transforms

class CreateData(data.Dataset):

    def __init__(self, rgb, depth, seg_label, class_label):
        self.rgb = rgb
        self.depth = depth
        self.seg_label = seg_label
        self.class_label = class_label
        
    def __getitem__(self, index):
        img = self.rgb[index]
        img_d = self.depth[index]
        seg_label = self.seg_label[index]
        class_label = self.class_label[index]
        
        img = torch.from_numpy(img)
        img_d = torch.from_numpy(img_d)
        return img, img_d, seg_label, class_label

    def __len__(self):
        return len(self.seg_label)

def get_data(dtype=np.float32, dset_type='NYU'):
    """
    Load NYU_v2 or SUN rgb-d dataset in hdf5 format from disk and prepare
    it for classifiers.
    """
    # Load the chosen data path
    if dset_type == 'SUN': 
        path = '/usr/stud/soenmeza/Desktop/FuseNet/data/h5_files/sunrgbd1_db.h5'
    elif dset_type == 'NYU':
        path = '/usr/stud/soenmeza/Desktop/FuseNet/data/h5_files/nyu_class_10_db.h5'
    else:
        print 'Wrong data requested. Please choose either "NYU" or "SUN".'
    
    h5file = h5py.File(path, 'r')

    # Create numpy arrays of training samples
    rgb_train = np.array(h5file['rgb_train'])
    rgb_train = rgb_train.astype(dtype)
    depth_train = np.array(h5file['depth_train'])
    depth_train = depth_train.astype(dtype)
    label_train = np.array(h5file['label_train'])
    label_train = label_train.astype(np.int64)
    label_class_train = np.array(h5file['class_train'])
    label_class_train = label_class_train.astype(np.int64)

    # Create numpy arrays of test samples
    rgb_test = np.array(h5file['rgb_test'])
    rgb_test = rgb_test.astype(dtype)
    
    depth_test = np.array(h5file['depth_test'])
    depth_test = depth_test.astype(dtype)
    
    label_test = np.array(h5file['label_test'])    
    label_test = label_test.astype(np.int64)
    
    label_class_test = np.array(h5file['class_test'])
    label_class_test = label_class_test.astype(np.int64)
    
    h5file.close()

    return (CreateData(rgb_train, depth_train, label_train, label_class_train), 
            CreateData(rgb_test, depth_test, label_test, label_class_test))
