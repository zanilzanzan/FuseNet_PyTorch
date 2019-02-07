import numpy as np
import h5py
import torch
import torch.utils.data as data


class CreateData(data.Dataset):

    def __init__(self, rgb, depth, label):
        self.rgb = rgb
        self.depth = depth
        self.label = label

    def __getitem__(self, index):
        img = self.rgb[index]
        img_d = self.depth[index]
        label = self.label[index]

        img = torch.from_numpy(img)
        img_d = torch.from_numpy(img_d)
        return img, img_d, label

    def __len__(self):
        return len(self.label)

def get_data(dtype=np.float32, dset_type='NYU'):
    """
    Load NYU_v2 or SUN rgb-d dataset in hdf5 format from disk and prepare
    it for classifiers.
    """
    # Load the chosen data path
    if dset_type == 'SUN': 
        path = '/usr/stud/soenmeza/Desktop/FuseNet/data/h5_files/sunrgbd1_db.h5'
    elif dset_type == 'NYU':
        path = '/usr/stud/soenmeza/Desktop/FuseNet/data/h5_files/nyu_class_db.h5'
    else:
        raise Exception('Wrong data requested. Please choose either "NYU" or "SUN".')
    
    h5file = h5py.File(path, 'r')

    # Create numpy arrays of training samples
    rgb_train = np.array(h5file['rgb_train'])
    rgb_train = rgb_train.astype(dtype)
    depth_train = np.array(h5file['depth_train'])
    depth_train = depth_train.astype(dtype)
    label_train = np.array(h5file['label_train'])
    label_train = label_train.astype(np.int64)

    # Create numpy arrays of test samples
    rgb_test = np.array(h5file['rgb_test'])
    rgb_test = rgb_test.astype(dtype)
    depth_test = np.array(h5file['depth_test'])
    depth_test = depth_test.astype(dtype)
    label_test = np.array(h5file['label_test'])    
    label_test = label_test.astype(np.int64)

    h5file.close()

    return (CreateData(rgb_train, depth_train, label_train), 
            CreateData(rgb_test, depth_test, label_test))
