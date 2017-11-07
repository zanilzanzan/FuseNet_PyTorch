import numpy as np
import torch
from torch.autograd import Variable
from FuseNetClass import FuseNet, CrossEntropy2d
from SolverClass_FuseNet import Solver_SS
import torch.nn.functional as F
from utils.data_utils_class import get_data

import time

import h5py
import scipy.io 
import os

############################################################################
#                                TRAIN                                     #
############################################################################

resume = False
dset_type = 'NYU'
train_data, test_data = get_data(dset_type=dset_type)
print ("[PROGRESS] %s dataset retrieved"  %(dset_type))
start_time = time.asctime(time.localtime(time.time()))
print(start_time)

train_loader    = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=1)
test_loader     = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

# Grid search for lambda values
lambdas = np.linspace(0.0004, 0.005, num=10)

for lam in lambdas:
    print(lam)
    if dset_type == 'NYU':
        model = FuseNet(40)
    else:
        model = FuseNet(37)

    solver = Solver_SS(optim_args={"lr":5e-3, "weight_decay": 0.0005}, loss_func=CrossEntropy2d)
    solver.train(model, lam, dset_type, train_loader, test_loader, resume, log_nth=5, num_epochs=300)

end_time = time.asctime(time.localtime(time.time()))
print(start_time, end_time)
