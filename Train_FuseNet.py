import datetime
import numpy as np
import torch
from FuseNetClass import FuseNet, CrossEntropy2d
from SolverClass_FuseNet import SolverSemSeg
from utils.data_utils_class import get_data


if __name__ == '__main__':
    gpu_device = 0
    torch.cuda.set_device(gpu_device)
    print('[INFO] Chosen GPU Device: ' + str(torch.cuda.current_device()))

    resume = False
    dset_name = 'NYU'

    if dset_name == 'NYU':
        seg_classes = 40
    else:
        seg_classes = 37

    train_data, test_data = get_data(dset_name=dset_name, use_train=True, use_test=True, use_class=True)
    print("[INFO] %s dataset retrieved" % dset_name)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    # Grid search for lambda values
    # Lambda is the coefficient of the classification loss
    # i.e.: total_loss = segmentation_loss + lambda * classification_loss
    lambdas = np.linspace(0.0004, 0.005, num=10)

    for lam in lambdas:
        print('[INFO] Lambda value for the next training session: %.5f' % lam)
        start_date_time  = datetime.datetime.now().replace(microsecond=0)

        model = FuseNet(seg_classes)

        solver = SolverSemSeg(optim_args={"lr":5e-3, "weight_decay": 0.0005}, loss_func=CrossEntropy2d)
        solver.train(gpu_device, model, lam, dset_name, train_loader, test_loader, resume, log_nth=5, num_epochs=1)

        end_date_time = datetime.datetime.now().replace(microsecond=0)

        print('[INFO] Start time of the previous training session: ' + start_date_time.strftime('%d.%m.%Y %H:%M:%S'))
        print('[INFO] End tim of the previous training session: ' + end_date_time.strftime('%d.%m.%Y %H:%M:%S'))
        print('[INFO] Total time the previous training session took: ', (end_date_time - start_date_time), '\n')
