import datetime
import torch
from torch.utils import data
from fusenet_solver_2 import Solver
from utils.data_utils import get_data
from utils.loss_utils import cross_entropy_2d

if __name__ == '__main__':
    gpu_device = 0
    torch.cuda.set_device(gpu_device)
    print('[INFO] Chosen GPU Device: ' + str(torch.cuda.current_device()))

    use_class = True
    resume = False
    dset_name = 'NYU'

    if dset_name == 'NYU':
        seg_classes = 40
    elif dset_name == 'SUN':
        seg_classes = 37
    else:
        raise NameError('Dataset name should be either NYU or SUN')

    train_data, test_data = get_data(dset_name=dset_name, use_train=True, use_test=True, use_class=use_class)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    print("[INFO] Data loaders for %s dataset have been created" % dset_name)

    # Grid search for lambda values
    # Lambda is the coefficient of the classification loss
    # i.e.: total_loss = segmentation_loss + lambda * classification_loss
    lambdas = torch.linspace(0.04, 0.05, steps=10).cuda(gpu_device)

    for lam in lambdas:

        start_date_time = datetime.datetime.now().replace(microsecond=0)
        solver = Solver(gpu_device, optim_args={"lr": 5e-3, "weight_decay": 0.0005}, loss_func=cross_entropy_2d, use_class=use_class)
        print('[INFO] Lambda value for this training session: %.5f' % lam)

        if use_class:
            solver.train_model(dset_name, train_loader, test_loader, resume, num_epochs=1, log_nth=5, lam=lam)
        else:
            solver.train_model(dset_name, train_loader, test_loader, resume, num_epochs=1, log_nth=5)
        end_date_time = datetime.datetime.now().replace(microsecond=0)

        print('[INFO] Start and end time of the previous training session: %s - %s'
              % (start_date_time.strftime('%d.%m.%Y %H:%M:%S'), end_date_time.strftime('%d.%m.%Y %H:%M:%S')))
        print('[INFO] Total time the previous training session took:', (end_date_time - start_date_time), '\n')