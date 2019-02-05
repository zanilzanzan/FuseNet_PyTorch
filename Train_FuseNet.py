import datetime
import torch
from torch.utils import data
from FuseNetClass import FuseNet
from SolverClass_FuseNet import SolverSemSeg
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
    else:
        seg_classes = 37

    train_data, test_data = get_data(dset_name=dset_name, use_train=True, use_test=True, use_class=use_class)
    print("[INFO] %s dataset retrieved" % dset_name)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    # Grid search for lambda values
    # Lambda is the coefficient of the classification loss
    # i.e.: total_loss = segmentation_loss + lambda * classification_loss
    lambdas = torch.linspace(0.04, 0.05, steps=10).cuda(gpu_device)
    for lam in lambdas:
        print('[INFO] Lambda value for the next training session: %.5f %s' % (lam, lam.dtype))
        start_date_time = datetime.datetime.now().replace(microsecond=0)

        model = FuseNet(seg_classes, gpu_device=gpu_device, use_class=use_class)

        solver = SolverSemSeg(optim_args={"lr": 5e-3, "weight_decay": 0.0005}, loss_func=cross_entropy_2d)
        # solver = Solver_SS(optim_args={"lr": 5e-3, "weight_decay": 0.0005}, loss_func=cross_entropy_2d)
        solver.train(gpu_device, model, dset_name, train_loader, test_loader, resume, num_epochs=1, log_nth=5, lam=lam)

        end_date_time = datetime.datetime.now().replace(microsecond=0)

        print('[INFO] Start and end time of the previous training session: %s - %s'
              % (start_date_time.strftime('%d.%m.%Y %H:%M:%S'), end_date_time.strftime('%d.%m.%Y %H:%M:%S')))
        print('[INFO] Total time the previous training session took:', (end_date_time - start_date_time), '\n')
