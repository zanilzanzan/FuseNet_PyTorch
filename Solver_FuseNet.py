import numpy as np
import torch
from torch.autograd import Variable
from random import shuffle
import os
import shutil
import torch.backends.cudnn as cudnn
import torch.optim

from time import time


class Solver_SS(object):
    default_sgd_args = {"lr": 1e-3,
                        "momentum": 0.9,
                        "weight_decay": 0.0005}

    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.SGD, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_sgd_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()
        self._reset_histories()

    def save_checkpoint(self, state, is_best, dset_type='NYU'):
        if dset_type == 'NYU':
            filename = 'models/nyu/'
        elif dset_type == 'SUN':
            filename = 'models/sun/' 
        else: 
            print ("[ERROR] Please correct dset_type. You can choose either SUN or NYU.")
        cp_filename = filename + 'checkpoint25.pth.tar'
        torch.save(state, cp_filename)
        if is_best:
            shutil.copyfile(cp_filename, filename+'model_best19.pth.tar')
            print('[PROGRESS] Model successfully updated')
        print('[PROGRESS] Checkpoint saved')

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.best_val_acc = 0.0
        self.start_epoch = 0
        self.best_model = None
        self.running_loss = 0.0

    def update_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
        """
        lr = self.optim_args['lr'] *  (0.9 ** (epoch // 40))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self, gpu_device, model, dset_type, train_loader, val_loader, resume=False, num_epochs=10, log_nth=0, lam=None):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - dset_type: data set type, string: SUN or NYU  
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - resume: bool parameter, indicating training mode
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        optim = self.optim(model.parameters(), **self.optim_args)
        criterion = self.loss_func
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        val_iter_per_epoch = len(val_loader)
        print(val_iter_per_epoch)

        if resume:
            print("[PROGRESS] Selected Training Mode: RESUME")
            if dset_type == 'NYU':
                model_path = '/remwork/atcremers72/soenmeza/FuseNet/models/nyu/checkpoint12.pth.tar'
            elif dset_type == 'SUN':
                model_path = '/remwork/atcremers72/soenmeza/FuseNet/models/sun/checkpoint12.pth.tar'
            if os.path.isfile(model_path):
                print("[PROGRESS] Loading checkpoint: '{}'".format(model_path))
                checkpoint = torch.load(model_path)
                self.best_model = model
                self.start_epoch = checkpoint['epoch']
                self.best_val_acc = checkpoint['best_val_acc']
                model.load_state_dict(checkpoint['state_dict'])
                self.best_model.load_state_dict(checkpoint['best_state_dict'])
                self.train_loss_history = checkpoint['train_loss_hist']
                self.train_acc_history = checkpoint['train_acc_hist']
                self.val_acc_history = checkpoint['val_acc_hist']

                optim.load_state_dict(checkpoint['optimizer'])
                print("[PROGRESS] Checkpoint loaded")
                print("[PROGRESS] Resuming from epoc {}"
                      .format(checkpoint['epoch']))
                print("[PROGRESS] TRAINING CONTINUES")
            else:
                print("[ERROR] No checkpoint found at '{}'".format(model_path))
        else:
            print("[PROGRESS] Selected Training Mode: NEW")
            print("[PROGRESS] TRAINING STARTS")

        #print(self.train_loss_history)
        #print(self.train_acc_history)
        #print(self.val_acc_history)

        end_epoch = self.start_epoch + num_epochs
        for epoch in range(self.start_epoch, end_epoch):  # loop over the dataset multiple times
            timestep1 = time()
            self.update_learning_rate(optim, epoch)
            running_loss = 0.0

            model.train()
            for i, data in enumerate(train_loader, 0):   
                timestep2 = time()
                rgb_inputs  = Variable(data[0].cuda(gpu_device))
                d_inputs    = Variable(data[1].cuda(gpu_device))
                labels      = Variable(data[2].cuda(gpu_device))
                
                batch_size = len(rgb_inputs)
                first_it = (i == 0) and (epoch == 0)
                epoch_end = ((i + 1) % iter_per_epoch) == 0
      
                # zero the parameter gradients
                optim.zero_grad()
                
                # forward + backward + optimize
                outputs = model(rgb_inputs, d_inputs)
                loss = criterion(outputs, labels, use_class=False)
                loss.backward()
                optim.step()
                self.running_loss += loss.item()
                running_loss += loss.item()
                
                # print statistics
                if (i+1) % log_nth == 0 or (i+1) == iter_per_epoch:    # print every log_nth mini-batches
                    timestep3 = time()
                    running_loss = running_loss / log_nth
                    print("\r[EPOCH: %d/%d Iter: %d/%d ] Loss: %.3f Best Acc: %.3f LR: %.2e Time: %.2f seconds" 
                          % (epoch +1, end_epoch, i + 1, iter_per_epoch, running_loss, self.best_val_acc, optim.param_groups[0]['lr'] ,(timestep3-timestep2))),
                    
                # log and save the accuracies     
                if epoch_end:
                    train_scores = []
                    val_scores = []
                    
                    self.running_loss /= (i+1)
                    # print(self.running_loss)
                    # print(self.running_loss, i+1)
                    self.train_loss_history.append(self.running_loss)

                    _, train_preds = torch.max(outputs, 1)
                    
                    labels_mask = labels > 0
                    labels = labels - 1
                    train_scores.append(np.mean((train_preds == labels)[labels_mask].data.cpu().numpy()))


                    model.eval()
                    for batch in val_loader:
                        val_rgb_inputs  = Variable(batch[0].cuda(gpu_device))
                        val_d_inputs    = Variable(batch[1].cuda(gpu_device))
                        val_labels      = Variable(batch[2].cuda(gpu_device))
                        val_outputs     = model(val_rgb_inputs, val_d_inputs)
                        _, val_preds    = torch.max(val_outputs, 1)

                        val_labels_mask = val_labels > 0
                        val_labels = val_labels - 1
                        val_scores.append(np.mean((val_preds == val_labels)[val_labels_mask].data.cpu().numpy()))
                    
                    train_acc = np.mean(train_scores)
                    val_acc = np.mean(val_scores)
                                
                    self.train_acc_history.append(train_acc)
                    self.val_acc_history.append(val_acc)
                    
                    print("[EPOCH: %d/%d] TRAIN Acc/Loss: %.3f/%.3f VALIDATION Acc: %.3f " % (epoch + 1, end_epoch, train_acc, self.running_loss, val_acc))
                    self.running_loss = 0.0
                    # Save the checkpoint and update the model
                    is_best = val_acc > self.best_val_acc
                    
                    if is_best:
                        self.best_model = model
                        if is_best or (epoch+1) % 10 == 0:
                            self.best_val_acc = max(val_acc, self.best_val_acc)
                            self.save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_state_dict':self.best_model.state_dict(),
                                'best_val_acc': self.best_val_acc,
                                'train_loss_hist': self.train_loss_history,
                                'train_acc_hist': self.train_acc_history,
                                'val_acc_hist': self.val_acc_history,
                                'optimizer' : optim.state_dict()},
                                is_best, dset_type)
                timestep4 = time()
            #print('Epoch %i took %.2f seconds' %(epoch + 1,timestep4 - timestep1))

        # Calculate IoU and Mean accuracies
        num_classes = val_outputs.size(1)
        val_confusion = np.zeros((num_classes,3))
        IoU = 0
        mean_acc = 0

        for batch in val_loader:
            val_rgb_inputs  = Variable(batch[0].cuda(gpu_device))
            val_d_inputs    = Variable(batch[1].cuda(gpu_device))
            val_labels      = Variable(batch[2].cuda(gpu_device))
            val_outputs     = self.best_model(val_rgb_inputs, val_d_inputs)
            _, val_preds    = torch.max	(val_outputs, 1)

            val_labels = val_labels - 1

            for i in range(num_classes):
                val_labels_mask = val_labels == i
                val_preds_mask = val_preds == i
                TP = np.sum((val_preds == val_labels)[val_labels_mask].data.cpu().numpy())
                #print TP
                val_confusion[i,0] += TP 
                val_confusion[i,1] += np.sum((val_labels==val_labels)[val_labels_mask].data.cpu().numpy()) - TP 
                val_confusion[i,2] += np.sum((val_preds==val_preds)[val_preds_mask].data.cpu().numpy()) - TP 

        for i in range(num_classes):
            TP, FP, FN = val_confusion[i]
            print(TP+FP,FN)
            IoU += TP / (TP + FP + FN)
            mean_acc += TP / (TP + FP)
        IoU /= num_classes
        mean_acc /= num_classes

        print("[FINAL] TRAINING COMPLETED")
        print("        Best VALIDATION Accuracy: %.3f IoU: %.3f Mean Accuracy: %.3f" % (self.best_val_acc, IoU, mean_acc))
        print("        Orgnal. FuseNet Accuracy: 0.66  IoU: 0.327 Mean Accuracy: 0.434")
