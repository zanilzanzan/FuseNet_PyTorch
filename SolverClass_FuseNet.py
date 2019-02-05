import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
import os
import shutil
from time import time


class SolverSemSeg(object):
    default_sgd_args = {"lr": 1e-3,
                        "momentum": 0.9,
                        "weight_decay": 0.0005}

    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.SGD, optim_args={}, loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_sgd_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()
        self._reset_histories()

    def save_checkpoint(self, state, lam, is_best, dset_type='NYU'):
        if dset_type == 'NYU':
            filename = './models/nyu/'
        elif dset_type == 'SUN':
            filename = 'models/sun/' 
        else: 
            print ("[ERROR] Please correct dset_type. You can choose either SUN or NYU.")
        cp_filename = filename + str(lam) + '_checkpoint_class_1.pth.tar'
        torch.save(state, cp_filename)
        if is_best:
            shutil.copyfile(cp_filename, filename + str(lam) + '_model_best_class_2.pth.tar')
            print('[PROGRESS] Model successfully updated')
        print('[PROGRESS] Checkpoint saved')

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_seg_loss_history = []
        self.train_class_loss_history = []
        self.train_seg_acc_history = []
        self.train_class_acc_history = []
        self.val_seg_acc_history = []
        self.val_class_acc_history = []
        self.best_val_acc = 0.0
        self.start_epoch = 0
        self.best_model = None
        self.running_loss = 0.0
        self.running_seg_loss = 0.0
        self.running_class_loss = 0.0

    def update_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
        """
        lr = self.optim_args['lr'] *  (0.9 ** (epoch // 40))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, gpu_device, model, dset_type, train_loader, val_loader, resume=False, num_epochs=10, log_nth=0, lam=1.0):
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
                model_path = '/usr/stud/soenmeza/Desktop/FuseNet/models/nyu/checkpoint_class_26.pth.tar'
                #model_path = '/remwork/atcremers72/soenmeza/FuseNet/models/nyu/checkpoint24.pth.tar'
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
                self.train_seg_acc_history = checkpoint['train_seg_acc_hist']
                self.train_seg_class_history = checkpoint['train_class_acc_hist']
                self.val_seg_acc_history = checkpoint['val_seg_acc_hist']
                self.val_class_acc_history = checkpoint['val_class_acc_hist']
                
                optim.load_state_dict(checkpoint['optimizer'])

                print("[PROGRESS] Checkpoint loaded")
                print("[PROGRESS] Resuming from epoc {}".format(checkpoint['epoch']))
                print("[PROGRESS] TRAINING CONTINUES")
            else:
                print("[ERROR] No checkpoint found at '{}'".format(model_path))
        else:
            print("[PROGRESS] Selected Training Mode: NEW")
            print("[PROGRESS] TRAINING STARTS")

        # print(self.train_loss_history)
        # print(self.train_seg_acc_history)
        # print(self.train_class_acc_history)
        # print(self.val_seg_acc_history)
        # print(self.val_class_acc_history)
        
        end_epoch = self.start_epoch + num_epochs
    
        # Start Training
        for epoch in range(self.start_epoch, end_epoch):  # loop over the dataset multiple times
            timestep1 = time()
            self.update_learning_rate(optim, epoch)
            running_loss = 0.0
            running_class_loss = 0.0
            running_seg_loss = 0.0
            
            # Train model in train() mode
            model.train()
            for i, data in enumerate(train_loader, 0):   
                timestep2 = time()
                rgb_inputs = Variable(data[0].cuda(gpu_device))
                d_inputs = Variable(data[1].cuda(gpu_device))
                labels = Variable(data[2].cuda(gpu_device))
                class_labels = Variable(data[3].cuda(gpu_device))

                # print('[SOLVER DATA INFO] ', rgb_inputs, d_inputs, labels, class_labels)
                
                batch_size = len(rgb_inputs)
                first_it = (i == 0) and (epoch == 0)
                epoch_end = ((i + 1) % iter_per_epoch) == 0
      
                # zero the parameter gradients
                optim.zero_grad()
                
                # forward + backward + optimize
                output_seg, output_class = model(rgb_inputs, d_inputs)
                loss, seg_loss, class_loss = criterion(output_seg, labels, output_class, class_labels, lambda_2=lam, use_class=True)
                loss.backward()
                optim.step()
                
                # self.running_seg_loss += seg_loss.data[0]
                # self.running_class_loss += class_loss.data[0]
                # self.running_loss += loss.data[0]
                # running_loss += loss.data[0]
                # running_seg_loss += seg_loss.data[0]
                # running_class_loss += class_loss.data[0]

                self.running_seg_loss += seg_loss.item()
                self.running_class_loss += class_loss.item()
                self.running_loss += loss.item()
                running_loss += loss.item()
                running_seg_loss += seg_loss.item()
                running_class_loss += class_loss.item()
                
                # print statistics
                if (i+1) % log_nth == 0 or (i+1) == iter_per_epoch:    # print every log_nth mini-batches
                    timestep3 = time()
                    running_loss /= log_nth
                    running_seg_loss /= log_nth
                    running_class_loss /= log_nth
                    print("\r[EPOCH: %d/%d Iter: %d/%d ] Loss: %.3f S_Loss: %.3f C_Loss: %.3f Best Acc: %.3f LR: %.2e Lam: %.5f Time: %.2f seconds" 
                          % (epoch +1, end_epoch, i + 1, iter_per_epoch, running_loss, running_seg_loss, running_class_loss, self.best_val_acc, optim.param_groups[0]['lr'], lam, (timestep3-timestep2))),
                    
                # log and save the accuracies     
                if epoch_end:
                    train_seg_scores = []
                    train_class_scores = []
                    val_seg_scores = []
                    val_class_scores = []
                    
                    self.running_loss /= (i+1)
                    self.running_seg_loss /= (i+1)
                    self.running_class_loss /= (i+1)
                    
                    # print(self.running_loss)
                    # print(self.running_loss, i+1)
                    
                    self.train_loss_history.append(self.running_loss)
                    self.train_seg_loss_history.append(self.running_seg_loss)
                    self.train_class_loss_history.append(self.running_class_loss)
                    
                    _, train_seg_preds = torch.max(output_seg, 1)
                    _, train_class_preds = torch.max(output_class, 1)    
                    
                    labels_mask = labels > 0
                    labels = labels - 1
                    train_seg_scores.append(np.mean((train_seg_preds == labels)[labels_mask].data.cpu().numpy()))

                    train_class_preds += 1
                    train_class_scores.append(np.mean((train_class_preds == class_labels).data.cpu().numpy()))
                
                    # Evaluate model in .eval() mode
                    model.eval()
                    for batch in val_loader:
                        val_rgb_inputs = Variable(batch[0].cuda(gpu_device))
                        val_d_inputs = Variable(batch[1].cuda(gpu_device))
                        val_labels = Variable(batch[2].cuda(gpu_device))
                        val_class = Variable(batch[3].cuda(gpu_device))
                        val_outputs_seg, val_outputs_class = model(val_rgb_inputs, val_d_inputs)
                        
                        
                        (val_outputs_class.data.cpu().numpy()[0])
                        _, val_preds_seg    = torch.max(val_outputs_seg, 1)
                        _, val_preds_class  = torch.max(val_outputs_class, 1)
                        val_preds_class = torch.max(val_preds_class)
                        val_preds_class = val_preds_class.view(-1)
                        
                        val_labels_mask = val_labels > 0
                        val_labels = val_labels - 1
                        val_seg_scores.append(np.mean((val_preds_seg == val_labels)[val_labels_mask].data.cpu().numpy()))
                        
                        val_preds_class += 1
                        #print(val_preds_class.data.cpu().numpy(), val_class.data.cpu().numpy())
                        val_class_scores.append(np.mean(val_preds_class.data.cpu().numpy() == val_class.data.cpu().numpy()))
                        #print(val_preds_class, val_preds_class.size())
                    #print('val class scores: ', val_class_scores)

                    train_seg_acc = np.mean(train_seg_scores)
                    train_class_acc = np.mean(train_class_scores)
                    val_seg_acc = np.mean(val_seg_scores)
                    val_class_acc = np.mean(val_class_scores)        
                        
                    self.train_seg_acc_history.append(train_seg_acc)
                    self.train_class_acc_history.append(train_class_acc)
                    self.val_seg_acc_history.append(val_seg_acc)
                    self.val_class_acc_history.append(val_class_acc)
                    
                    print("[EPOCH: %d/%d] TRAIN SegAcc/ClassAcc/Loss/S_Loss/C_Loss: %.3f/%.3f/%.3f/%.3f/%.3f VALIDATION Seg Acc: %.3f Class Acc: %.3f" % (epoch + 1, end_epoch, train_seg_acc, train_class_acc, self.running_loss, self.running_seg_loss, self.running_class_loss, val_seg_acc, val_class_acc))
                    self.running_loss = 0.0
                
                    # Save the checkpoint and update the model
                    if (epoch+1) > 0:
                        is_best = val_seg_acc > self.best_val_acc
                        if is_best:
                            self.best_model = model
                        if is_best or (epoch+1) % 10 == 0:
                            self.best_val_acc = max(val_seg_acc, self.best_val_acc)
                            self.save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_state_dict':self.best_model.state_dict(),
                                'best_val_acc': self.best_val_acc,
                                'train_loss_hist': self.train_loss_history,
                                'train_seg_loss_hist': self.train_seg_loss_history,
                                'train_class_loss_hist': self.train_class_loss_history,
                                'train_seg_acc_hist': self.train_seg_acc_history,
                                'train_class_acc_hist': self.train_class_acc_history,
                                'val_seg_acc_hist': self.val_seg_acc_history,
                                'val_class_acc_hist': self.val_class_acc_history,
                                'optimizer' : optim.state_dict()},
                                lam, is_best, dset_type)

                timestep4 = time()
                #print('Epoch %i took %.2f seconds' %(epoch + 1,timestep4 - timestep1))

        # Calculate IoU and Mean accuracies
        num_classes = val_outputs_seg.size(1)
        print(num_classes)
        val_confusion = np.zeros((num_classes,3))
        IoU = 0
        mean_acc = 0

        for batch in val_loader:
            val_rgb_inputs = Variable(batch[0].cuda(gpu_device))
            val_d_inputs = Variable(batch[1].cuda(gpu_device))
            val_labels = Variable(batch[2].cuda(gpu_device))
            val_class_labels = Variable(batch[3].cuda(gpu_device))
            val_outputs, val_class_outputs = self.best_model(val_rgb_inputs, val_d_inputs)
            _, val_preds = torch.max(val_outputs, 1)
                   
            val_labels = val_labels - 1

            for i in range(num_classes):
                val_labels_mask = val_labels == i
                val_preds_mask = val_preds == i
                TP = np.sum((val_preds == val_labels)[val_labels_mask].data.cpu().numpy())

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
