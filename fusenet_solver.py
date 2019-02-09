import os
import datetime
from time import time
import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
from fusenet_model import FuseNet


class Solver(object):
    default_sgd_args = {"lr": 1e-3,
                        "momentum": 0.9,
                        "weight_decay": 0.0005}
    # default_adam_args = {"lr": 1e-4,
    #                      "betas": (0.9, 0.999),
    #                      "eps": 1e-8,
    #                      "weight_decay": 0.0}

    def __init__(self, opt, dset_info, loss_func=torch.nn.CrossEntropyLoss):

        self.opt = opt
        self.dset_name, self.seg_class_num = next(iter(dset_info.items()))
        self.gpu_device = opt.gpu_id
        print('[INFO] Chosen GPU Device: %s' % torch.cuda.current_device())

        # Set the optimizer
        optim_args = {"lr": opt.lr, "weight_decay": opt.weight_decay}
        optim_args_merged = self.default_sgd_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        if opt.optim.lower() == 'sgd':
            self.optim = torch.optim.SGD

        self.loss_func = loss_func()
        self.use_class = opt.use_class
        self.states = dict()

        # Create the FuseNet model
        self.model = FuseNet(self.seg_class_num, self.gpu_device, self.use_class)
        print(self.model)

    def reset_histories_and_losses(self):
        """
        Resets train and val histories for accuracy and the loss.
        """
        self.states['epoch'] = 0
        self.states['train_loss_hist'] = []
        self.states['train_seg_acc_hist'] = []

        if self.use_class:
            self.states['train_seg_loss_hist'] = []
            self.states['train_class_loss_hist'] = []
            self.states['train_class_acc_hist'] = []
            self.states['val_class_acc_hist'] = []

        self.states['val_seg_acc_hist'] = []
        self.states['best_val_seg_acc'] = 0.0

    def save_checkpoint(self, state, lam, is_best):
        """ Write docstring
        """
        print('[PROGRESS] Saving the model', end="", flush=True)
        checkpoint_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.dset_name.lower())
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        lam_text = ''
        if lam:
            lam_text = ('_class_' + '%.5f' % lam).replace('.', '_')
        now = datetime.datetime.now()

        # Save checkpoint with the name including epoch, - if exists, lambda value for classification - and date
        checkpoint_filename = os.path.join(checkpoint_dir, 'model_checkpoint' + lam_text + '_{}'.format(state['epoch'] + 1)
                                           + now.strftime('_%d%m%Y') + '.pth.tar')

        # If the model also the best performing model in the training session save it separately
        if is_best:
            best_model_filename = os.path.join(checkpoint_dir, 'best_model' + lam_text + '.pth.tar')
            self.states['best_model_name'] = best_model_filename
            torch.save(state, best_model_filename)
            # shutil.copyfile(checkpoint_filename, best_model_filename)
            print('\r[INFO] Best model has been successfully updated: %s' % best_model_filename)
            # shutil.copyfile(best_model_filename, checkpoint_filename)
            # print('[INFO] Checkpoint has been saved: %s' % checkpoint_filename)
            # return

        torch.save(state, checkpoint_filename)
        print('[INFO] Checkpoint has been saved: %s' % checkpoint_filename)

    def load_checkpoint(self, checkpoint_path, optim=None, only_model=False):
        """ Write docstring
        """
        if os.path.isfile(checkpoint_path):
            print('[PROGRESS] Loading checkpoint: {}'.format(checkpoint_path), end="", flush=True)

            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load the model state dictionary from the checkpoint
            self.model.load_state_dict(checkpoint['state_dict'])
            print('\r[INFO] Checkpoint has been loaded: {}'.format(checkpoint_path))

            if not only_model:
                # Load optimization method parameters from the checkpoint
                optim.load_state_dict(checkpoint['optimizer'])
                # Load the necessary checkpoint key values to the states dictionary which contains loss and history values/lists
                self.states.update({key: value for key, value in checkpoint.items() if key not in ['optimizer', 'state_dict']})

                print('[INFO] History lists have been loaded')
                print('[INFO] Resuming from epoch {}'.format(checkpoint['epoch']+1))

                # content of checkpoint is loaded to the instance; so, delete checkpoint variable to create space on the GPU
                del checkpoint
                torch.cuda.empty_cache()

                return optim
        else:
            raise FileNotFoundError('Checkpoint file not found: %s' % checkpoint_path)

    def update_learning_rate(self, optim, epoch):
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
        """
        lr = self.optim_args['lr'] * (0.9 ** (epoch // 40))
        for param_group in optim.param_groups:
            param_group['lr'] = lr

    def update_model_state(self, optim):
        """
        :return: dictionary of model parameters to be saved
        """
        return_dict = self.states
        return_dict.update({'state_dict': self.model.state_dict(), 'optimizer': optim.state_dict()})
        return return_dict

    def validate_model(self, val_loader):
        """ Write docstring
        :param val_loader:
        :return:
        """
        print('\n[PROGRESS] Validating the model',  end="", flush=True)
        # Evaluate model in eval mode
        self.model.eval()
        val_seg_scores = []
        val_class_scores = []

        for batch in val_loader:
            val_rgb_inputs = Variable(batch[0].cuda(self.gpu_device))
            val_d_inputs = Variable(batch[1].cuda(self.gpu_device))
            val_labels = Variable(batch[2].cuda(self.gpu_device))

            if self.use_class:
                val_class_labels = Variable(batch[3].cuda(self.gpu_device))
                # Infer segmentation and classification results
                val_seg_outputs, val_class_outputs = self.model(val_rgb_inputs, val_d_inputs)
                # val_class_preds.data.cpu().numpy()[0]
                _, val_preds_class = torch.max(val_class_outputs, 1)
                val_preds_class += 1
                val_class_scores.append(np.mean(val_preds_class.data.cpu().numpy() == val_class_labels.data.cpu().numpy()))
            else:
                # Infer only segmentation results
                val_seg_outputs = self.model(val_rgb_inputs, val_d_inputs)

            _, val_preds_seg = torch.max(val_seg_outputs, 1)
            val_labels_mask = val_labels > 0
            val_labels = val_labels - 1
            val_seg_scores.append(np.mean((val_preds_seg == val_labels)[val_labels_mask].data.cpu().numpy()))
            del val_preds_seg, val_seg_outputs, val_labels_mask

        self.states['val_seg_acc_hist'].append(np.mean(val_seg_scores))
        if self.use_class:
            self.states['val_class_acc_hist'].append(np.mean(val_class_scores))
        print('\r[INFO] Validation has been completed')
        print('VAL SEG HISTORY: ', self.states['val_seg_acc_hist'])

    def train_model(self, train_loader, val_loader, num_epochs=10, log_nth=0, lam=None):
        """
        Train a given model with the provided data.

        Parameters
        ----------
        train_loader:
            train data in torch.utils.data.DataLoader
        val_loader:
            val data in torch.utils.data.DataLoader
        num_epochs: int - default: 10
            total number of training epochs
        log_nth: int - default: 0
            log training accuracy and loss every nth iteration
        lam: torch.float32
            lambda value used as weighting coefficient for classification loss
        """
        # Initiate/reset history lists and running-loss parameters
        self.reset_histories_and_losses()

        # Based on dataset sizes determine how many iterations per epoch will be done
        iter_per_epoch = len(train_loader)

        # Initiate optimization method and loss function
        optim = self.optim(self.model.parameters(), **self.optim_args)

        criterion = self.loss_func
        # Load pre-trained model parameters if resume option is chosen
        if self.opt.resume_train:
            print('[INFO] Selected training mode: RESUME')
            optim = self.load_checkpoint(self.opt.load_checkpoint, optim)
            print('[INFO] TRAINING CONTINUES')
        else:
            print('[INFO] Selected training mode: NEW')
            print('[INFO] TRAINING STARTS')

        # Determine at which epoch training session must end
        start_epoch = self.states['epoch']
        end_epoch = start_epoch + num_epochs
    
        # Start Training
        for epoch in range(start_epoch, end_epoch):
            # timestep1 = time()

            running_loss = 0.0
            running_class_loss = 0.0
            running_seg_loss = 0.0
            train_seg_scores = []
            train_class_scores = []

            self.update_learning_rate(optim, epoch)

            # Train model in training mode
            self.model.train()
            for i, data in enumerate(train_loader):
                time_stamp_2 = time()

                # Zero parameter gradients
                optim.zero_grad()

                # Retrieve batch-size of input images and labels from training dataset loader
                rgb_inputs = Variable(data[0].cuda(self.gpu_device))
                d_inputs = Variable(data[1].cuda(self.gpu_device))
                train_seg_labels = Variable(data[2].cuda(self.gpu_device))

                if self.use_class:
                    class_labels = Variable(data[3].cuda(self.gpu_device))
                    # forward + backward + optimize with segmentation and class loss
                    output_seg, output_class = self.model(rgb_inputs, d_inputs)
                    loss, seg_loss, class_loss = criterion(output_seg, train_seg_labels, output_class, class_labels, lambda_2=lam)
                else:
                    # forward + backward + optimize only with segmentation loss
                    output_seg = self.model(rgb_inputs, d_inputs)
                    # loss = seg_loss = criterion(output_seg, train_seg_labels)
                    loss = criterion(output_seg, train_seg_labels)

                loss.backward()
                optim.step()

                # Update running losses
                running_loss += loss.item()
                del loss

                if self.use_class:
                    running_seg_loss += seg_loss
                    running_class_loss += class_loss

                    _, train_class_preds = torch.max(output_class, 1)

                    train_class_preds += 1
                    train_class_scores.append(np.mean((train_class_preds == class_labels).data.cpu().numpy()))
                    del class_labels, train_class_preds

                _, train_seg_preds = torch.max(output_seg, 1)

                labels_mask = train_seg_labels > 0
                train_seg_labels = train_seg_labels - 1

                train_seg_scores.append(np.mean((train_seg_preds == train_seg_labels)[labels_mask].data.cpu().numpy()))
                del train_seg_preds, train_seg_labels, labels_mask

                # Print statistics
                # Print each log_nth mini-batches or at the end of the epoch
                if (i+1) % log_nth == 0 or (i+1) == iter_per_epoch:
                    time_stamp_3 = time()
                    running_loss /= log_nth

                    if self.use_class:
                        running_seg_loss /= log_nth
                        running_class_loss /= log_nth
                        print("\r[EPOCH: %d/%d Iter: %d/%d ] Total_Loss: %.3f Seg_Loss: %.3f "
                              "Class_Loss: %.3f Best_Acc: %.3f LR: %.2e Lam: %.5f Time: %.2f seconds         "
                              % (epoch + 1, end_epoch, i + 1, iter_per_epoch, running_loss, running_seg_loss,
                                 running_class_loss, self.states['best_val_seg_acc'], optim.param_groups[0]['lr'], lam,
                                 (time_stamp_3-time_stamp_2)), end='\r')
                    else:
                        print("\r[EPOCH: %d/%d Iter: %d/%d ] Seg_Loss: %.3f Best_Acc: %.3f LR: %.2e Time: %.2f seconds       "
                              % (epoch + 1, end_epoch, i + 1, iter_per_epoch, running_loss, self.states['best_val_seg_acc'],
                                 optim.param_groups[0]['lr'], (time_stamp_3-time_stamp_2)), end='\r')

            # Log and save accuracy and loss values
            # Average accumulated loss values over the whole dataset
            running_loss /= iter_per_epoch
            self.states['train_loss_hist'].append(running_loss)

            if self.use_class:
                running_seg_loss /= iter_per_epoch
                self.states['train_seg_loss_hist'].append(running_seg_loss)
                running_class_loss /= iter_per_epoch
                self.states['train_class_loss_hist'].append(running_class_loss)
                # print('Train Class Scores shape and itself: ', len(train_class_scores), train_class_scores)
            # print('Train Seg Scores shape and itself: ', len(train_seg_scores), train_seg_scores)

            train_seg_acc = np.mean(train_seg_scores)
            self.states['train_seg_acc_hist'].append(train_seg_acc)

            # Run the model on the validation set and gather segmentation and classification accuracy
            self.validate_model(val_loader)

            if self.use_class:
                train_class_acc = np.mean(train_class_scores)
                self.states['train_class_acc_hist'].append(train_class_acc)

                print("[EPOCH: %d/%d] TRAIN Seg_Acc/Class_Acc/Loss/Seg_Loss/Class_Loss: %.3f/%.3f/%.3f/%.3f/%.3f "
                      "VALIDATION Seg_Acc/Class_Acc: %.3f %.3f"
                      % (epoch + 1, end_epoch, train_seg_acc, train_class_acc, running_loss, running_seg_loss,
                         running_class_loss, self.states['val_seg_acc_hist'][-1], self.states['val_class_acc_hist'][-1]))
            else:
                print("[EPOCH: %d/%d] TRAIN Seg_Acc/Seg_Loss: %.3f/%.3f VALIDATION Seg_Acc: %.3f"
                      % (epoch + 1, end_epoch, train_seg_acc, running_loss, self.states['val_seg_acc_hist'][-1]))

            # Save the checkpoint and update the model
            if (epoch+1) > self.opt.save_epoch_freq:
                current_val_seg_acc = self.states['val_seg_acc_hist'][-1]
                best_val_seg_acc = self.states['best_val_seg_acc']
                is_best = current_val_seg_acc > best_val_seg_acc
                self.states['epoch'] = epoch

                if is_best or (epoch+1) % 10 == 0:
                    self.states['best_val_seg_acc'] = max(current_val_seg_acc, best_val_seg_acc)

                    # model_state = self.update_model_state(epoch, self.model)
                    self.save_checkpoint(self.update_model_state(optim), lam, is_best)

        print("[FINAL] TRAINING COMPLETED")
        self.seg_acc_calculation(val_loader)

    def seg_acc_calculation(self, val_loader):
        print("[INFO] Calculating Global, IoU, and Mean accuracy. This may take up to a minute. Because the script is a little crappy."
              " I admit that. If you think you can optimize it, please be my guest and send a merge request.")
        self.load_checkpoint(self.states['best_model_name'], only_model=True)

        # Calculate IoU and Mean accuracy for semantic segmentation
        num_classes = self.seg_class_num
        val_confusion_mtx = np.zeros((num_classes, 3))
        iou = 0
        mean_acc = 0

        for batch in val_loader:
            val_rgb_inputs = Variable(batch[0].cuda(self.gpu_device))
            val_d_inputs = Variable(batch[1].cuda(self.gpu_device))
            val_labels = Variable(batch[2].cuda(self.gpu_device))

            if self.use_class:
                val_outputs, _ = self.model(val_rgb_inputs, val_d_inputs)
            else:
                val_outputs = self.model(val_rgb_inputs, val_d_inputs)

            _, val_preds = torch.max(val_outputs, 1)

            val_labels = val_labels - 1

            for idx in range(num_classes):
                val_labels_mask = val_labels == idx
                val_preds_mask = val_preds == idx
                tp = np.sum((val_preds == val_labels)[val_labels_mask].data.cpu().numpy())

                val_confusion_mtx[idx, 0] += tp
                val_confusion_mtx[idx, 1] += np.sum((val_labels == val_labels)[val_labels_mask].data.cpu().numpy()) - tp
                val_confusion_mtx[idx, 2] += np.sum((val_preds == val_preds)[val_preds_mask].data.cpu().numpy()) - tp

        for i in range(num_classes):
            tp, fp, fn = val_confusion_mtx[i]
            print(tp + fp, fn)
            iou += tp / (tp + fp + fn)
            mean_acc += tp / (tp + fp)
        iou /= num_classes
        mean_acc /= num_classes

        print("[INFO] Best VALIDATION (NYU-v2) Segmentation Accuracy: %.3f IoU: %.3f Mean Accuracy: %.3f"
              % (self.states['best_val_seg_acc'], iou, mean_acc))
        print("[INFO] Orgnal. FuseNet (NYU-v2) Segmentation Accuracy: 0.660 IoU: 0.327 Mean Accuracy: 0.434")
