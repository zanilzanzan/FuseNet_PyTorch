import os
import shutil
import datetime
from time import time, sleep
import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
from fusenet_model import FuseNet


class Solver(object):
    default_sgd_args = {"lr": 1e-3,
                        "momentum": 0.9,
                        "weight_decay": 0.0005}

    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.SGD, optim_args={}, loss_func=torch.nn.CrossEntropyLoss, use_class=False):
        optim_args_merged = self.default_sgd_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim

        self.loss_func = loss_func()
        self.use_class = use_class
        self.states = dict()

        self.model = FuseNet(40, 0, self.use_class)
        # print('AFTER MODEL CREATION.')
        # sleep(5)

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

    def save_checkpoint(self, state, lam, is_best, dset_name='NYU'):
        """ Write docstring
        """
        if dset_name == 'NYU':
            checkpoint_dir = './checkpoints/nyu/'  # take the output dir from user!
        else:
            checkpoint_dir = './checkpoints/sun/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        lam_text = ''
        if lam:
            lam_text = ('_' + '%.5f' % lam).replace('.', '_')
        now = datetime.datetime.now()

        # Save checkpoint with the name including epoch, - if exists, lambda value for classification - and date
        checkpoint_filename = os.path.join(checkpoint_dir, 'model_checkpoint' + lam_text + '_{}'.format(state['epoch'] + 1)
                                           + now.strftime('_%d%m%Y') + '.pth.tar')

        # state = {**self.states, **{'state_dict': self.model.state_dict(), 'optimizer': optim.state_dict()}}
        torch.save(state, checkpoint_filename)

        # If the model also the best performing model in the training session save it separately
        if is_best:
            best_model_filename = os.path.join(checkpoint_dir, 'best_model' + lam_text + '.pth.tar')
            shutil.copyfile(checkpoint_filename, best_model_filename)
            print('[INFO] Best model has been successfully updated: %s' % best_model_filename)
        print('[INFO] Checkpoint has been saved: %s' % checkpoint_filename )

    def load_checkpoint(self, optim, checkpoint_path='./checkpoints/nyu/model.pth.tar'):
        """ Write docstring
        """
        if os.path.isfile(checkpoint_path):
            print('[PROGRESS] Loading checkpoint: {}'.format(checkpoint_path), end="", flush=True)

            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load optimization method parameters and the model state dictionary from the checkpoint
            optim.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['state_dict'])

            # Load the necessary checkpoint key values to the states dictionary which contains loss and history values/lists
            self.states.update({key: value for key, value in checkpoint.items() if key not in ['optimizer', 'state_dict']})

            print('\r[INFO] Checkpoint has been loaded: {}'.format(checkpoint_path))
            print('[INFO] History lists have been loaded')
            print('[INFO] Resuming from epoch {}'.format(checkpoint['epoch']))
            print('[INFO] TRAINING CONTINUES')
            print('WAITING AFTER CHECKPOINT IS LOADED')

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

    def validate_model(self, gpu_device, val_loader, lam):
        """ Write docstring
        :param gpu_device:
        :param model:
        :param val_loader:
        :param lam:
        :return:
        """
        # Evaluate model in eval mode
        self.model.eval()
        print('EVAL MODE!')

        val_seg_scores = []
        val_class_scores = []

        for batch in val_loader:
            val_rgb_inputs = Variable(batch[0].cuda(gpu_device))
            val_d_inputs = Variable(batch[1].cuda(gpu_device))
            val_labels = Variable(batch[2].cuda(gpu_device))

            if self.use_class:
                val_class_labels = Variable(batch[3].cuda(gpu_device))
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

        print('VAL SEG SCORES: ', val_seg_scores)
        self.states['val_seg_acc_hist'].append(np.mean(val_seg_scores))
        if self.use_class:
            self.states['val_class_acc_hist'].append(np.mean(val_class_scores))
        print('VAL SEG HISTORY: ', self.states['val_seg_acc_hist'])

    def train_model(self, gpu_device, dset_name, train_loader, val_loader, resume=False, num_epochs=10, log_nth=0, lam=None):
        """
        Train a given model with the provided data.

        Parameters
        ----------
        gpu_device: int
            ID of the selected GPU device. type: int
        model:
            model object initialized from a torch.nn.Module
        dset_name: str
            data set type, string: SUN or NYU
        train_loader:
            train data in torch.utils.data.DataLoader
        val_loader:
            val data in torch.utils.data.DataLoader
        resume: bool - default: False
            parameter that indicates training mode
        num_epochs: int - default: 10
            total number of training epochs
        log_nth: int - default: 0
            log training accuracy and loss every nth iteration
        lam: torch.float32
            lambda value used as weighting coefficient for classification loss
        """
        loss = 0.0
        # Initiate/reset history lists and running-loss parameters
        self.reset_histories_and_losses()

        # Based on dataset sizes determine how many iterations per epoch will be done
        iter_per_epoch = len(train_loader)

        # Initiate optimization method and loss function
        optim = self.optim(self.model.parameters(), **self.optim_args)

        criterion = self.loss_func
        # Load pre-trained model parameters if resume option is chosen
        if resume:
            print('[INFO] Selected training mode: RESUME')
            optim = self.load_checkpoint(optim)

        else:
            print('[PROGRESS] Selected Training Mode: NEW')
            print('[PROGRESS] TRAINING STARTS')

        # Determine at which epoch training session must end
        star_epoch = self.states['epoch']
        end_epoch = star_epoch + num_epochs
    
        # Start Training
        for epoch in range(star_epoch, end_epoch):
            # timestep1 = time()

            running_loss = 0.0
            running_class_loss = 0.0
            running_seg_loss = 0.0
            train_seg_scores = []
            train_class_scores = []

            self.update_learning_rate(optim, epoch)

            # Train model in training mode
            self.model.train()
            print('TRAIN MODE!')
            for i, data in enumerate(train_loader):
                time_stamp_2 = time()

                # Zero parameter gradients
                optim.zero_grad()

                # Retrieve batch-size of input images and labels from training dataset loader
                rgb_inputs = Variable(data[0].cuda(gpu_device))
                d_inputs = Variable(data[1].cuda(gpu_device))
                train_seg_labels = Variable(data[2].cuda(gpu_device))

                if self.use_class:
                    class_labels = Variable(data[3].cuda(gpu_device))
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
                        print("\r[EPOCH: %d/%d Iter: %d/%d ] Loss: %.3f S_Loss: %.3f "
                              "C_Loss: %.3f Best Acc: %.3f LR: %.2e Lam: %.5f Time: %.2f seconds"
                              % (epoch +1, end_epoch, i + 1, iter_per_epoch, running_loss, running_seg_loss,
                                 running_class_loss, self.states['best_val_seg_acc'], optim.param_groups[0]['lr'], lam,
                                 (time_stamp_3-time_stamp_2)))
                    else:
                        print("\r[EPOCH: %d/%d Iter: %d/%d ] S_Loss: %.3f Best Acc: %.3f LR: %.2e Time: %.2f seconds"
                              % (epoch +1, end_epoch, i + 1, iter_per_epoch, running_loss, self.states['best_val_seg_acc'],
                                 optim.param_groups[0]['lr'], (time_stamp_3-time_stamp_2)))

            # Log and save accuracy and loss values
            # Average accumulated loss values over the whole dataset
            running_loss /= iter_per_epoch
            self.states['train_loss_hist'].append(running_loss)

            if self.use_class:
                running_seg_loss /= iter_per_epoch
                self.states['train_seg_loss_hist'].append(running_seg_loss)
                running_class_loss /= iter_per_epoch
                self.states['train_class_loss_hist'].append(running_class_loss)
                print('Train Class Scores shape and itself: ', len(train_class_scores), train_class_scores)

            print('Train Seg Scores shape and itself: ', len(train_seg_scores), train_seg_scores)
            train_seg_acc = np.mean(train_seg_scores)
            self.states['train_seg_acc_hist'].append(train_seg_acc)
            print('HERE 1')

            # Run the model on the validation set and gather segmentation and classification accuracy
            self.validate_model(gpu_device, val_loader, lam)
            print('HERE 2')

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
            if (epoch+1) > 0:
                current_val_seg_acc = self.states['val_seg_acc_hist'][-1]
                best_val_seg_acc = self.states['best_val_seg_acc']
                is_best = current_val_seg_acc > best_val_seg_acc
                self.states['epoch'] = epoch

                if is_best or (epoch+1) % 10 == 0:
                    self.states['best_val_seg_acc'] = max(current_val_seg_acc, best_val_seg_acc)

                    # model_state = self.update_model_state(epoch, self.model)
                    self.save_checkpoint(self.update_model_state(optim), lam, is_best, dset_name)

        self.reset_histories_and_losses()
        self.model = None

# # Calculate IoU and Mean accuracies
# num_classes = val_outputs_seg.size(1)
# print(num_classes)
# val_confusion = np.zeros((num_classes,3))
# IoU = 0
# mean_acc = 0
#
# for batch in val_loader:
#     val_rgb_inputs = Variable(batch[0].cuda(gpu_device))
#     val_d_inputs = Variable(batch[1].cuda(gpu_device))
#     val_labels = Variable(batch[2].cuda(gpu_device))
#     val_class_labels = Variable(batch[3].cuda(gpu_device))
#     val_outputs, val_class_outputs = self.best_model(val_rgb_inputs, val_d_inputs)
#     _, val_preds = torch.max(val_outputs, 1)
#
#     val_labels = val_labels - 1
#
#     for i in range(num_classes):
#         val_labels_mask = val_labels == i
#         val_preds_mask = val_preds == i
#         TP = np.sum((val_preds == val_labels)[val_labels_mask].data.cpu().numpy())
#
#         val_confusion[i,0] += TP
#         val_confusion[i,1] += np.sum((val_labels==val_labels)[val_labels_mask].data.cpu().numpy()) - TP
#         val_confusion[i,2] += np.sum((val_preds==val_preds)[val_preds_mask].data.cpu().numpy()) - TP
#
# for i in range(num_classes):
#     TP, FP, FN = val_confusion[i]
#     print(TP+FP,FN)
#     IoU += TP / (TP + FP + FN)
#     mean_acc += TP / (TP + FP)
# IoU /= num_classes
# mean_acc /= num_classes

        print("[FINAL] TRAINING COMPLETED")
# print("        Best VALIDATION Accuracy: %.3f IoU: %.3f Mean Accuracy: %.3f" % (self.best_val_seg_acc, IoU, mean_acc))
# print("        Orgnal. FuseNet Accuracy: 0.66  IoU: 0.327 Mean Accuracy: 0.434")
