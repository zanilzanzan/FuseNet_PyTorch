import torch
import torch.nn.functional as F

sun_weights = torch.cuda.FloatTensor([0.31142759, 0.26649606, 0.45942909, 0.32240534, 0.54789394, 0.42697880,
                                      0.76315141, 1.11409545, 0.96722591, 0.57659554, 1.66651666, 0.85155034,
                                      1.03507304, 0.59151018, 1.07225466, 0.76207125, 0.67946768, 2.38537860,
                                      1.64862466, 1.75271165, 3.24660635, 1.16477966, 2.37583423, 0.87280464,
                                      1.55249476, 5.12412119, 1.94428802, 0.64293331, 3.18023825, 0.85495919,
                                      3.15664768, 2.11753082, 0.55160081, 1.57176685, 5.13662910, 0.45877823,
                                      4.90023994])

# nyu_weights = torch.cuda.FloatTensor([0.272491, 0.568953, 0.432069, 0.354511, 0.821780, 0.506488, 1.133686,
#                                       0.812170, 0.789383, 0.380358, 1.650497, 1.000000, 0.650831, 0.757218,
#                                       0.950049, 0.614332, 0.483815, 1.842002, 0.635787, 1.176839, 1.196984,
#                                       1.111907, 1.927519, 0.695354, 1.057833, 4.179196, 1.571971, 0.432408,
#                                       3.705966, 0.549132, 1.282043, 2.329812, 0.992398, 3.114945, 5.466101,
#                                       1.085242, 6.968411, 1.093939, 1.336520, 1.228912])


def cross_entropy_2d():
    def wrap(seg_preds, seg_targets, class_inputs=None, class_targets=None,
             lambda_1=1.0, lambda_2=1.0, weight=None, pixel_average=True, use_class=True):

        # If the dataset is SUN RGB-D use class normalization weights in order to introduce balance to calculated loss as the number
        # of classes in SUN RGB-D dataset are not uniformly distributed.
        n, c, h, w = seg_preds.size()
        if c == 37:
            weight = sun_weights

        # Calculate segmentation loss
        seg_inputs = seg_preds.transpose(1, 2).transpose(2, 3).contiguous()
        seg_inputs = seg_inputs[seg_targets.view(n, h, w, 1).repeat(1, 1, 1, c) > 0].view(-1, c)

        # Exclude the 0-valued pixels from the loss calculation as 0 values represent the pixels that are not annotated.
        seg_targets_mask = seg_targets > 0
        # Subtract 1 from all classes, in the ground truth tensor, in order to match the network predictions.
        # Remember, in network predictions, label 0 corresponds to label 1 in ground truth.
        seg_targets = seg_targets[seg_targets_mask] - 1

        # Calculate segmentation loss value using cross entropy
        seg_loss = F.cross_entropy(seg_inputs, seg_targets, weight=weight, size_average=False)

        # Average the calculated loss value over each labeled pixel in the ground-truth tensor
        if pixel_average:
            seg_loss /= seg_targets_mask.float().data.sum()
        loss = lambda_1 * seg_loss

        # If scene classification function is utilized, calculate class loss, multiply with coefficient, lambda_2, sum with total loss
        if use_class:
            # Calculate classification loss
            class_targets -= 1
            class_loss = F.cross_entropy(class_inputs, class_targets)
            # Combine losses
            loss += lambda_2 * class_loss
            return loss, seg_loss, class_loss
        return loss
    return wrap
