import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from collections import OrderedDict
import os
import math

gpu_device = 0
torch.cuda.set_device(gpu_device)

class FuseNet(nn.Module):
    def __init__(self, num_labels):
        super(FuseNet, self).__init__()

        
        batchNorm_momentum = 0.1
        feats = list(models.vgg16(pretrained=True).features.children())
        feats2 = list(models.vgg16(pretrained=True).features.children())

        #print('feats[0] shape: ', feats[0].weight.data.size())
        #print('feats[1] shape: ', feats[2].weight.data.size())

        # Take the average of the weights for the depth branch over channel dimension 
        avg = torch.mean(feats[0].cuda(gpu_device).weight.data, dim=1)

        ########  DEPTH ENCODER  ########

        self.conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1).cuda(gpu_device)
        self.conv11d.weight.data = avg 

        self.CBR1_D = nn.Sequential(
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats[1].cuda(gpu_device),
            feats[2].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats[3].cuda(gpu_device),
        )
        self.CBR2_D = nn.Sequential(
            feats[5].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats[6].cuda(gpu_device),
            feats[7].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats[8].cuda(gpu_device),
        )
        self.CBR3_D = nn.Sequential(
            feats[10].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[11].cuda(gpu_device),
            feats[12].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[13].cuda(gpu_device),
            feats[14].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats[15].cuda(gpu_device),
        )

        self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR4_D = nn.Sequential(
            feats[17].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[18].cuda(gpu_device),
            feats[19].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[20].cuda(gpu_device),
            feats[21].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[22].cuda(gpu_device),
        )

        self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR5_D = nn.Sequential(
            feats[24].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[25].cuda(gpu_device),
            feats[26].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[27].cuda(gpu_device),
            feats[28].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats[29].cuda(gpu_device),
        )    

        ########  RGB ENCODER  ########

        self.CBR1_RGB = nn.Sequential (
            feats2[0].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats2[1].cuda(gpu_device),
            feats2[2].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats2[3].cuda(gpu_device),
        )

        self.CBR2_RGB = nn.Sequential (
            feats2[5].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats2[6].cuda(gpu_device),
            feats2[7].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats2[8].cuda(gpu_device),
        )

        self.CBR3_RGB = nn.Sequential (        
            feats2[10].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[11].cuda(gpu_device),
            feats2[12].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[13].cuda(gpu_device),
            feats2[14].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[15].cuda(gpu_device),
        )

        self.dropout3 = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR4_RGB = nn.Sequential (
            feats2[17].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[18].cuda(gpu_device),
            feats2[19].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[20].cuda(gpu_device),
            feats2[21].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[22].cuda(gpu_device),
        )

        self.dropout4 = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR5_RGB = nn.Sequential (        
            feats2[24].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[25].cuda(gpu_device),
            feats2[26].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[27].cuda(gpu_device),
            feats2[28].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[29].cuda(gpu_device),
        )

        self.dropout5 = nn.Dropout(p=0.5).cuda(gpu_device)

        ########  RGB DECODER  ########

        self.CBR5_Dec = nn.Sequential (        
        nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR4_Dec = nn.Sequential (        
        nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Conv2d(512, 256, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(256, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR3_Dec = nn.Sequential (        
        nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(256, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(256, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Conv2d(256,  128, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(128, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR2_Dec = nn.Sequential (
        nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(128, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(64, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),
        )

        self.CBR1_Dec = nn.Sequential (                
        nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
        nn.BatchNorm2d(64, momentum= batchNorm_momentum).cuda(gpu_device),
        nn.ReLU().cuda(gpu_device),        	
        nn.Conv2d(64, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
        )

    def forward(self, rgb_inputs, depth_inputs):

        ########  DEPTH ENCODER  ########

        # Stage 1
        x = self.conv11d(depth_inputs)
        x_1 = self.CBR1_D(x)
        x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)
        
        # Stage 2
        x_2 = self.CBR2_D(x)
        x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x_3 = self.CBR3_D(x)
        x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_D(x)
        x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_D(x)

        ########  RGB ENCODER  ########

        # Stage 1
        y = self.CBR1_RGB(rgb_inputs)
        y = torch.add(y,x_1)
        y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        y = self.CBR2_RGB(y)
        y = torch.add(y,x_2)
        y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        y = self.CBR3_RGB(y)
        y = torch.add(y,x_3)
        y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB(y)
        y = torch.add(y,x_4)
        y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB(y)
        y = torch.add(y,x_5)
        y_size = y.size() 

        y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout5(y)

        ########  DECODER  ########

        # Stage 5 dec
        y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
        y = self.CBR5_Dec(y)

        # Stage 4 dec
        y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
        y = self.CBR4_Dec(y)

        # Stage 3 dec
        y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
        y = self.CBR3_Dec(y)

        # Stage 2 dec
        y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
        y = self.CBR2_Dec(y)

        # Stage 1 dec
        y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
        y = self.CBR1_Dec(y)

        return y

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model: %s' % path
        torch.save(self, path)
        print 'Model saved: %s' % path

def CrossEntropy2d():
    def wrap(inputs, targets, weight=None, pixel_average=True):
        n, c, h, w = inputs.size()

        if c == 37:
            weight = torch.cuda.FloatTensor([0.31142759, 0.26649606, 0.45942909, 0.32240534, 0.54789394, 0.4269788, 0.76315141, 1.11409545, 0.96722591, 0.57659554, 1.66651666, 0.85155034, 1.03507304, 0.59151018, 1.07225466, 0.76207125, 0.67946768, 2.3853786, 1.64862466, 1.75271165, 3.24660635, 1.16477966, 2.37583423, 0.87280464, 1.55249476, 5.12412119, 1.94428802, 0.64293331, 3.18023825, 0.85495919, 3.15664768, 2.11753082, 0.55160081, 1.57176685, 5.1366291, 0.45877823, 4.90023994])
        #elif c == 40:
        #    weight = torch.cuda.FloatTensor([0.272491, 0.568953, 0.432069, 0.354511, 0.82178, 0.506488, 1.133686, 0.81217, 0.789383, 0.380358, 1.650497, 1, 0.650831, 0.757218, 0.950049, 0.614332, 0.483815, 1.842002, 0.635787, 1.176839, 1.196984, 1.111907, 1.927519, 0.695354, 1.057833, 4.179196, 1.571971, 0.432408, 3.705966, 0.549132, 1.282043, 2.329812, 0.992398, 3.114945, 5.466101, 1.085242, 6.968411, 1.093939, 1.33652, 1.228912])
        #print("was here: NYU weight")

        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        inputs = inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) > 0].view(-1, c)

        targets_mask = targets > 0
        targets = targets[targets_mask] - 1

        loss = F.cross_entropy(inputs, targets, weight=weight, size_average=False)
        if pixel_average:
            loss /= targets_mask.data.sum()
        return loss
    return wrap

