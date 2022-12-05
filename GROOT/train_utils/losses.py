import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_MLSM_Loss(nn.Module):
        def __init__(self):
            super(Custom_MLSM_Loss, self).__init__()

        def forward(self, input, target,wts=None ,smooth=1e-8):
            bs=target.shape[0]
            classes=target.shape[1]

            if wts is None:
                target1 = target.view(-1)
            else: 
                target1 = (target*wts).view(-1) #*((1-input)**2)   
            input = input.view(-1)
            target = target.view(-1)

            partial_ce=(target1*(target*torch.log(F.sigmoid(input)+smooth)) + ((1-target)*torch.log(F.sigmoid((1-input))+smooth))).sum()
                         
            loss_classes_avg = partial_ce/classes
            final_loss=-(loss_classes_avg/(bs))

            return final_loss