import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    def __init__(self,args):
        super(DeepFM,self).__init__()
        