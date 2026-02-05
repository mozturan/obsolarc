import torch.nn as nn
from abc import abstractmethod

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

class BasePolicy(BaseNetwork):
    def __init__(self):
        super(BasePolicy, self).__init__()

    @staticmethod
    def initialize_weights(module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        # """
        # Initialize the weights of the given module 
        # using Xavier uniform initialization 
        # for linear layers.
        # """
        # if isinstance(module, nn.Linear):
        #     nn.init.xavier_uniform_(module.weight)
        #     if module.bias is not None:
        #         nn.init.constant_(module.bias, 0.0)
