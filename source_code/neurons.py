import torch
import torch.nn as nn
from typing import List

class CW_softIF(nn.Module):
    def __init__(self, batch_size: int, num_channels: int, height: int, width: int, threshold: List[float] = None, reset:float=0.0) -> None:
        super(CW_softIF, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.reset = reset
        self.threshold = None
        if not threshold:
            self.threshold = [1.0] * num_channels
        elif threshold:
            if len(threshold)==self.num_channels:
                self.threshold = self.__setthreshold(threshold)
            else:
                raise Exception ('size of threshold must be equal to number of channels')

        self.v = torch.zeros(size=(batch_size, num_channels, height, width),
                             device='cuda' if torch.cuda.is_available() else 'cpu')
        self.s = torch.zeros(size=(batch_size, num_channels, height, width),
                             device='cuda' if torch.cuda.is_available() else 'cpu')


    def update_threshold(self, threshold, scaling_factor=1.):
        if len(threshold) == self.num_channels:
            self.threshold = self.__setthreshold(threshold, scaling_factor)
        else:
            raise Exception('size of threshold must be equal to number of channels')

    def __setthreshold(self, threshold, scaling_factor=1.):
        update_threshold = torch.zeros(size=(self.num_channels, self.height, self.width),
                                       device='cuda' if torch.cuda.is_available() else 'cpu')

        for j in range(len(threshold)):
            update_threshold[j].fill_(threshold[j] * scaling_factor)
        return update_threshold

    def reset_variables(self, batch_size:int):
        self.batch_size = batch_size
        self.v = torch.full(size=(batch_size, self.v.size(1), self.v.size(2), self.v.size(3)),
                            fill_value=self.rst.item(), device=self.v.device)
        self.s = torch.zeros(size=(batch_size, self.s.size(1), self.s.size(2), self.s.size(3)), device=self.s.device)

    def init_vars(self):
        self.v.fill_(self.reset)
        self.s.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [batch size, # channels, height, width] input currents
            v: [batch size, # channels, height, width] membrane potentials
            s: [batch size, # channels, height, width] spikes
        :param x:
        :return:
        """
        th = self.threshold.expand(self.batch_size, self.threshold.shape[0],self.threshold.shape[1], self.threshold.shape[2])
        self.v += x
        self.s[:] = (self.v >= th).float()
        self.v[self.v >= th] -=th[self.v >= th]
        return self.s
    pass


class PB_neurons(nn.Module):
    def __init__(self, batch_size: int, num_features: int,reset:float=0.0) -> None:
        """
            v: [batch size, # features] membrane potentials
        """
        super(PB_neurons, self).__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        self.reset = reset
        self.v = torch.zeros(size=(batch_size, num_features),
                             device='cuda' if torch.cuda.is_available() else 'cpu')

    def init_vars(self):
        self.v.fill_(self.reset)

    def reset_variables(self, batch_size:int):
        self.v =torch.full(size=(batch_size, self.v.size(1)), fill_value=self.reset, device=self.v.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x.size: [batch size, # features]
        :param x:
        :return:
        """
        self.v += x
        return self.v
