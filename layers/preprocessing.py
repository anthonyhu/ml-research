import torch
import torch.nn as nn


class ImagePreprocessing(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.mean = torch.nn.Parameter(self.mean, requires_grad=False)
        self.std = torch.nn.Parameter(self.std, requires_grad=False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x
