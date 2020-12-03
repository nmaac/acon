import torch
from torch import nn

class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x
    """
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Conv2d(width, width//16, kernel_size=1, stride=1, bias=False)
        self.fc2 = nn.Conv2d(width//16, width, kernel_size=1, stride=1, bias=False)

        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        beta = self.sigmoid(self.fc2(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))
        return (self.p1 * x - self.p2 * x) * self.sigmoid( beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

