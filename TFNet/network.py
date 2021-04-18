import torch
import torch.nn as nn

class Acon_FReLU(nn.Module):
    r""" ACON activation (activate or not) based on FReLU:
    # eta_a(x) = x, eta_b(x) = dw_conv(x), according to
    # "Funnel Activation for Visual Recognition" <https://arxiv.org/pdf/2007.11824.pdf>.
    """
    def __init__(self, width, stride=1):
        super().__init__()
        self.stride = stride

        # eta_b(x)
        self.conv_frelu = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=width, bias=True)
        self.bn1 = nn.BatchNorm2d(width)

        # eta_a(x)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        if self.stride == 2:
            x1 = self.maxpool(x)
        else:
            x1 = x

        x2 = self.bn1(self.conv_frelu(x))

        return self.bn2( (x1 - x2) * self.sigmoid(x1 - x2) + x2 )


class TFBlock(nn.Module):
    def __init__(self, inp, stride):
        super(TFBlock, self).__init__()
        self.oup = inp * stride
        self.stride = stride

        branch_main = [
            # pw conv
            nn.Conv2d(inp, inp, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(inp),
            Acon_FReLU(inp),
            # pw conv
            nn.Conv2d(inp, inp, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(inp)
        ]
        self.branch_main = nn.Sequential(*branch_main)

        self.acon = Acon_FReLU(self.oup, stride)

    def forward(self, x):
        x_proj = x
        x = self.branch_main(x)

        if self.stride==1:
            return self.acon(x_proj + x)

        elif self.stride==2:
            return self.acon(torch.cat((x_proj, x), 1))


class TFNet(nn.Module):
    def __init__(self, n_class=1000, model_size=0.5):
        super(TFNet, self).__init__()
        print('model size is ', model_size)

        self.stages = [2, 3, 8, 3]
        self.in_channel = int(16 * model_size)
        self.out_channel = 1024
        self.model_size = model_size

        # building the first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, self.in_channel, 3, 2, 1, bias=True),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

        # building the four stages' features
        self.features = []
        for stage in self.stages:
            for i in range(stage):
                self.features.append(
                        TFBlock(self.in_channel, stride = 1 if i > 0 else 2))
                self.in_channel = self.in_channel * 2 if i == 0 else self.in_channel
        self.features = nn.Sequential(*self.features)

        # building the last layer
        self.conv_last = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.out_channel),
            Acon_FReLU(self.out_channel),
        )
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size > 0.5:
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.out_channel, n_class, bias=True))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size > 0.5:
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.out_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'frelu' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
