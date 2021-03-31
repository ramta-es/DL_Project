# -*- encoding: utf-8 -*-
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),#64x64@1 ->60x60@32
            nn.PReLU(32),
            nn.MaxPool2d(3, stride=3),#60x60@32 -> 20x20@32
            nn.Conv2d(32, 64, kernel_size=6),#20x20@32 -> 15x15x@64
            nn.PReLU(64),
            nn.MaxPool2d(3, stride=3),#15x15@64 -> 5x5@64
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),
            nn.PReLU(256),
            nn.Linear(256, 256))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

###################################################################
###################################################################

class SiameseNetworkFullyConv(nn.Module):#128x128
    def __init__(self):
        super(SiameseNetworkFullyConv, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9),#128x128@1-> 120x120@32
            nn.PReLU(32),#120x120@32
            nn.MaxPool2d(3, stride=3),#120x120@32 -> 40x40@32
            nn.Conv2d(32, 64, kernel_size=5),#40x40@32 -> 36x36@64
            nn.PReLU(64),#36x36@64
            nn.MaxPool2d(2, stride=2),#36x36@64 -> 18x18@64
            nn.Conv2d(64, 128, kernel_size=3),#18x18@64 -> 16x16@128
            nn.PReLU(128),#16x16@128
            nn.MaxPool2d(4, stride=4),#16x16@128 -> 4x4@128
            nn.Conv2d(128, 128, kernel_size=3),#4x4@128 -> 2x2@128
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=2),  # 2x2@128 -> 1x1@256
            )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
###########################################################################
###########################################################################

class SiameseNetworkSecondVersion(nn.Module):
    def __init__(self):
        super(SiameseNetworkSecondVersion, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9),#128x128@1 ->120x120@32
            nn.PReLU(32),
            nn.MaxPool2d(2, stride=2),#120x120@32 -> 60x60@32
            nn.Conv2d(32, 64, kernel_size=7),#60x60@32 -> 54x54x@64
            nn.PReLU(64),
            nn.MaxPool2d(3, stride=3),#54x54@64 -> 18x18@64
            nn.Conv2d(64, 128, kernel_size=5),#18x18@64 -> 14x14@128
            nn.PReLU(128),
            nn.MaxPool2d(2, stride=2),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.PReLU(256),
            nn.Linear(256, 256))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
###########################################################################################
###########################################################################################



class SiameseNetworkFullyConvSmall(nn.Module):#64x64
    def __init__(self):
        super(SiameseNetworkFullyConvSmall, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),#64x64@1-> 60x60@32
            nn.PReLU(32),#60x60@32
            nn.MaxPool2d(3, stride=3),#60x60@32 -> 20x20@32
            nn.Conv2d(32, 64, kernel_size=6),#20x20@32 -> 15x15@64
            nn.PReLU(64),#15x15@64
            nn.MaxPool2d(3, stride=3),#15x15@64 -> 5x5@64
            nn.Conv2d(64, 128, kernel_size=3),#5x5@64 -> 3x3@128
            nn.PReLU(128),#3x3@128
            nn.Conv2d(128, 128, kernel_size=2),#3x3@128 -> 2x2@128
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=2),  # 2x2@128 -> 1x1@256
            )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2