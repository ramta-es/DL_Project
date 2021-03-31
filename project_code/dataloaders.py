import os
import numpy as np
import torch
import rasterio
import pandas as pd
from PIL import Image
from torch.utils.data import dataset
from torchvision import transforms as T
import torch.utils
import torchvision
from pathlib import Path

import matplotlib.pyplot as plt

SpaceNetDataset(folder, path_csv, T, [1,3,7])





# ---------------------------------------------------------------------------------


def get_transform(train):
    # albumutations
    # set minimal area th
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# ---------------------------------------------------------------------------------
dataset = SpaceNet_dataset('/Users/ramtahor/Desktop/Penn_dataset/PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4)

# ---------------------------------------------------------------------------------
