import rasterio.plot
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from pathlib import Path
import rasterio
import imgaug
from collections import defaultdict
import imgaug.augmenters as iaa
import random
import torch

class SpaceNetDataset(Dataset):
    def __init__(self, root, csv_path, bands: list):
        self.root = Path(root)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.polygons = defaultdict(list)  # image name: list of shapely polygons
        self.parse_polygons(csv_path)
        self.bands = bands
        self.images = list(self.root.joinpath('MUL-PanSharpen').glob('*.tif'))
        self.seq = iaa.Sequential([
            iaa.CropToFixedSize(width=128, height=128),
        ])

        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def parse_polygons(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df[df.BuildingId != -1]
        for i, row in df.iterrows():
            points_str = row.PolygonWKT_Pix[10:-2].split(',')
            points_arr = np.array([np.fromstring(s[:-2], dtype=float, sep=' ') for s in points_str])
            try:
                self.polygons['MUL-PanSharpen_'+row.ImageId].append(imgaug.Polygon(points_arr))
            except Exception:
                print(i, points_str)

    def __getitem__(self, idx):
        # load images and polygon
        img_path = self.images[idx]
        with rasterio.open(img_path.as_posix()) as img:
            img_list = [img.read(j) for j in self.bands]
        img_list = [im.astype(np.float32) / (2 ** 16 - 1) for im in img_list]  # [0-1]
        img = np.dstack(img_list)
        polygons = imgaug.PolygonsOnImage(polygons=self.polygons[img_path.stem], shape=img.shape)  # array of points (array(4,2))
        crop1, poly1 = self.seq(image=img, polygons=polygons)  # random factor
        crop2, poly2 = self.seq(image=img, polygons=polygons)
        try:
            lbl1 = int(poly1.remove_out_of_image_fraction_(0.1).empty)
        except: lbl1 = 0
        pass
        #print(poly1.remove_out_of_image_fraction_(0.1).empty)
        try:
            lbl2 = int(poly2.remove_out_of_image_fraction_(0.1).empty)
        except: lbl2 = 0,
        pass
        t1 = torch.from_numpy(np.transpose(crop1, (2, 0, 1)))
        t2 = torch.from_numpy(np.transpose(crop2, (2, 0, 1)))
        try:
            lbl_all = int(not (lbl1 ^ lbl2))
        except: lbl_all = 0
        pass
        return t1, t2, lbl_all

    def get_dataloaders(self, batch_size, train_split=0.8):
        idx = np.arange(len(self.images))
        train_size = int(len(self.images)*train_split)
        train_idx = idx[:train_size]
        val_idx = idx[train_size:]
        train_dl = DataLoader(dataset=Subset(self, indices=train_idx), shuffle=True, batch_size=batch_size)
        val_dl = DataLoader(dataset=Subset(self, indices=val_idx), batch_size=batch_size)
        return train_dl, val_dl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv_file', type=str, help='path to polygons csv file to parse')
    parser.add_argument('root', type=str, help='path to images root')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='add debug prints')
    args = parser.parse_args()
    print(vars(args))

    dataset = SpaceNetDataset(root=args.root, csv_path=args.csv_file, bands=[1, 2])
    x = dataset[2]
    #csv_file = pd.read_csv()

