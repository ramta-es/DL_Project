from space_net_dataset import SpaceNetDataset
from contrastive import ContrastiveLoss
from net import SiameseNetwork, SiameseNetworkFullyConv

import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import trange
import argparse

def save_checkpoint(root: Path, model, epoch, optimizer, better):
    if better:
        fpath = root.joinpath('best_checkpoint.pth')
    else:
        fpath = root.joinpath('last_checkpoint.pth')
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, fpath)

def train(device, model, criterion, optimizer, train_loader, test_loader, epochs, root: Path):
    model.zero_grad()
    best_loss = 2 ** 16
    train_writer = SummaryWriter(root.joinpath('train'))
    test_writer = SummaryWriter(root.joinpath('test'))

    for epoch in trange(epochs, desc='train'):
        model.train()
        train_loss = 0
        for x1, x2, y in train_loader:
            in1, in2, labels = x1.to(device), x2.to(device), y.to(device)
            out1, out2 = model(in1, in2)
            optimizer.zero_grad()
            loss = criterion(out1, out2, labels)
            loss.backward()
            train_loss += loss.item() / len(train_loader)
            optimizer.step()
        train_writer.add_scalar(tag='loss', scalar_value=train_loss, global_step=epoch)

        model.eval()
        test_loss = 0
        for x1, x2, y in test_loader:
            in1, in2, labels = x1.to(device), x2.to(device),  y.to(device)
            out1, out2 = model(in1, in2)
            optimizer.zero_grad()
            loss = criterion(out1, out2, labels)
            test_loss += loss.item() / len(test_loader)
        test_writer.add_scalar(tag='loss', scalar_value=test_loss, global_step=epoch)

        if test_loss < best_loss:
            best_loss = test_loss
            save_checkpoint(root, model, epoch, optimizer, test_loss == best_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv_file', type=str, help='path to polygons csv file to parse')
    parser.add_argument('root', type=str, help='path to images root')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='add debug prints')
    args = parser.parse_args()

    dataset = SpaceNetDataset(root=args.root, csv_path=args.csv_file, bands=[7])

    if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print('using device: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print('using cpu')

    criterion = ContrastiveLoss()
    train_loader, test_loader = dataset.get_dataloaders(batch_size=16)
    model = SiameseNetwork()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    p = Path('exp1')
    train(device, model, criterion, optimizer, train_loader, test_loader, epochs=100, root=p)


    # cd /Users/ramtahor/PycharmProjects/SpaceNet_PreProcessing
    # ./venv/bin/tensorboard --logdir exp1