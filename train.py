import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import argparse
import os

import util
from net import Net

def train(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainDataset = util.build_dataset(args.img_dir, isTrain=True)
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    print('Training Started')
    start_time = time.time()

    model.train()

    for epoch in range(args.epochs):

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            img, label = data
            img = img.to(device)
            label = label.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # print statistics after every 20 batch
            if((i+1) % 20 == 0):
                print('Epoch: [{}/{}][{}/{}]\t'
                      'Loss {:.4f}'.format(
                       epoch+1, args.epochs, i+1, len(train_loader), loss.item()))
            
        _adjust_lr(optimizer, args.init_lr, epoch+1) 
    
    if(not os.path.exists(args.out_dir)):
        os.makedirs(args.out_dir)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))
    print('Training Finished')
    print('Training Time: {:.2f}s'.format(time.time()-start_time))


# adust lr every 2 epochs
def _adjust_lr(optimizer, init_lr, epoch_num):
    lr = init_lr * (0.8 ** (epoch_num//2))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img_dir', type=str, default="./dataset/images/train")
    parser.add_argument('--out_dir', type=str, default="./models")
    args = parser.parse_args()

    model = Net()
    train(model, args)