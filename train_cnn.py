import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import os

import util
from net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
train the input model on the training dataset
"""
def train(model, args):
    trainDataset = util.build_dataset(args.train_dir, isTrain=True)
    valDataset = util.build_dataset(args.val_dir, isTrain=False)
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    print('Training Started!')
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        mean_loss = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            img, label = data
            img = img.to(device)
            label = label.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # update information
            correct += torch.sum(torch.argmax(output, dim=1)==label)
            total += img.shape[0]
            mean_loss += loss.item()

            # print statistics after every 40 batches
            if((i+1) % 40 == 0):
                print('Epoch: [{}/{}][{}/{}]\t'
                      'Loss {:.3f} | Train Acc {:.3f}'
                      .format(epoch+1, args.epochs, i+1,
                       len(train_loader), loss.item(), correct/total))

        # print statistics after one epoch
        mean_loss /= len(train_loader)
        train_acc = correct/total
        val_acc = evaluate(model, val_loader)
        print('Epoch: [{}/{}][{}/{}]\t'
                      'Loss {:.3f} | Train Acc {:.3f} | Val Acc {:.3f}'
                      .format(epoch+1, args.epochs, len(train_loader),len(train_loader),
                       mean_loss, train_acc, val_acc))

        # check if need to adjust lr
        _adjust_lr(optimizer, args.init_lr, epoch+1) 
    
    # output model when training is over
    if(not os.path.exists(args.out_dir)):
        os.makedirs(args.out_dir)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))
    print('Training Finished!')
    print('Training Time: {:.2f}s'.format(time.time()-start_time))
    print("Final Val Acc: {:.3f}".format(val_acc))


"""
Evaluate on the validation/test set
"""
def evaluate(model, val_loader):
    model.eval()
    count = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            img, label = data
            output = model(img.to(device))
            output = np.argmax(output.cpu().numpy(), axis=1)
            count += np.sum(output==label.numpy())
            total += img.shape[0]
    acc = count / total
    return acc

"""
adjust lr every `step_size` epochs by `decay`
"""
def _adjust_lr(optimizer, init_lr, epoch_num, decay=0.3, step_size=10):
    lr = init_lr * (decay ** (epoch_num//step_size))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=56)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_dir', type=str, default="./dataset/images/train")
    parser.add_argument('--val_dir', type=str, default="./dataset/images/val")
    parser.add_argument('--out_dir', type=str, default="./models")
    args = parser.parse_args()

    model = Net()
    model = model.to(device)
    train(model, args)