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
from dataset import build_dataset
import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
train the input model on the training dataset
"""
def train(model, args):
    trainDataset = build_dataset(args.train_dir, isTrain=True)
    valDataset = build_dataset(args.val_dir, isTrain=False)
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    train_loss_epoch = []
    val_loss_epoch = []
    train_acc_epoch = []
    val_acc_epoch = []

    print('Training Started!')
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        train_loss = 0
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
            with torch.no_grad():
                correct += torch.sum(torch.argmax(output, dim=1)==label).item()
            total += img.shape[0]
            train_loss += loss.item()

            # print statistics after every 40 batches
            if((i+1) % 40 == 0):
                print('Epoch: [{}/{}][{}/{}]\t'
                      'Loss {:.3f} | Train Acc {:.3f}'
                      .format(epoch+1, args.epochs, i+1,
                       len(train_loader), loss.item(), correct/total))

        # update information after one epoch
        train_loss /= len(train_loader)
        train_acc = correct/total
        val_acc, val_loss = evaluate(model, criterion, val_loader)
        train_acc_epoch.append(train_acc)
        val_acc_epoch.append(val_acc)
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)

        # print statistics after one epoch
        print('Epoch: [{}/{}][{}/{}]\t'
                      'Loss {:.3f} | Train Acc {:.3f} | Val Acc {:.3f}'
                      .format(epoch+1, args.epochs, len(train_loader),len(train_loader),
                       train_loss, train_acc, val_acc))

        # check if need to adjust lr
        _adjust_lr(optimizer, args.init_lr, epoch+1) 
    
    # output model when training is over
    util.save_model(model, args.model_dir, args.model)
    util.plot_performance(train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, args.out_dir)
    print('Training Finished!')
    print('Training Time: {:.2f}s'.format(time.time()-start_time))
    print("Final acc on the training set: {:.3f}".format(train_acc))
    print("Final acc on the validation set: {:.3f}".format(val_acc))


"""
Evaluate on the validation set
"""
def evaluate(model, criterion, val_loader):
    model.eval()
    count = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            img, label = data
            output = model(img.to(device))
            val_loss += criterion(output, label).item()
            output = np.argmax(output.cpu().numpy(), axis=1)
            count += np.sum(output==label.numpy())
            total += img.shape[0]
    acc = count / total
    val_loss = val_loss / len(val_loader)
    return acc, val_loss

"""
adjust lr every `step_size` epochs by `decay`
"""
def _adjust_lr(optimizer, init_lr, epoch_num, decay=0.6, step_size=2):
    lr = init_lr * (decay ** (epoch_num//step_size))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--train_dir', type=str, default="./dataset/images/train")
    parser.add_argument('--val_dir', type=str, default="./dataset/images/val")
    parser.add_argument('--model_dir', type=str, default="./models")
    parser.add_argument('--out_dir', type=str, default="./results")
    args = parser.parse_args()

    if(args.model == "customized"):
        model = Net()
    elif(args.model == "resnet18"):
        model = resnet.resnet18(num_classes=7)
    elif(args.model == "resnet50"):
        model = resnet.resnet50(num_classes=7)

    model = model.to(device)
    train(model, args)