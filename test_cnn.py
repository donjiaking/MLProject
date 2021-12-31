from numpy.core.fromnumeric import size
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

import util
from net import Net
from resnet import resnet18
from dataset import build_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Evaluate the model given a dataset
"""
def evaluate(model, testDataset, args):
    test_loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)
    model.eval()

    count = 0.0
    total = 0.0
    print("Testing Started")  
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # get the input
            img, label = data
            img = img.to(device)
            label = label.to(device)
            # get the output
            output = model(img)
            output = torch.argmax(output, dim=1)
            # count the number
            count += torch.sum(output==label)
            total += img.shape[0]

            # print statistics after every 10 batch
            if((i+1) % 10 == 0):
                print('Process: [{}/{}]\t'.format(i+1, len(test_loader)))
            
            if i == 0:
                all_predicted = output
                all_labels = label
            else:
                all_predicted = torch.cat((all_predicted, output),dim=0)
                all_labels = torch.cat((all_labels, label),dim=0)

    util.plot_confusion_matrix(all_labels.cpu().numpy(), all_predicted.cpu().numpy(), args.out_dir)
    acc = count / total
    print('Accuracy: {:.1f}%'.format(acc*100))
    print('Testing Time: {:.2f}s'.format(time.time()-start_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_dir', type=str, default="./dataset/images/val")
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--model_dir', type=str, default="./models")
    parser.add_argument('--out_dir', type=str, default="./results")
    args = parser.parse_args()

    if(args.model == "customized"):
        model = Net()
    elif(args.model == "resnet18"):
        model = resnet18(num_classes=7)

    model = model.to(device)
    util.load_model(model, args.model_dir, args.model)
    testDataset = build_dataset(args.img_dir, isTrain=False)

    evaluate(model, testDataset, args)
