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
            # get the output
            output = model(img.to(device))
            output = np.argmax(output.cpu().numpy(), axis=1)
            # count the number
            count += np.sum(output==label.numpy())
            total += img.shape[0]

            # print statistics after every 10 batch
            if((i+1) % 10 == 0):
                print('Process: [{}/{}]\t'.format(i+1, len(test_loader)))

    acc = count / total
    print('Accuracy: {:.1f}%'.format(acc*100))
    print('Testing Time: {:.2f}s'.format(time.time()-start_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_dir', type=str, default="./dataset/images/val")
    parser.add_argument('--model_dir', type=str, default="./models")
    parser.add_argument('--out_dir', type=str, default="./results")
    args = parser.parse_args()

    model = Net()
    model = model.to(device)
    model = util.load_model(model, args.model_dir)
    testDataset = build_dataset(args.img_dir, isTrain=False)

    evaluate(model, testDataset, args)
