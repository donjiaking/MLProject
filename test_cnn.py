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


"""
Evaluate the model given a dataset
"""
def evaluate(model, testDataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    print("Testing Started")  
    start_time = time.time()

    model.eval()

    count = 0.0
    total = 0.0
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
    parser.add_argument('--out_dir', type=str, default="./result")
    args = parser.parse_args()

    model = Net()
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    testDataset = util.build_dataset(args.img_dir, isTrain=False)

    evaluate(model, testDataset, args)
