import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from torchvision import models, transforms
import argparse

import util
from net import NetA, NetB, NetC
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

    util.plot_confusion_matrix(all_labels.cpu().numpy(), all_predicted.cpu().numpy(),
         args.out_dir, args.model)
    acc = count / total
    print('Accuracy: {:.1f}%'.format(acc*100))
    print('Testing Time: {:.2f}s'.format(time.time()-start_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_dir', type=str, default="./dataset/images/test")
    parser.add_argument('--train_dir', type=str, default="./dataset/images/train")
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--model_dir', type=str, default="./models")
    parser.add_argument('--out_dir', type=str, default="./results")
    args = parser.parse_args()

    if(args.model == "netA"):
        model = NetA()
        data_transform=transforms.Compose([  
            transforms.ToTensor(),
            transforms.Grayscale()
            ])
    elif(args.model == "netB"):
        model = NetB()
        data_transform=transforms.Compose([  
            transforms.ToTensor(),
            transforms.Grayscale()
            ])
    elif(args.model == "netC"):
        model = NetC(util.get_pca_model(args.train_dir, D=256))
        data_transform = transforms.Compose([    
            transforms.ToTensor(),
            transforms.Grayscale(),
        ])
    elif(args.model == "resnet18"):
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 7)
        data_transform=transforms.Compose([  
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
    elif(args.model == "vgg19"):
        model = models.vgg19()
        model.classifier = nn.Sequential(*list(model.children())[-1][:4])
        model.classifier[-1].out_features = 7
        data_transform=transforms.Compose([  
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])


    model = model.to(device)
    util.load_model(model, args.model_dir, args.model)
    testDataset = build_dataset(args.test_dir, transform=data_transform)

    evaluate(model, testDataset, args)
