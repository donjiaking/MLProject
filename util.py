import torch
from torchvision import transforms

from dataset import FaceDataset

"""
util function for buiding cnn dataset
"""
def build_dataset(img_dir, isTrain=True):
    ## train data augmentation
    transform_train = transforms.Compose([    
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30)
        ])

    ## test data augmentation
    transform_test = transforms.Compose([  
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        ])
    
    if(isTrain):
        faceDataset = FaceDataset(img_dir, transform_train)
    else:
        faceDataset = FaceDataset(img_dir, transform_test)

    return faceDataset 


