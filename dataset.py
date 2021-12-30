from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import cv2

import util

"""
util function for buiding a cnn dataset
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

"""
Wrapper class for FER Face Dateset
"""
class FaceDataset(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform

        self.img_names = {i:name for i,name in enumerate(os.listdir(img_dir))}

    def __len__(self):
        return len(self.img_names)
        # return 2000

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.img_names[idx])

        label = int(self.img_names[idx][-5])

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if(self.transform):
            img = self.transform(img)

        return img, label


#### For testing
if __name__ == '__main__':
    imgDataset = build_dataset('./dataset/images/val', isTrain=False)
    print(len(imgDataset))
    print(imgDataset[0][0].shape)
    print(imgDataset[30][0], imgDataset[0][1])