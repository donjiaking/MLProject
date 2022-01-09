import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os

train_path = "./dataset/train.csv"
test_path = "./dataset/test.csv"
img_path = "./dataset/images"

def generate_train_images():
    if(not os.path.exists(train_path)):
        print("You need train.csv to read input!")
        exit()
    if(not os.path.exists(img_path)):
        os.mkdir(img_path)
        os.mkdir(img_path+'/val')
        os.mkdir(img_path+'/test')
        os.mkdir(img_path+'/train')

    train = pd.read_csv(train_path)
    print(train.head())
    data = train[['pixels']]
    label = train[['emotion']]

    X = np.zeros(shape=(len(data['pixels']),48,48))
    for i in range(len(data['pixels'])):
        X[i] = np.reshape((np.fromstring(data.loc[i,'pixels'],dtype=int,sep=' ')),(48,48))
    y = np.array(list(map(int, label['emotion'])))

    X_train, X_valid, y_train, y_valid = \
        train_test_split(X, y, test_size=0.2, random_state=1)

    # generate visible images for train set
    for i in range(X_train.shape[0]):
        image = X_train[i, :]
        cv2.imwrite(img_path + "/train/" + '{}_{}.jpg'.format(i, y_train[i]), image)
        
    # generate visible images for validation set
    for i in range(X_valid.shape[0]):
        image = X_valid[i, :]
        cv2.imwrite(img_path + "/val/" + '{}_{}.jpg'.format(i, y_valid[i]), image)


def generate_test_images():
    if(not os.path.exists(test_path)):
        print("You need test.csv to read input!")
        exit()
    if(not os.path.exists(img_path)):
        os.mkdir(img_path)
        os.mkdir(img_path+'/val')
        os.mkdir(img_path+'/test')
        os.mkdir(img_path+'/train')

    test = pd.read_csv(test_path)
    print(test.head())
    data = test[['pixels']]
    label = test[['emotion']]

    X = np.zeros(shape=(len(data['pixels']),48,48))
    for i in range(len(data['pixels'])):
        X[i] = np.reshape((np.fromstring(data.loc[i,'pixels'],dtype=int,sep=' ')),(48,48))
    y = np.array(list(map(int, label['emotion'])))

    # generate visible images for test set
    for i in range(X.shape[0]):
        image = X[i, :]
        cv2.imwrite(img_path + "/test/" + '{}_{}.jpg'.format(i, y[i]), image)
        

if __name__ == "__main__":
    # generate_train_images()
    generate_test_images()
