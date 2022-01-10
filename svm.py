import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import time

from feature_extraction import *

# sift_bof
K = 15
BATCH_SIZE = 256
MAX_ITER = 1000

# PCA & NMF
D = 20

# LDA
R = 6

# feature extraction method
EXTRACTOR = nmf

# SVC parameters
SVC_PARAMS = {'kernel':'rbf',
          'decision_function_shape':'ovo',
          }


def train(X_train, y_train, params=None):
    print("Start training...")
    start_time = time.time()

    # if params is not given, use GridSearchCV to tune
    if(params is None):
        params = {'C': [0.01, 0.1, 1, 10, 100, 500],
                'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                'decision_function_shape': ['ovo']}
        svc = GridSearchCV(SVC(random_state=0), params, cv=3)
        svc.fit(X_train, y_train)
        model = svc.best_estimator_
    else:
        model = SVC(**params).fit(X_train, y_train)

    print("  - Training time: {0:.1f} sec".format(time.time()-start_time))
    print("  - Model Parameters: ", model.get_params(), sep='\n')

    return model


def evaluate(model, X_val, y_val):
    print("Start evaluating...")
    start_time = time.time()
    y_val_hat = model.predict(X_val)
    # print(y_val_hat)

    print("  - Evalution time: {0:.1f} sec".format(time.time() - start_time))
    print("  - Accuracy Score: {:.1f}%".format(100*sum(y_val==y_val_hat)/len(y_val)))
    print("  - Confusion Matrix: ", confusion_matrix(y_val, y_val_hat), sep='\n')


def load_data(path, tag):
    print("Loading {} data...".format(tag))
    X = []
    y = []
    size = {"training":22900, "validation":5700}
    img_path_list = os.listdir(path)
    for d in img_path_list[:size[tag]]:
        image_path = path+"/"+d
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        X.append(img)
        y.append(int(d[-5]))
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X_train, y_train = load_data("./dataset/images/train", "training") # N*H*W
    X_val, y_val = load_data("./dataset/images/val", "validation")

    if EXTRACTOR is sift_bof:
        X_train_new, X_val_new, y_train, y_val = EXTRACTOR(X_train, X_val, y_train, y_val, K, BATCH_SIZE, MAX_ITER)
    elif EXTRACTOR is pca:
        X_train_new, X_val_new = EXTRACTOR(X_train, X_val, D)
    elif EXTRACTOR is nmf:
        X_train_new, X_val_new = EXTRACTOR(X_train, X_val, D)
    elif EXTRACTOR is lda:
        X_train_new, X_val_new = EXTRACTOR(X_train, y_train, X_val, R)

    model = train(X_train_new, y_train, params=None)
    evaluate(model, X_val_new, y_val)
