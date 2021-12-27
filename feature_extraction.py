import os
import cv2
import numpy as np
from sklearn.decomposition import NMF, PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import time


def sift_bof(X_train, X_val, K, BATCH_SIZE, MAX_ITER):
    print("Extracting features...")
    start_time = time.time()

    sift = cv2.xfeatures2d.SIFT_create()
    train_dst = [sift.detectAndCompute(img, None)[1] for img in X_train]
    val_dst = [sift.detectAndCompute(img, None)[1] for img in X_val]
    train_features = np.array([f for d in train_dst for f in d])
    # val_features = np.array([f for d in train_dst for f in d])
    
    model = MiniBatchKMeans(n_clusters=K, batch_size=BATCH_SIZE, max_iter=MAX_ITER).fit(train_features)
    bof_train = [np.bincount(i, minlength=K) for i in [model.predict(d) for d in train_dst]]
    bof_val = [np.bincount(i, minlength=K) for i in [model.predict(d) for d in val_dst]]

    scaler = StandardScaler()
    X_train_new = scaler.fit_transform(bof_train)
    X_val_new = scaler.transform(bof_val)

    print("  - Extraction time: {0:.1f} sec".format(time.time() - start_time))

    return X_train_new, X_val_new


def pca(X_train, X_val, D):
    print("Extracting features...")
    start_time = time.time()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_train = X_train.T
    X_val = X_val.T

    mean = np.mean(X_train, axis=1)   # the mean of train set
    X_train_c = X_train - mean.reshape(-1,1) # centralize
    X_val_c = X_val - mean.reshape(-1,1)  # centralize

    U, sigma, VT = np.linalg.svd(X_train_c)
    
    singular_values = sigma[:D]
    Ud = U[:, :D]

    X_train_pca = np.dot(Ud.T, X_train_c).T
    X_val_pca = np.dot(Ud.T, X_val_c).T

    scaler = StandardScaler()
    X_train_pca = scaler.fit_transform(X_train_pca)
    X_val_pca = scaler.transform(X_val_pca)

    print("  - Extraction time: {0:.1f} sec".format(time.time() - start_time))
    
    return X_train_pca, X_val_pca

def nmf(X_train, X_val, D):
    print("Extracting features...")
    start_time = time.time()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    model = NMF(n_components=D, max_iter=400, init='nndsvda')
    X_train_nmf = model.fit_transform(X_train)
    X_val_nmf = model.transform(X_val)

    scaler = StandardScaler()
    X_train_nmf = scaler.fit_transform(X_train_nmf)
    X_val_nmf = scaler.transform(X_val_nmf)

    print("  - Extraction time: {0:.1f} sec".format(time.time() - start_time))

    return X_train_nmf, X_val_nmf