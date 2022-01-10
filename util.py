import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.decomposition import PCA
import numpy as np
import cv2
import os

def load_model(model, model_dir, model_name):
    model.load_state_dict(torch.load(os.path.join(model_dir, "{}.pth".format(model_name))))

def save_model(model, model_dir, model_name):
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, "{}.pth".format(model_name)))


def get_pca_model(img_dir, D=128):
    X = []
    img_path_list = os.listdir(img_dir)
    for d in img_path_list:
        img = cv2.imread(img_dir+"/"+d, cv2.IMREAD_GRAYSCALE)
        X.append(img.flatten())
    X = np.array(X)/255.0

    pca_model = PCA(n_components=D)
    pca_model.fit(X)
    return pca_model


def plot_confusion_matrix(labels, predicted, out_dir, name, normalize=True):
    if(not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    matrix = confusion_matrix(labels, predicted)
    # print(matrix)
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    # print(matrix)

    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title("Confusion Matrix", fontsize=18)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45) # x: predicted
    plt.yticks(ticks, classes) # y: true

    threshold = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], '.2f' if normalize else 'd'),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > threshold else "black")

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"confusion_matrix_{}.png".format(name)), format='png')


def plot_performance(train_loss_epoch, val_loss_epoch, train_acc_epoch, val_acc_epoch, out_dir):
    if(not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    x = list(range(1, len(train_loss_epoch)+1))

    plt.plot(x, train_loss_epoch, color='#d46b5f', label="training set")
    plt.plot(x, val_loss_epoch, color='#50c3e6', label="validation set")
    plt.xlabel('epoch')
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.ylim(0, 2)
    plt.legend()
    plt.text(x[-1]-0.02,train_loss_epoch[-1]-0.02,"loss={:.3f}".format(train_loss_epoch[-1]),color='#d46b5f')
    plt.text(x[-1]-0.02,val_loss_epoch[-1]+0.02,"loss={:.3f}".format(val_loss_epoch[-1]),color='#50c3e6')
    plt.scatter([x[-1]], [train_loss_epoch[-1]], s=25, c='#d46b5f')
    plt.scatter([x[-1]], [val_loss_epoch[-1]], s=25, c='#50c3e6')
    plt.savefig(os.path.join(out_dir,"loss"))
    plt.clf()

    plt.plot(x, train_acc_epoch, color='#d46b5f', label="training set")
    plt.plot(x, val_acc_epoch, color='#50c3e6', label="validation set")
    plt.xlabel('epoch')
    plt.ylabel("Accuracy")
    plt.title("Accuracy on the training/validation set")
    plt.ylim(0, 1)
    plt.legend()
    plt.text(x[-1]-0.02,train_acc_epoch[-1]+0.02,"acc={:.3f}".format(train_acc_epoch[-1]),color='#d46b5f')
    plt.text(x[-1]-0.02,val_acc_epoch[-1]-0.02,"acc={:.3f}".format(val_acc_epoch[-1]),color='#50c3e6')
    plt.scatter([x[-1]], [train_acc_epoch[-1]], s=25, c='#d46b5f')
    plt.scatter([x[-1]], [val_acc_epoch[-1]], s=25, c='#50c3e6')
    plt.savefig(os.path.join(out_dir,"acc"))


#### For testing
if __name__ == "__main__":
    # plot_performance([1,2,3,4], [3,4,5,6], [4,5,6,7], [8,7,6,5], "./models")

    # y = [0,1,2,2,3,4,5,6]
    # y_hat = [0,1,1,2,3,4,5,6]
    # plot_confusion_matrix(y, y_hat, "./results", True)

    model = get_pca_model('./dataset/images/train')
    model.transform(np.ones((1,2304)))

