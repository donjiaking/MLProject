import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os

def load_model(model, model_dir):
    return model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))

def save_model(model, model_dir):
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))


def plot_performance(loss_epoch, train_acc_epoch, val_acc_epoch, out_dir):
    if(not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    x = list(range(1, len(loss_epoch)+1))

    plt.plot(x, loss_epoch, color='#d46b5f')
    plt.xlabel('epoch')
    plt.ylabel("Loss")
    plt.title("Loss on the training set")
    plt.savefig(os.path.join(out_dir,"loss"))
    plt.clf()

    plt.plot(x, train_acc_epoch, color='#d46b5f', label="training set")
    plt.plot(x, val_acc_epoch, color='#50c3e6', label="validation set")
    plt.xlabel('epoch')
    plt.ylabel("Accuracy")
    plt.title("Accuracy on the training/validation set")
    plt.legend()
    plt.text(x[-1]-0.2,train_acc_epoch[-1]+0.2,"acc={:.3f}".format(train_acc_epoch[-1]),color='#d46b5f')
    plt.text(x[-1]-0.2,val_acc_epoch[-1]+0.2,"acc={:.3f}".format(val_acc_epoch[-1]),color='#50c3e6')
    plt.scatter([x[-1]], [train_acc_epoch[-1]], s=25, c='#d46b5f')
    plt.scatter([x[-1]], [val_acc_epoch[-1]], s=25, c='#50c3e6')
    plt.savefig(os.path.join(out_dir,"acc"))


#### For testing
if __name__ == "__main__":
    plot_performance([1,2,3,4], [4,5,6,7], [8,7,6,5], "./results")


