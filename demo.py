from torchvision import models, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from net import *
import util
import cv2

"""
A simple demo for predicting expression given an image
"""
def what_is_expression(img_dir, model_name):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)
    util.load_model(model, './models', model_name)

    data_transform=transforms.Compose([  
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = data_transform(img)
    img = torch.stack([img], dim=0)
    with torch.no_grad():
        output = model(img)
    output = F.softmax(output, dim=1)

    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plt.bar(range(7), list(output.numpy()[0]), tick_label=labels)
    plt.title("What is you expression???")
    plt.xlabel("Expressions")
    plt.ylabel("Probability")
    plt.show()


if __name__ == "__main__":
    what_is_expression("./dataset/demo_test2.jpg", "resnet18")