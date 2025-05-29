import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
from cnn import NeuralNet



if __name__ == "__main__":

    cnn = NeuralNet()

    cnn.load_state_dict(torch.load("trained_net.pth"))


    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


    new_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5), std = (0.5))
    ])


    images = ["test.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg"]


    def load_image(image_path):
        image = Image.open(image_path)
        image = new_transform(image)
        image = image.unsqueeze(0)
        return image


    loaded_images = [load_image(image) for image in images]

    cnn.eval()

    with torch.no_grad():
        for image in loaded_images:
            output = cnn(image)
            _, predicted = torch.max(output, 1)
            print(f"Predicition: {classes[predicted.item()]}")