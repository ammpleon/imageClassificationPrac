from cnn import NeuralNet
import torch
import torch.nn as nn
from dataloaders import get_test_loader
from torchvision.datasets import CIFAR10
from transformation import transform_data


if __name__ == "__main__":


    cnn = NeuralNet()

    transform = transform_data(0.5, 0.5)


    test_data = CIFAR10(root = "data", train = False, transform = transform, download = True)
    cnn.load_state_dict(torch.load("trained_net.pth"))


    test_loader = get_test_loader(test_data, batch_size=64, num_workers=2)


    correct = 0
    total = 0
    cnn.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = cnn(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct/total)*100

    print(f"Accuracy: {accuracy}%")
