import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from cnn import NeuralNet
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from dataloaders import get_train_loader, get_test_loader
from transformation import transform_data
import time
import tqdm



transform = transform_data(0.5, 0.5)

device = torch.device("cuda")


train_data = CIFAR10(root = "data", train = True, transform = transform, download = True)


image, label = train_data[0]


conv = NeuralNet()
loss_function = CrossEntropyLoss()


def modelTraining(train_set, loss_function, model, lr = 0.001, momentum = 0.9, epochs = 20, batch_size = 64, num_workers = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    

    train_loader = get_train_loader(train_set, batch_size=batch_size, num_workers= num_workers)
    optimizer = SGD(model.parameters(), lr = lr, momentum=momentum)

    start_time = time.time()
    for epoch in range(epochs):

        print(f"Training epoch {epoch+1}...")
        
        running_loss = 0.0
        

        for i, data in tqdm.tqdm(enumerate(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            #backpropagation
            loss.backward()
            #take a step in SGD
            optimizer.step()

            running_loss += loss.item()

            current_loss = round(running_loss/len(train_loader),4)

        print(f"Loss: {current_loss}")
        if current_loss <= 0.1:
            break
    
    end_time = time.time()
    print(f"Total time of operation in seconds {end_time-start_time}")
    torch.save(model.state_dict(), "trained_net.pth")


modelTraining(train_data, loss_function=loss_function, model = conv, epochs=350, batch_size=64, lr = 0.001)

