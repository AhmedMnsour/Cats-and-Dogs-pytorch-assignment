from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset import CatDogDataset
from model import CNN_1
from model import CNN_2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim


def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


input_size = 224 * 244 * 3  
output_size = 2  



mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize((224, 224)), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])



train_path    = 'dataset/train'
test_path = 'dataset/test'
train_dataset = CatDogDataset(train_path, transform=transform)
test_data = CatDogDataset(test_path, transform=transform)


shuffle     = True
batch_size  = 64
num_workers = 0

trainLoader  = DataLoader(dataset=train_dataset, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)

testLoader  = DataLoader(dataset=test_data, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)





accuracy_list = []

CNN_2_model = CNN_2()
CNN_1_model = CNN_1()

optimizer = optim.SGD(CNN_2_model.parameters(), lr=0.01, momentum=0.5)
#optimizer = optim.SGD(CNN_1_model.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(CNN_1_model)))



def train(epoch, model):
    for batch_idx, (image, target) in enumerate(trainLoader):
        optimizer.zero_grad()
        output = model(image)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(trainLoader.dataset),
                       100. * batch_idx / len(trainLoader), loss.item()))


def test(model):

    model.eval()
    test_loss = 0
    correct = 0

    for data, target in testLoader:
        output = model(data)

        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(testLoader.dataset)
    accuracy = 100. * correct / len(testLoader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testLoader.dataset),
        accuracy))

if __name__ == "__main__":
    for epoch in range(0, 1):

        train(epoch, CNN_2_model)
        test(CNN_2_model)

        #train(epoch, CNN_1_model)
        #test(CNN_1_model)