import torch
from torchvision import transforms
import torchvision


def tran_test_split(dataset:str, data_path:str, batch_size:int):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if dataset=='MNIST':
        trainset = torchvision.datasets.MNIST(root=data_path, train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, )
        testset = torchvision.datasets.MNIST(root=data_path, train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True)
        return trainloader, testloader
