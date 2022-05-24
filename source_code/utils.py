import torch
from torchvision import transforms
import torchvision
from typing import Any
import torch.nn as nn



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

def cnn_training(train_loader, model, criterion, optimizer):
    model.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += float(torch.argmax(outputs, dim=1).eq(labels).sum())
        total_loss += float(loss)
        total_images += images.size(0)

        # if (step + 1) % 100 == 0:
        #     print("step: {} - loss: {} - acc: {}".format(step + 1, total_loss / total_images, num_corrects / total_images))

def cnn_testing(val_loader, model, criterion):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)

            loss = criterion(outputs, labels)

            num_corrects += float(torch.argmax(outputs, dim=1).eq(labels).sum())
            total_loss += float(loss)
            total_images += images.size(0)

    val_loss = total_loss / total_images
    val_acc = num_corrects / total_images

    return val_loss, val_acc


def proposed_method(train_loader: Any,
              model,
              num_steps: int,
              scaling_factor: float=1.):
    """
    This function implements our proposed threshold-balancing technique algorithm that finds the proper thresholds
    for ANN-SNN conversion. The function assumes that the input model is a SNN.
    """
    ths = []  # The number of learnable layers
    for m in model.named_children():
        if isinstance(m[1], nn.Conv2d):
            num_channels = m[1].out_channels
            ths.append([0.0]*num_channels)
        elif isinstance(m[1], nn.Linear):
            ths.append(0.0)

    #remove the last layer value
    ths = ths[:-1]

    print(model.conv1_if.threshold)
    for l in range(len(ths)):
        print(l)
        for it, (images, _) in enumerate(train_loader):
            model.init_neuron_models()
            with torch.no_grad():
                for t in range(num_steps):
                    p = 0
                    if torch.cuda.is_available():
                        x = images.cuda()
                    for m in model.named_children():
                        # Assume that snn is a sequential model
                        x = m[1](x)

                        if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear):
                            if p == l:
                                if isinstance(m[1], nn.Conv2d):
                                    sorted_matrix = torch.amax(x, 0)
                                    for i in range(len(ths[l])):
                                        ths[l][i] = max(ths[l][i], sorted_matrix[i].max())
                                    break
                                if isinstance(m[1], nn.Linear):
                                    ths[l] = max(ths[l], x.max())
                                    break
                            else:  # p < l
                                p += 1

        model.update_threshold(ths)

    return ths
