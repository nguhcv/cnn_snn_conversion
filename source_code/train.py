from torch import nn
import torch
from source_code.nets import Diehl_2015
from source_code.utils import tran_test_split,train,test
import argparse

def app(opt):
    import time
    train_loader, test_loader = tran_test_split(dataset=opt.dataset, data_path=opt.data_path,batch_size=opt.batch_size)

    # define network
    model = Diehl_2015()
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    best_epoch = -1
    best_acc = 0

    for epoch in range(opt.num_epochs):
        print("epoch: {}".format(epoch))
        train(train_loader, model, criterion, optimizer)

        loss, acc = test(test_loader, model, criterion)
        print("In test, loss: {} - acc: {}".format(loss, acc))

        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            state_dict = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': model.state_dict(),
            }
            torch.save(state_dict,'model.pt')

        print()

    print("epoch: {} - best acc: {}".format(best_epoch, best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--save', default='saved_model/model.pt')

    app(parser.parse_args())



