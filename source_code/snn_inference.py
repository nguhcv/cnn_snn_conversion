from torch import nn
import torch
from source_code.nets import Diehl_2015,SNN_Huynh2021
from source_code.utils import tran_test_split,proposed_method
import argparse

def app(opt):
    import time
    train_loader, test_loader = tran_test_split(dataset=opt.dataset, data_path=opt.data_path,batch_size=opt.batch_size)

    ann = Diehl_2015()
    ann.load_state_dict(torch.load(opt.save)['model'])
    ann.eval()
    if torch.cuda.is_available():
        ann.cuda()

    snn = SNN_Huynh2021(batch_size=opt.batch_size)

    saved_state_dict = torch.load(opt.save)
    print(saved_state_dict['epoch'])
    print(saved_state_dict['best_acc'])
    for index, m in enumerate(saved_state_dict['model']):
        snn.state_dict()[m].copy_(saved_state_dict['model'][m])

    snn.eval()
    snn.cuda()
    threshold = proposed_method(train_loader=train_loader, model=snn, num_steps=opt.num_steps)
    snn.update_threshold(threshold)

    total_images = 0
    num_corrects = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            snn.init_neuron_models()
            for t in range(opt.num_steps):
                outs = snn(images, encode=True)

            num_corrects += torch.argmax(outs, dim=1).eq(labels).sum(dim=0)
            total_images += images.size(0)

            print(num_corrects)
            print(total_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_steps', default=64, type=int)
    parser.add_argument('--save', default='model.pt')
    app(parser.parse_args())










