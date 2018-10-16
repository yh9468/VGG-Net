import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from os import errno
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import utils
import os
from torch.autograd import Variable
import argparse
from utils import get_training_dataset, get_test_set
from model import VGG

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--save_path', type=str, default='model', help='model save path')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--gpuids', default=[0, 1, 2, 3], nargs='+', help='GPU ID for using')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
opt = parser.parse_args()

opt.gpuids = list(map(int,opt.gpuids))
print(opt)


use_cuda = opt.cuda
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


train_set = get_training_dataset(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)

training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=128, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=100, shuffle=False)


print("Building model...")
net = VGG('VGG19')
best_acc = 0

if use_cuda:
    torch.cuda.set_device(opt.gpuids[0])
    net = nn.DataParallel(net, device_ids=opt.gpuids, output_device=opt.gpuids[0]).cuda()

net_optim = optim.Adam(net.parameters(), lr=opt.lr)

criterion = nn.MSELoss()

def train(epoch):
    print('\n Epoch:',epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        net_optim.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        net_optim.step()

        train_loss += loss.item()

    print("===>Complete: Avg. Loss: {:.4f}".format(train_loss / len(training_data_loader)))


def test():
    global best_acc
    test_loss = 0

    for i, batch in enumerate(testing_data_loader):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        output = net(input)
        loss = criterion(output, target)

        test_loss += loss.item()

    print("===>Complete: Avg. Loss: {:.4f}".format(test_loss / len(testing_data_loader)))


def checkpoint(epoch):
    try:
        if not(os.path.isdir('model')):
            os.makedirs(os.path.join('model'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    model_out_path = "model/model_epoch_{}.pth".format(epoch)

    torch.save(net.state_dict(), model_out_path)
# torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1,opt.epochs +1):
    train(epoch)
    test()
    if(epoch %10 ==0):
        checkpoint(epoch)
