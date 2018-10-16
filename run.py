from __future__ import print_function
from os.path import join
import argparse
import torch.nn as nn
import torch
from os import listdir
import utils
import torchvision.transforms as transforms
from model import VGG
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpuids', default=[0,1], nargs='+', help='GPU ID for using')
opt = parser.parse_args()

opt.gpuids = list(map(int, opt.gpuids))

print("loading model...")
model_name = join("model", opt.model)
#model = torch.load(model_name)

model = VGG('VGG19')

if opt.cuda:
    model = nn.DataParallel(model, device_ids=opt.gpuids).cuda()

model.load_state_dict(torch.load(model_name))


run_dir = "dataset/run"
image_filename = [join(run_dir, x) for x in listdir(run_dir) if utils.is_image_file(x)]

with open(run_dir+'/new_attribute6.txt', 'r') as f:
    persons = [line.split('\t') for line in f.read().splitlines()]

f.close()



transform = transforms.Compose([
    transforms.CenterCrop(40),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

inputs = []
false = 0
true = 0

for i in range(0, len(image_filename)):
    input = utils.load_img(image_filename[i])
    inputs.append(transform(input).unsqueeze_(0))

f = open("answer.txt",'w')

for i in range(0, len(inputs)):
    if opt.cuda:
        inputs[i].cuda()
    temp = image_filename[i]
    temp = temp.split('/')
    temp = temp[2]
    temp = temp[:7]
    
    #output 찾는 과정.
    for person in persons:
        if(temp == person[0]):
            output = person[1]
            output = utils.match_one_hot_vector(output)
            break

    
    out = model(inputs[i])

    out = out.detach()
    _,out = torch.max(out,1)
    
    out = out.item()
    output = output.tolist()
    max_num = max(output)
    max_index = output.index(max_num)
    

    if(out == max_index):
        true = true+1
    else:
        false = false+1

print("true : {}        false : {}".format(true,false))
    
#correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(output,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(correct_prediction)

f.close()
