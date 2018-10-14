from __future__ import print_function
from os.path import join
import argparse
import torch
from os import listdir
import utils
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

model_name = join("model", opt.model)
model = torch.load(model_name)

run_dir = "dataset/run"
image_filename = [join(run_dir, x) for x in listdir(run_dir) if utils.is_image_file(x)]

with open(run_dir+'/new_attribute6.txt', 'r') as f:
    persons = [line.split('\t') for line in f.read().splitlines()]

f.close()


transform = transforms.Compose([
    transforms.CenterCrop(40),          #transforms.RandomCrop(32, padding=4)
    transforms.ToTensor(),
    transforms.Normalize((0.49, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

inputs = []
false = 0
true = 0

for i in range(0, len(image_filename)):
    input = utils.load_img(image_filename[i])
    inputs.append(transform(input))

f = open("answer.txt",'w')

for i in range(0, len(inputs)):
    if opt.cuda:
        model = model.cuda()
        inputs[i].cuda()

    #output 찾는 과정.
    for person in persons:
        if(image_filename[i] == person[0]):
            output = person[1]
            output = utils.match_one_hot_vector(output)
            break

    out = model(inputs[i])
    max_num = max(out)
    max_num = out.index(max_num)

    max_out = max(output)
    max_out = output.index(max_out)
    if(max_out == max_num):
        true = true + 1
    else:
        false = false + 1


print("true : {}".format(true))
print("false : {}".format(false))







