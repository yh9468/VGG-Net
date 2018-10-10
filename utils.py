
import torch.utils.data as data
import torch
import sys
import time
import os
from os import listdir
from os.path import join
from PIL import Image, ImageFilter
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop, Lambda
import PIL
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in ["txt"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    return img



class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, txt_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.txt_filenames = [join(txt_dir, x) for x in listdir(txt_dir) if is_txt_file(x)]

        for filename in self.txt_filenames:
            with open(filename, 'r') as f:
                self.txt_data = [(line.split('\t'))[1] for line in f.read().splitlines()]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = RandomCrop(300)(load_img(self.image_filenames[index]))
        target = match_one_hot_vector(self.txt_data[index])

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform(crop_size, upscale_factor):
    return Compose([
        transforms.RandomCrop(crop_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def target_transform(crop_size):
    return Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def get_training_dataset(upscale_factor):
    train_dir = ""
    crop_size = calculate_valid_crop_size(224, upscale_factor)

    return DatasetFromFolder(train_dir, input_transform= input_transform(crop_size, upscale_factor))


def get_test_set(upscale_factor):
    test_dir = ''
    crop_size = calculate_valid_crop_size(224, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor))

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.wceight, 1)
            init.constant(m.wceight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def match_one_hot_vector(str):
    classes = ["Criminal Damage", "Probation Violation", "Liquor Violation", "Failure to Appear", "Forgery",
           "Fraud", "Robbery", "Crimes Against Children", "Sex Crimes", "DUI", "Courts and Civil Proceedings Violations",
           "Obstruction", "Assault", "Offenses against Public Order", "Transportation Violations",
           "Homicide", "Drug Offenses", "Kidnapping", "Weapons and Explosives", "Profession and Occupation Violations",
           "County Regulations Violations", "Family Offenses", "Criminal Trespass and Burglary", "ANIMAL CRUELTY",
           "Eavesdropping and Communication", "Interfere with Judicial Process", "Theft", "Other"]

    for i in range(0, len(classes)):
        if(str == classes[i]):
            index = i

    values = array(classes)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded[index]





