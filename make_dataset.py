import csv
import torch.utils.data as data
import os.path
import glob
import torch

w = open("new_attribute5.txt",'w')

with open('dataset/txt/attribute_5.txt', 'r') as f:
    data = [line.split(',') for line in f.read().splitlines()]
    print(data)
    sorted(data, key=lambda people: people[0])


for people in data:
    try:
        w.write("{}\t{}\n".format(people[0], people[2]))
    except IndexError:
        continue



