import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import re, os

import glob


def normalize_age(age, max_age=100.0):
    return 1/max_age * (age - max_age/2)

def invert_normalize_age(age, max_age=100.0):
    return age * max_age + max_age/2


prefix='filelists/AAF/'
train_people = glob.glob(prefix + 'train/*.jpg')
test_people = glob.glob(prefix + 'test/*.jpg')
def num_to_str(num):
    str_ = ''
    if num == 0:
        str_ = '000'
    elif num < 100:
        str_ = '0' + str(int(num))
    else:
        str_ = str(int(num))
    return str_

def get_person(person, prefix='filelists/AAF/train/', max_age=100.0):
    faces   = []
    targets = []

    train_transforms = transforms.Compose([transforms.ToTensor()])

    fname  = person
    img    = Image.open(fname).convert('RGB')
    img    = train_transforms(img)

    # normaliz the age to [-1, 1]
    age = normalize_age(float(re.search('A(.*)_gs.jpg', os.path.basename(person)).group(1)))
    targets = torch.Tensor([age])
    faces.append(img)

    faces = torch.stack(faces)
    #targets = torch.stack(targets).squeeze()
    return faces, targets


def get_batch(train_people=train_people):
    # read in the jpg files
    inputs  = []
    targets = []

    for person in train_people:
        inps, targs = get_person(person)
        inputs.append(inps)
        targets.append(targs)

    return torch.stack(inputs), torch.stack(targets)
