from glob import glob
files = glob("*.jpg")

# resize and downsample color


import numpy as np
import matplotlib.pylab as plt

from skimage.color import rgb2gray  # using luminance
from skimage.io import imread, imsave
from skimage import transform
import os

for f in files:

    if '_gs' not in f:
        img = imread(f)
        img = rgb2gray(img)
        # set desired image size
        out_size = (80, 80)  # height, width
        # set the color of the padded area. Here: "95% luminance"

        resized_img = transform.resize(img, out_size)


        imsave(os.path.splitext(f)[0] + '_gs.jpg', resized_img)

