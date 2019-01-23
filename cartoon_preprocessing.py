# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 10:20:03 2018
"""

import numpy as np
import scipy.misc
from os import listdir
from os.path import isfile, join
import os
import tqdm
imsize = 64
path = '/home/shf/DataCenterS2/cartoonset100k_all/'
path_output = '/home/shf/datasets/cartoonset100k_small'
if(os.path.isdir(path_output)==False):
    os.mkdir(path_output)
files = [f for f in listdir(path) if isfile(join(path, f))]
image_count=len(files)
if(image_count==0):
    raise Exception("no image found in "+path)
print("image count = "+str(image_count))

for n in tqdm.trange(image_count):
    f_name = files[n]
    image = scipy.misc.imread(path+f_name)
    image = image[122:389,130:415,:3]#if RGBA, convert to RGB
    if len(image) != imsize:
        image = scipy.misc.imresize(image,[imsize,imsize],interp='bicubic')
        scipy.misc.imsave(path_output+'/'+f_name,image)
        
