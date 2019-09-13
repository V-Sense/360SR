#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:52:27 2019

@author: aakanksha
"""
import numpy as np
from ws_ssim import ws_ssim
from ws_psnr import ws_psnr
import os
from glob import glob
from PIL import Image
from scipy.ndimage import imread
from scipy.misc import imresize

# folder that has original images
inputfolder = '/media/nas/DL_compression/_input/SRLearning/trans_Equi/test/'
# result folder for the bicubic method
testfolder = '/home/aakanksha/SRGAN/results/Uncompressed/8-bi-nn/bicubic/'
# result folder for the NN method
testfolder2 = '/home/aakanksha/SRGAN/results/Uncompressed/8-bi-nn/nn/'
files_list = glob(os.path.join(inputfolder, '*.png'))

results1 = []
results2 = []
for a_file in sorted(files_list):
    path, IMAGE_NAME = os.path.split(a_file) 
    im1 = path+'/' +IMAGE_NAME
    im2 = testfolder+'/' +IMAGE_NAME
    im3 = testfolder2+'/' +IMAGE_NAME
    image1 = imread(im1)                   #       numpy.asarray(Image.open(path+'/' +IMAGE_NAME))
    image2 = imread(im2)                  #       numpy.asarray(Image.open(testfolder+'/' +IMAGE_NAME))
    image3 = imread(im3) 
    _, _, c = image1.shape
    lch1 = []
    lch2 = []
    for _c in range(c):
        lch1.append(ws_ssim(image1[:,:,_c], image2[:,:,_c]))
        lch2.append(ws_ssim(image1[:,:,_c], image3[:,:,_c]))
    fin_score1 = np.mean(lch1)
    fin_score2 = np.mean(lch2)
    results1.append(fin_score1)
    results2.append(fin_score2)
    