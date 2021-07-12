# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:13:39 2021

@author: sheri
"""

#------------------------------------------------------------------------------
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#------------------------------------------------------------------------------
# Make dataset that contains 200 images
import os
import shutil
import random

os.makedirs('training/training200')

root_path200 = 'training/training200'
folders = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
for folder in folders:
    os.mkdir(os.path.join(root_path200,folder))


    
for i in range(10):
    src = 'training/training/n' + str(i)
    src_files = random.sample(os.listdir(src), 20)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        shutil.copy(full_file_name, 'training/training200/n' + str(i))

#------------------------------------------------------------------------------
# Make dataset that contains 200 images with 300 more augmented images
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest')


os.makedirs('training/training500')   

from distutils.dir_util import copy_tree
fromDirectory = "training/training200"
toDirectory = "training/training500"
copy_tree(fromDirectory, toDirectory)


for i in range(10):
    src = 'training/training500/n' + str(i)
    src_files = random.sample(os.listdir(src), 1)
    imglink = src_files[0]
    imglinkfull = src + '/' + imglink
    augimg = load_img(imglinkfull)
    x = img_to_array(augimg)  
    x = x.reshape((1,) + x.shape)
    
    j = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='training/training500/n' + str(i), 
                          save_prefix='img', 
                          save_format='jpeg'):
        j += 1
        if j >= 30:
            break  # break the loop, otherwise it will never 
#------------------------------------------------------------------------------
# 1000 more on original data augmentation for all so that there are 2000+ images
os.makedirs('training/trainin2000')   

from distutils.dir_util import copy_tree
fromDirectory = "training/training"
toDirectory = "training/training2000"
copy_tree(fromDirectory, toDirectory)


for i in range(10):
    src = 'training/training2000/n' + str(i)
    for each in os.listdir(src):
        imglink = each
        imglinkfull = src + '/' + imglink
        augimg = load_img(imglinkfull)
        x = img_to_array(augimg)  
        x = x.reshape((1,) + x.shape)
    
        j = 0
        for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='training/training2000/n' + str(i), 
                          save_prefix='img', 
                          save_format='jpeg'):
            j += 1
            if j >= 1:
                break  # break the loop, otherwise it will never 

