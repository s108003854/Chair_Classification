# Import Tensorflow & Keras
from tensorflow import keras
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model

import tensorflow as tf

# Helper libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import skimage
from skimage.transform import resize
import pandas as pd
import os

#Import model
from SE_Block import squeeze_excite_block
from ResNet import resnet_block
from SENet import create_se_resnet


# Import Lable
y=[];l=0
for i in os.listdir('./data'):
    for j in os.listdir('./data/'+i):
        y.append(l)
    l+=1

y=np.array(y)
y=y.reshape(y.shape[0],1)

# Import Image

#check max and min of Length, max and min of width
l_max=0;w_max=0;l_min=0;w_min=0
for i in os.listdir('./data'):
    for j in os.listdir('./data/'+i):
        path='./data/'+i+'/'+j
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        l_min=img.shape[0];w_min=img.shape[0]
        if(img.shape[0]>l_max):
            l_max=img.shape[0]
        if(img.shape[0]<l_min):
            l_min=img.shape[0]
        if(img.shape[1]>w_max):
            w_max=img.shape[1]
        if(img.shape[1]<w_min):
            w_min=img.shape[1]

X=[]
for i in os.listdir('./data'):
    for j in os.listdir('./data/'+i):
        path='./data/'+i+'/'+j
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #print(img.shape)
        img = skimage.transform.resize(img, (l_min, w_min))
        img = np.asarray(img)
        X.append(img)

X = np.asarray(X)
X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)

#split data to training and test
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

# Normalize pixel values to be between 0 and 1
train_images.astype('float32');test_images.astype('float32')
train_images, test_images = train_images / 255.0, test_images / 255.0

# Encoding
train_labels = to_categorical(np.array(train_labels[:, 0]))
test_labels = to_categorical(np.array(test_labels[:, 0]))

# Call Model
img_rows, img_cols = l_min, w_min
input_shape = (img_rows, img_cols, 1)
initial_conv_filters=64
depth=[2, 2, 2, 2]
filters=[64, 128, 256, 512]
width=1
weight_decay=1e-4
include_top=True
weights=None
input_tensor=None
pooling=None
classes=4

img_input = Input(shape=input_shape)
   
x = create_se_resnet(classes, img_input, include_top, initial_conv_filters,
                          filters, depth, width, weight_decay, pooling)

model = Model(img_input, x, name='resnext')
    
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, validation_split=0.25)

score = model.evaluate(test_images, test_labels, verbose=0)
print('accuracy: ',score[1])
print('loss: ',score[0])

model.save('model.h5')