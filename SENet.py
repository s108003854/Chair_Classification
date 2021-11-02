from tensorflow import keras
from keras.models import Model
from keras.layers import *
from keras.utils import conv_utils
from keras.engine.topology import get_source_inputs
from ResNet import resnet_block

def create_se_resnet(classes, img_input, include_top, initial_conv_filters, filters,
                      depth, width, weight_decay, pooling):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    for i in range(N[0]):
            x = resnet_block(x, filters[0], width)

    for k in range(1, len(N)):
            x = resnet_block(x, filters[k], width, strides=(2, 2))

    for i in range(N[k] - 1):
            x = resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='softmax')(x)
    
    return x