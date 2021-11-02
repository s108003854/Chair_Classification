from tensorflow import keras
from keras.models import Model
from keras.layers import *
from keras.utils import conv_utils
from keras.engine.topology import get_source_inputs


def squeeze_excite_block(input, ratio=16):
    
    init = input
    filters = init._keras_shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = multiply([init, se])
    
    return x