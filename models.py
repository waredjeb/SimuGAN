import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, LSTM, Permute, Reshape, Masking, TimeDistributed, MaxPooling1D, Flatten, Bidirectional
from keras.layers.merge import *
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import concatenate, maximum, dot, average, add, subtract
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers import Conv1D, GlobalMaxPooling1D, Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D
from keras.layers.merge import *
from keras.optimizers import *
from keras.regularizers import *
from keras.models import load_model

import tensorflow as tf
import numpy as np

def generator_model(input_size, output_size, alpha = 0.2):

    gen_in = Input(shape =(input_size,), name ="Input-layer_G")

    gen = Dense(128, kernel_initializer = 'glorot_normal')(gen_in)
    gen = LeakyReLU(alpha = alpha)(gen) #alpha = 0.2
    gen = BatchNormalization()(gen)

    gen = Dense(64)(gen)
    gen = LeakyReLU(alpha = alpha)(gen)
    gen = BatchNormalization()(gen)

    gen = Dense(32)(gen)
    gen = LeakyReLU(alpha = alpha)(gen)

    gen_out = Dense(output_size,activation = "tanh")(gen)

    generator = Model(gen_in,gen_out)

    return generator

def discriminator_model(output_size, alpha = 0.2, momentum = 0.99):

    dis_in = Input(shape = (output_size,), name = "Input-layer_D")

    dis = Dense(128)(dis_in)
    dis = LeakyReLU(alpha = alpha)(dis)
    dis = BatchNormalization(momentum = momentum)(dis)

    dis = Dense(64)(dis)
    dis = LeakyReLU(alpha = alpha)(dis)
    dis = BatchNormalization(momentum = momentum)(dis)

    dis_out = Dense(1,activation = "sigmoid")(dis)

    discriminator = Model(dis_in,dis_out)

    return discriminator


def generator_conv(input_size, output_size, alpha = 0.2):

    conv_in = Input(shape =(input_size,), name = 'Conv-input')
    
    dense = Dense(128, kernel_initializer = 'glorot_uniform')(conv_in)
    dense = LeakyReLU(alpha = alpha)(dense)
    dense = BatchNormalization()(dense)

    dense = Reshape([8,8,2])(dense)

    #conv = Conv2DTranspose(64, kernel_size = 2, strides = 1, padding = 'same')(dense)
    #conv = LeakyReLU(alpha = alpha)(conv)
    #conv = BatchNormalization()(conv)

    conv = Conv2DTranspose(32, kernel_size = 2, strides = 1, padding = 'same')(dense)
    conv = LeakyReLU(alpha = alpha)(conv)
    conv = BatchNormalization()(conv)

    conv = Conv2DTranspose(16, kernel_size = 3, strides = 1, padding = 'same')(conv)
    conv = LeakyReLU(alpha = alpha)(conv)
    conv = BatchNormalization()(conv)

    conv = Conv2DTranspose(8, kernel_size = 3, strides = 1, padding = 'same')(conv)
    conv = LeakyReLU(alpha = alpha)(conv)
    conv = BatchNormalization()(conv)

    conv = Conv2DTranspose(4, kernel_size = 3, strides = 1, padding = 'same')(conv)
    conv = LeakyReLU(alpha = alpha)(conv)
    conv = BatchNormalization()(conv)

    #Output
    conv = Flatten()(conv)
    conv_out = Dense(output_size)(conv)
    conv_out = Activation("tanh")(conv_out)

    generator = Model(conv_in,conv_out)

    return generator

def discriminator_conv(output_size,alpha = 0.2,dropout = 0):

    conv_in = Input(shape=(output_size,))

    dense = Dense(128)(conv_in)
    dense = Reshape((8,8,2))(dense)

    conv = Conv2D(64,kernel_size = 3, strides = 1, padding ='same')(dense)
    conv = LeakyReLU(alpha = alpha)(conv)

    conv = Conv2D(32,kernel_size = 3, strides = 1, padding ='same')(dense)
    conv = LeakyReLU(alpha = alpha)(conv)

    conv = Conv2D(16, kernel_size = 3, strides = 1, padding = 'same')(conv)
    conv = LeakyReLU(alpha = alpha)(conv)

    #output

    conv = Flatten()(conv)
    conv = LeakyReLU(alpha = alpha)(conv)

    if(dropout > 0):
        conv = Dropout(dropout)(conv)

    conv_out = Dense(1,activation = 'sigmoid')(conv)


    discriminator = Model(conv_in,conv_out)

    return discriminator
