# -*- coding: utf-8 -*-
# python 3
# @Authors: Tom Lotze, Berend Jansen
# @Date:   2020-01-08 12:16:10
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-01-08 17:16:17

# imports
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# import helper functions
from autoenc_helpers import makedirs, list_of_distances, print_and_write, list_of_norms
# from data_preprocessing import batch_elastic_transform

######## IMPORT DATA #########

# create data folder
makedirs("MNIST_data")


######## SET PARAMETERS #########

# COPIED FROM THE ORIGINAL IMPLEMENTATION
# training parameters
learning_rate = 0.002
training_epochs = 1500
batch_size = 250          # the size of a minibatch

# frequency of testing and saving
test_display_step = 100   # how many epochs we do evaluate on the test set once
save_step = 50            # how frequently do we save the model to disk

# elastic deformation parameters
sigma = 4
alpha = 20

# lambda's are the ratios between the four error terms
lambda_class = 20
lambda_ae = 1 # autoencoder
lambda_1 = 1 # push prototype vectors to have meaningful decodings in pixel space
lambda_2 = 1 # cluster training examples around prototypes in latent space


input_height = input_width =  28    # MNIST data input shape 
n_input_channel = 1     # the number of color channels; for MNIST is 1.
input_size = input_height * input_width * n_input_channel   # 784
n_classes = 10

# Network Parameters
n_prototypes = 15         # the number of prototypes
n_layers = 4

# height and width of each layers' filters
f_1 = 3
f_2 = 3
f_3 = 3
f_4 = 3

# stride size in each direction for each of the layers
s_1 = 2
s_2 = 2
s_3 = 2
s_4 = 2

# number of feature maps in each layer
n_map_1 = 32
n_map_2 = 32
n_map_3 = 32
n_map_4 = 10

# the shapes of each layer's filter

# [out channel, in_channel, 3, 3

filter_shape_1 = [n_map_1, n_input_channel, f_1, f_1]
filter_shape_2 = [n_map_2, n_map_1, f_2, f_2]
filter_shape_3 = [n_map_3, n_map_2, f_3, f_3]
filter_shape_4 = [n_map_4, n_map_3, f_4, f_4]

# strides for each layer (changed to tuples)
stride_1 = [s_1, s_1]
stride_2 = [s_2, s_2]
stride_3 = [s_3, s_3]
stride_4 = [s_4, s_4]

print(type(stride_1))



######## CREATE TRAINING EXAMPLES AND LABELS #########





######## INITIALIZE THE ENCODER AND DECODER #########
    
std_weights = 0.01

weights = {
    'enc_f1': nn.Parameter(std_weights * torch.randn(filter_shape_1,
                                           dtype=torch.float32)),
    'enc_f2': nn.Parameter(std_weights * torch.randn(filter_shape_2,
                                           dtype=torch.float32)), 
    'enc_f3': nn.Parameter(std_weights * torch.randn(filter_shape_3,
                                           dtype=torch.float32)), 
    'enc_f4': nn.Parameter(std_weights * torch.randn(filter_shape_4,
                                           dtype=torch.float32)), 
    'dec_f4': nn.Parameter(std_weights * torch.randn(filter_shape_4,
                                           dtype=torch.float32)), 
    'dec_f3': nn.Parameter(std_weights * torch.randn(filter_shape_3,
                                           dtype=torch.float32)), 
    'dec_f2': nn.Parameter(std_weights * torch.randn(filter_shape_2,
                                           dtype=torch.float32)),
    'dec_f1': nn.Parameter(std_weights * torch.randn(filter_shape_1,
                                           dtype=torch.float32)),
}
print("weights")
for weight in weights.keys():
    print(weight, weights[weight].shape)

biases = {
    'enc_b1': nn.Parameter(torch.zeros([n_map_1], dtype=torch.float32)),
    'enc_b2': nn.Parameter(torch.zeros([n_map_2], dtype=torch.float32)),
    'enc_b3': nn.Parameter(torch.zeros([n_map_3], dtype=torch.float32)),
    'enc_b4': nn.Parameter(torch.zeros([n_map_4], dtype=torch.float32)),
    'dec_b4': nn.Parameter(torch.zeros([n_map_3], dtype=torch.float32)),
    'dec_b3': nn.Parameter(torch.zeros([n_map_2], dtype=torch.float32)),
    'dec_b2': nn.Parameter(torch.zeros([n_map_1], dtype=torch.float32)),
    'dec_b1': nn.Parameter(torch.zeros([n_input_channel], dtype=torch.float32)),
}

print("biases")
for b in biases.keys():
    print(b, biases[b].shape)

last_layer = {
    'w': nn.Parameter(torch.randn([n_prototypes, n_classes],
                                       dtype=torch.float32))
}

print("last_layer")
print(last_layer['w'].shape)




######## FUNCTIONS FOR LAYERS #########

# padding can be either "SAME" or "VALID"
def conv_layer(input, filter, bias, strides, padding="VALID",
               nonlinearity = nn.ReLU()):
    conv = F.conv2d(input, filter, bias=bias, stride=strides,
       padding=padding)
    out = nonlinearity(conv)
    return out
#### STRIDE MUST BE TUPLE FOR TORCH, IS A LIST IN TENSORFLOW
#### PADDING IS DIFFERENT, TF USES SAME/VALID, TORCH A INT OR LIST OF INTS
### IS THE FILTER THE SAME AS WEIGHTS ARGUMENT FOR THE CONV2D?

# tensorflow's conv2d_transpose needs to know the shape of the output
def deconv_layer(input, filter, bias, strides, padding="VALID",
                 nonlinearity=nn.ReLU()):
    deconv = F.conv_transpose2d(input, filter, bias=bias, stride=strides,
                                padding=padding)
    out = nonlinearity(deconv)
    return out

def fc_layer(input, weight, bias, nonlinearity=nn.ReLU()):
    return nonlinearity(torch.mm(input, weight) + bias)


######## CONSTRUCT THE MODEL #########

# create X

X = torch.empty(batch_size, n_input_channel, input_width, input_height)

PADDING_FLAG = 1
# eln means the output of the nth layer of the encoder
el1 = conv_layer(X, weights['enc_f1'], biases['enc_b1'], stride_1, PADDING_FLAG)
el2 = conv_layer(el1, weights['enc_f2'], biases['enc_b2'], stride_2, PADDING_FLAG)
el3 = conv_layer(el2, weights['enc_f3'], biases['enc_b3'], stride_3, PADDING_FLAG)
el4 = conv_layer(el3, weights['enc_f4'], biases['enc_b4'], stride_4, PADDING_FLAG)

# we compute the output shape of each layer because the deconv_layer function
# requires it
# I THINK THIS WOULD NOT BE NECESSARY IN TORCH (THIS ARGUMENT NOT NEEDED)
# l1_shape = el1.get_shape().as_list()
# l2_shape = el2.get_shape().as_list()
# l3_shape = el3.get_shape().as_list()
# l4_shape = el4.get_shape().as_list()

l4_shape = el4.shape
print("l4_shape", l4_shape)

flatten_size = l4_shape[1] * l4_shape[2] * l4_shape[3]
n_features = flatten_size

# feature vectors is the flattened output of the encoder
feature_vectors = torch.reshape(el4, shape=[-1, flatten_size])

# the list prototype feature vectors
prototype_feature_vectors = nn.Parameter(torch.empty(size=
                                        [n_prototypes, n_features],
                                        dtype=torch.float32).uniform_())

print(prototype_feature_vectors)










