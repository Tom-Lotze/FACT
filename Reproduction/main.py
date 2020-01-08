# -*- coding: utf-8 -*-
# python 3
# @Authors: Tom Lotze, Berend Jansen
# @Date:   2020-01-08 12:16:10
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-01-08 13:07:30

# imports
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# import helper functions
from autoenc_helpers import makedirs, list_of_distances, print_and_write, list_of_norms
from data_preprocessing import batch_elastic_transform

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
filter_shape_1 = [f_1, f_1, n_input_channel, n_map_1]
filter_shape_2 = [f_2, f_2, n_map_1, n_map_2]
filter_shape_3 = [f_3, f_3, n_map_2, n_map_3]
filter_shape_4 = [f_4, f_4, n_map_3, n_map_4]

# strides for each layer
stride_1 = [1, s_1, s_1, 1]
stride_2 = [1, s_2, s_2, 1]
stride_3 = [1, s_3, s_3, 1]
stride_4 = [1, s_4, s_4, 1]



######## CREATE TRAINING EXAMPLES AND LABELS #########





######## INITIALIZE THE ENCODER AND DECODER #########





######## IMPORT DATA #########












