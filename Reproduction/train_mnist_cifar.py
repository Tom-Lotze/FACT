#!/usr/bin/env python
# coding: utf-8

from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from math import ceil
import pickle
from PIL import Image as PILImage
from skimage.color import rgb2gray
from google_drive_downloader import GoogleDriveDownloader as gdd


# CHOOSE DATASET

# Uncomment one of the assignments below
# WHICH_DATA_FLAG = "mnist_original"
WHICH_DATA_FLAG = "cifar"
# WHICH_DATA_FLAG = "mnist_color"
# WHICH_DATA_FLAG = "mnist_rgb2gray"


# Helper functions

# borrowed from the original paper of Li et al. (2018)
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def list_of_norms(X):
    return torch.sum(torch.pow(X, 2), dim=1)


# Create necessary directories

# data folder
data_folder = os.path.join(os.getcwd(), "data")
makedirs(data_folder)

# Download datasets
gdd.download_file_from_google_drive(
    file_id='1fgjFKJ1_2VaPbKROeH-aZJiX52eetw9w',
    dest_path='./data/github_data.zip',
    unzip=True, showsize=True)

# Parameters dependent on dataset, no input needed
batch_size = 250
n_input_channels = 1
image_size = 28
if WHICH_DATA_FLAG == "mnist_original":
    model_folder = os.path.join(os.getcwd(), "saved_model",
                                "mnist_model_standard")
    # model_folder = model_original
    model_filename = "mnist_original"
elif WHICH_DATA_FLAG == "mnist_color":
    model_folder = os.path.join(os.getcwd(), "saved_model",
                                "mnist_model_color28_")
    # model_folder = model_mnist_color
    model_filename = "mnist_color"
    n_input_channels = 3
elif WHICH_DATA_FLAG == "mnist_rgb2gray":
    model_folder = os.path.join(os.getcwd(), "saved_model",
                                "gray_mnist_model_color28_")
    # model_folder = model_mnist_rgb2gray
    model_filename = "mnist_rgb2gray"
elif WHICH_DATA_FLAG == "cifar":
    model_folder = os.path.join(os.getcwd(), "saved_model", "cifar")
    # model_folder = model_cifar
    model_filename = "cifar"
    n_input_channels = 3
    image_size = 32


# Helper functions to load cifar (borrowed from Deep Learning Assignment)
# all cifar10 code below is borrowed from the UvA Deep Learning course
# assignments

def load_cifar10_batch(batch_filename):
    with open(batch_filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        X = batch['data']
        Y = batch['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(
            0, 2, 3, 1).astype(np.float32)
        Y = np.array(Y)
        return X, Y


def load_cifar10(cifar10_folder):
    Xs = []
    Ys = []
    for b in range(1, 6):
        batch_filename = os.path.join(cifar10_folder, 'data_batch_' + str(b))
        X, Y = load_cifar10_batch(batch_filename)
        Xs.append(X)
        Ys.append(Y)
    X_train = np.concatenate(Xs)
    Y_train = np.concatenate(Ys)
    X_test, Y_test = load_cifar10_batch(
        os.path.join(cifar10_folder, 'test_batch'))
    return X_train, Y_train, X_test, Y_test


def get_cifar10_raw_data(data_dir):
    X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)
    return X_train, Y_train, X_test, Y_test


def preprocess_cifar10_data(X_train_raw, Y_train_raw, X_test_raw, Y_test_raw):
    X_train = X_train_raw.copy()
    Y_train = Y_train_raw.copy()
    X_test = X_test_raw.copy()
    Y_test = Y_test_raw.copy()

    # Transpose
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    return X_train, Y_train, X_test, Y_test


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            "images.shape: {0}, labels.shape: {1}".format(str(images.shape),
                                                          str(labels.shape)))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(data_dir, one_hot=True, validation_size=5000):
    # Extract CIFAR10 data
    train_images_raw, train_labels_raw, test_images_raw, test_labels_raw = get_cifar10_raw_data(
        data_dir)
    train_images, train_labels, test_images, test_labels = preprocess_cifar10_data(
        train_images_raw, train_labels_raw, test_images_raw, test_labels_raw)

    # Apply one-hot encoding if specified
    if one_hot:
        num_classes = len(np.unique(train_labels))
        train_labels = dense_to_one_hot(train_labels, num_classes)
        test_labels = dense_to_one_hot(test_labels, num_classes)

    # Subsample the validation set from the train set
    if not 0 <= validation_size <= len(train_images):
        raise ValueError("Validation size should be between 0 and {0}.",
                         "Received: {1}.".format(
                             len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    # Create datasets
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return {'train': train, 'validation': validation, 'test': test}


# changed data_dir = cifar_folder -> data_dir
def get_cifar10(data_dir, one_hot=True, validation_size=5000):
    return read_data_sets(data_dir, one_hot, validation_size)


# Function to transform original MNIST to colored MNIST or rgb2gray MNIST
def color_dataset(raw_data, to_gray=False):
    N = len(raw_data)
    if to_gray:
        n_channels = 1
    else:
        n_channels = 3

    raw_data = raw_data.view(N, 28, 28, 1)

    try:
        lena = PILImage.open('./resources/lena.png')
    except:
        print("Lena image could not be found, please check",
              "./resources/lena.png")
        return 1

    # Extend to RGB
    data_rgb = np.concatenate([raw_data, raw_data, raw_data], axis=3)

    # Make binary
    data_binary = (data_rgb > 0.5)
    data_color = np.zeros((N, 28, 28, n_channels))

    for i in range(N):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 28)
        y_c = np.random.randint(0, lena.size[1] - 28)
        image = lena.crop((x_c, y_c, x_c + 28, y_c + 28))
        # / 255.0 REMOVED DIVISION HERE TO MAKE EVERY DATASET EQUAL
        image = np.asarray(image)

        # COPIED IMAGE BECAUSE "READ-ONLY" ERROR
        new_image = image.copy()

        for j in range(3):
            new_image[:, :, j] = (new_image[:, :, j] +
                                  np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number
        new_image[data_binary[i]] = 1 - new_image[data_binary[i]]
        if to_gray:
            data_color[i] = np.reshape(rgb2gray(new_image), (28, 28, 1))
        else:
            data_color[i] = new_image

    return torch.from_numpy(data_color)


#  Load the datasets
''' The function below describes how the data can be generated. This function
is not used in this script. The data used in this script is stored on google
drive for reproducibility purposes. When generating the data yourself, every
dataset will differ slightly as a result of the randomly generated backgrounds
and colors for MNIST color. No random seed has been set.'''


def get_data(data_flag):
    # check if flag is correct
    assert data_flag in ["mnist_original",
                         "cifar", "mnist_color", "mnist_rgb2gray"]

    if data_flag == "cifar":
        # extract cifar data
        data = get_cifar10(os.path.join(data_folder, "cifar"), one_hot=False)

        # extract training,validation and test data
        X_train, Y_train = torch.from_numpy(
            data['train'].images), torch.from_numpy(data['train'].labels)
        X_validation, Y_validation = torch.from_numpy(
            data['validation'].images), torch.from_numpy(
                data['validation'].labels)
        X_test, Y_test = torch.from_numpy(
            data['test'].images), torch.from_numpy(data['test'].labels)

        # create datasets
        train_data = TensorDataset(X_train, Y_train)
        valid_data = TensorDataset(X_validation, Y_validation)
        test_data = TensorDataset(X_test, Y_test)
        return train_data, valid_data, test_data

    # Load datasets into reproduction/data/mnist
    mnist_train = DataLoader(torchvision.datasets.MNIST(os.path.join(
        data_folder, "mnist_original"), train=True, download=True))
    mnist_test = DataLoader(torchvision.datasets.MNIST(os.path.join(
        data_folder, "mnist_original"), train=False, download=True))

    mnist_train_data = mnist_train.dataset.data
    mnist_train_targets = mnist_train.dataset.targets

    x_test = mnist_test.dataset.data
    y_test = mnist_test.dataset.targets

    if data_flag == "mnist_original":
        x_train = mnist_train_data[0:55000]
        y_train = mnist_train_targets[0:55000]

        x_valid = mnist_train_data[55000:60000]
        y_valid = mnist_train_targets[55000:60000]

        train_data = TensorDataset(x_train, y_train)
        valid_data = TensorDataset(x_valid, y_valid)
        test_data = TensorDataset(x_test, y_test)

        return train_data, valid_data, test_data

    to_gray = (data_flag == "mnist_rgb2gray")
    x_train_color = color_dataset(mnist_train_data, to_gray)
    x_test_color = color_dataset(x_test, to_gray)

    # ADDED THIS
    mnist_train = TensorDataset(x_train_color, mnist_train_targets)
    mnist_test = TensorDataset(x_test_color, y_test)

    # first 55000 examples for training
    x_train = mnist_train[0:55000][0]
    y_train = mnist_train[0:55000][1]
    # y_train = mnist_train_targets[0:55000]

    # 5000 examples for validation set
    x_valid = mnist_train[55000:60000][0]
    y_valid = mnist_train[55000:60000][1]

    # 10000 examples in test set
    x_test = mnist_test[:][0]
    y_test = mnist_test[:][1]

    train_data = TensorDataset(x_train, y_train)
    valid_data = TensorDataset(x_valid, y_valid)
    test_data = TensorDataset(x_test, y_test)

    return train_data, valid_data, test_data


if WHICH_DATA_FLAG == "mnist_color":
    with open("./data/github_data/mnist_color28/" +
              "MNIST_color28_train.p", "rb") as f:
        mnist_train = pickle.load(f)

    with open("./data/github_data/mnist_color28/" +
              "MNIST_color28_test.p", "rb") as f:
        mnist_test = pickle.load(f)

    # first 55000 examples for training
    x_train = mnist_train[0:55000][0]
    y_train = mnist_train[0:55000][1]
    # y_train = mnist_train_targets[0:55000]

    # 5000 examples for validation set
    x_valid = mnist_train[55000:60000][0]
    y_valid = mnist_train[55000:60000][1]

    # 10000 examples in test set
    x_test = mnist_test[:][0]
    y_test = mnist_test[:][1]

    train_data = TensorDataset(x_train, y_train)
    valid_data = TensorDataset(x_valid, y_valid)
    test_data = TensorDataset(x_test, y_test)

elif WHICH_DATA_FLAG == "mnist_rgb2gray":
    with open("./data/github_data/mnist_color28_gray/"
              "MNIST_color28_gray_train.p", "rb") as f:
        mnist_train = pickle.load(f)

    with open("./data/github_data/mnist_color28_gray/"
              "MNIST_color28_gray_test.p", "rb") as f:
        mnist_test = pickle.load(f)

    # first 55000 examples for training
    x_train = mnist_train[0:55000][0]
    y_train = mnist_train[0:55000][1]
    # y_train = mnist_train_targets[0:55000]

    # 5000 examples for validation set
    x_valid = mnist_train[55000:60000][0]
    y_valid = mnist_train[55000:60000][1]

    # 10000 examples in test set
    x_test = mnist_test[:][0]
    y_test = mnist_test[:][1]

    train_data = TensorDataset(x_train, y_train)
    valid_data = TensorDataset(x_valid, y_valid)
    test_data = TensorDataset(x_test, y_test)

elif WHICH_DATA_FLAG == "mnist_original":
    mnist_train = torchvision.datasets.MNIST(
        "./data/github_data/mnist/", train=True, download=False)
    mnist_test = torchvision.datasets.MNIST(
        "./data/github_data/mnist/", train=False, download=False)

    mnist_train = DataLoader(mnist_train)
    mnist_test = DataLoader(mnist_test)

    mnist_train_data = mnist_train.dataset.data
    mnist_train_targets = mnist_train.dataset.target

    # 10000 examples in test set
    x_test = mnist_test.dataset.data
    y_test = mnist_test.dataset.targets

    # first 55000 examples for training
    x_train = mnist_train_data[0:55000]
    y_train = mnist_train_targets[0:55000]

    # 5000 examples for validation set
    x_valid = mnist_train_data[55000:60000]
    y_valid = mnist_train_targets[55000:60000]

    train_data = TensorDataset(x_train, y_train)
    valid_data = TensorDataset(x_valid, y_valid)
    test_data = TensorDataset(x_test, y_test)

elif WHICH_DATA_FLAG == "cifar":
    data = get_cifar10(
        './data/github_data/cifar-10-batches-py/', one_hot=False)

    # extract training,validation and test data
    X_train, Y_train = torch.from_numpy(
        data['train'].images), torch.from_numpy(data['train'].labels)
    X_validation, Y_validation = torch.from_numpy(
        data['validation'].images), torch.from_numpy(data['validation'].labels)
    X_test, Y_test = torch.from_numpy(
        data['test'].images), torch.from_numpy(data['test'].labels)

    # create datasets
    train_data = TensorDataset(X_train, Y_train)
    valid_data = TensorDataset(X_validation, Y_validation)
    test_data = TensorDataset(X_test, Y_test)


# Model parameters

# COPIED FROM THE ORIGINAL IMPLEMENTATION by Li et al. (2018)
# training parameters
learning_rate = 0.002
training_epochs = 31

# frequency of testing and saving
# how many epochs we do evaluate on the test set once, default 100
test_display_step = 5

# elastic deformation parameters
sigma = 4
alpha = 20

# lambda's are the ratios between the four error terms
# will be redefined later during grid search
lambda_class = 20  # classification error
lambda_ae = 1  # autoencoder
lambda_1 = 1  # push prototype vectors to meaningful decodings in pixel space
lambda_2 = 1  # cluster training examples around prototypes in latent space

input_height = input_width = image_size   # either 28 or 32
input_size = input_height * input_width * n_input_channels   # 784
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
# [out channel, in_channel, 3, 3]
filter_shape_1 = [n_map_1, n_input_channels, f_1, f_1]
filter_shape_2 = [n_map_2, n_map_1, f_2, f_2]
filter_shape_3 = [n_map_3, n_map_2, f_3, f_3]
filter_shape_4 = [n_map_4, n_map_3, f_4, f_4]

# strides for each layer (changed to tuples)
stride_1 = [s_1, s_1]
stride_2 = [s_2, s_2]
stride_3 = [s_3, s_3]
stride_4 = [s_4, s_4]


# Model construction
class Encoder(nn.Module):
    '''Encoder, encodes the traning instances to latent space. Consists
    of four convolutional layers with the same kernel size
    and uses ReLU as non-linearity.'''

    def __init__(self):
        super(Encoder, self).__init__()

        # define layers
        self.enc_l1 = nn.Conv2d(n_input_channels, 32,
                                kernel_size=3, stride=2, padding=0)
        self.enc_l2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0)
        self.enc_l3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0)
        self.enc_l4 = nn.Conv2d(32, 10, kernel_size=3, stride=2, padding=0)
        self.relu = nn.ReLU()

    def pad_image(self, img):
        ''' Takes an input image (batch) and pads according to Tensorflow's
        SAME padding'''
        input_h = img.shape[2]
        input_w = img.shape[3]
        stride = 2
        filter_h = 3

        if input_h % stride == 0:
            pad_height = max((filter_h - stride), 0)
        else:
            pad_height = max((filter_h - (input_h % stride), 0))

        pad_width = pad_height

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_img = torch.zeros(
            img.shape[0], img.shape[1], input_h + pad_height, input_w +
            pad_width)
        padded_img[:, :, pad_top:-pad_bottom, pad_left:-pad_right] = img

        return padded_img.to(device)

    def forward(self, x):
        pad_x = self.pad_image(x)
        el1 = self.relu(self.enc_l1(pad_x))

        pad_el1 = self.pad_image(el1)
        el2 = self.relu(self.enc_l2(pad_el1))

        pad_el2 = self.pad_image(el2)
        el3 = self.relu(self.enc_l3(pad_el2))

        pad_el3 = self.pad_image(el3)
        el4 = self.relu(self.enc_l4(pad_el3))

        return el4


class Decoder(nn.Module):
    '''Decoder with similar architecture as encoder. kernel sizes of 3.
    Last layer has a sigmoid as non-linearity instead of ReLU to map 
    to pixel values between 0 and 1'''

    def __init__(self):
        super(Decoder, self).__init__()

        if image_size == 28:
            padding_correction = 0
        else:
            padding_correction = 1

        # define layers
        self.dec_l4 = nn.ConvTranspose2d(
            10, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_l3 = nn.ConvTranspose2d(
            32, 32, kernel_size=3, stride=2, padding=1,
            output_padding=padding_correction)
        self.dec_l2 = nn.ConvTranspose2d(
            32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_l1 = nn.ConvTranspose2d(
            32, n_input_channels, kernel_size=3, stride=2, padding=1,
            output_padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, enc_x):
        dl4 = self.relu(self.dec_l4(enc_x))
        dl3 = self.relu(self.dec_l3(dl4))
        dl2 = self.relu(self.dec_l2(dl3))
        decoded_x = self.sigmoid(self.dec_l1(dl2))

        return decoded_x


class nn_prototype(nn.Module):
    '''Complete model. Uses the encoder and decoder, prototype layer 
    and classification'''

    def __init__(self, n_prototypes=15, n_layers=4, n_classes=10):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        n_features = 40  # size of encoded x - 250 x 10 x 2 x 2
        self.prototype_feature_vectors = nn.Parameter(torch.empty(size=(
            n_prototypes, n_features), dtype=torch.float32).uniform_())

        self.last_layer = nn.Linear(n_prototypes, 10)

    def list_of_distances(self, X, Y):
        '''
        Given a list of vectors, X = [x_1, ..., x_n], and another list
        of vectors, Y = [y_1, ... , y_m], we return a list of vectors
                [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
                 ...
                 [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
        where the distance metric used is the sqared euclidean distance.
        The computation is achieved through a clever use of broadcasting.
        '''
        XX = torch.reshape(list_of_norms(X), shape=(-1, 1))
        YY = torch.reshape(list_of_norms(Y), shape=(1, -1))
        output = XX + YY - 2 * torch.mm(X, Y.t())

        return output

    def forward(self, x):

        # encoder step
        enc_x = self.encoder(x)

        # decoder step
        dec_x = self.decoder(enc_x)

        # flatten encoded x to compute distance with prototypes
        n_features = enc_x.shape[1] * enc_x.shape[2] * enc_x.shape[3]
        feature_vectors_flat = torch.reshape(enc_x, shape=[-1, n_features])

        # distance to prototype
        prototype_distances = self.list_of_distances(
            feature_vectors_flat, self.prototype_feature_vectors)

        # distance to feature vectors
        feature_vector_distances = self.list_of_distances(
            self.prototype_feature_vectors, feature_vectors_flat)

        # classification layer
        logits = self.last_layer(prototype_distances)

        return dec_x, logits, feature_vector_distances, prototype_distances


# Cost function
'''
the error function consists of 4 terms, the autoencoder loss,
the classification loss, and the two requirements that every feature vector in
X look like at least one of the prototype feature vectors and every prototype
feature vector look like at least one of the feature vectors in X.
'''


def loss_function(X_decoded, X_true, logits, Y, feature_dist, prototype_dist,
                  lambdas=None, print_flag=False):
    if lambdas is None:
        lam_class, lam_ae = lambda_class, lambda_ae
        lam_1, lam_2 = lambda_1, lambda_2

    ae_error = torch.mean(list_of_norms(X_decoded - X_true))
    class_error = F.cross_entropy(logits, Y, reduction="mean")
    error_1 = torch.mean(torch.min(feature_dist, axis=1)[0])
    error_2 = torch.mean(torch.min(prototype_dist, axis=1)[0])

    # total_error is the our minimization objective
    total_error = lam_class * class_error + lam_ae * \
        ae_error + lam_1 * error_1 + lam_2 * error_2

    if print_flag is True:
        print('classification error', class_error.item())
        print('AE error: ', ae_error.item())
        print('Error 1: %f and 2: %f' % (error_1.item(), error_2.item()))
    return total_error


# Accuracy
def compute_acc(logits, labels):
    batch_size = labels.shape[0]
    predictions = logits.argmax(dim=1)
    total_correct = torch.sum(predictions == labels).item()
    accuracy = total_correct / batch_size

    return(accuracy)


# Function to visualize prototypes
def visualize_prototypes(model, epoch, save=True):
    # get saved prototypes
    encoded_prototypes = model.prototype_feature_vectors
    encoded_prototypes_reshaped = encoded_prototypes.view(
        n_prototypes, 10, 2, 2)

    # decode prototypes
    decoded_prototypes = model.decoder(
        encoded_prototypes_reshaped).cpu().detach().numpy()

    dec_prot = decoded_prototypes.transpose(0, 2, 3, 1)

    for i in range(n_prototypes):
        plt.imshow(dec_prot[i].squeeze())
        if save:
            makedirs(img_folder+"/prototypes_epoch_" + str(epoch))
            plt.savefig(img_folder+"/prototypes_epoch_" +
                        str(epoch)+"/"+str(i)+".png")
        else:
            plt.show()


# Training loop

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get validation and test set
valid_dl = DataLoader(valid_data, batch_size=5000,
                      drop_last=False, shuffle=False)
test_dl = DataLoader(test_data, batch_size=10000,
                     drop_last=False, shuffle=False)

# initialize optimizer
model = nn_prototype(15, 4, 10)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# initialize storage for results
train_accs = []
train_losses = []
test_accs = []
test_losses = []
valid_accs = []
valid_losses = []

model_folder += str(lambda_class) + '_' + str(lambda_ae) + '_' \
    + str(lambda_1) + '_' + str(lambda_2)

makedirs(model_folder)

# Image folder
img_folder = os.path.join(model_folder, "img")
makedirs(img_folder)

model_filename += str(lambda_class) + '_' + str(lambda_ae) + \
    '_' + str(lambda_1) + '_' + str(lambda_2)

# training loop
for epoch in range(training_epochs):
    print("\nEpoch:", epoch)

    # load the training data and reshuffle
    train_dl = DataLoader(
        train_data, batch_size=batch_size, drop_last=False,
        shuffle=True)

    # loop over the batches
    model.train()
    for step, (x, Y) in enumerate(train_dl):
        optimizer.zero_grad()

        if WHICH_DATA_FLAG == "mnist_original" \
           or WHICH_DATA_FLAG == "mnist_color":
            x = x.view(x.shape[0], n_input_channels,
                       x.shape[1], x.shape[2]).float() / 255
        elif WHICH_DATA_FLAG == "cifar":
            x = x / 255
        else:
            x = x.view(x.shape[0], n_input_channels,
                       x.shape[1], x.shape[2]).float()

        x = x.to(device)
        Y = Y.long()
        Y = Y.to(device)

        # perform forward pass
        X_decoded, logits, feature_dist, prot_dist = model(x)

        # compute the loss
        total_loss = loss_function(
            X_decoded, x, logits, Y, feature_dist, prot_dist)

        # backpropagate over the loss
        total_loss.backward()

        # update the weights
        optimizer.step()

        # compute and save accuracy and loss
        train_accuracy = compute_acc(logits, Y)
        train_accs.append(train_accuracy)
        train_losses.append(total_loss.item())

    # print information after a batch
    print('Last train loss of batch:', total_loss.item())
    print('Train acc on batch:', np.mean(train_accs[-step:]))
    print("Last train acc", train_accuracy)

    model.eval()
    with torch.no_grad():
        if epoch % test_display_step == 0:

            torch.save(model, model_folder + "/" +
                       model_filename + "_epoch_"
                       + str(epoch) + '.pt')

            visualize_prototypes(model, epoch, save=True)
            print("Model and prototypes of epoch %d are saved"
                  % epoch)

            # perform testing

            for (x_test, y_test) in test_dl:
                if WHICH_DATA_FLAG == "mnist_original" \
                   or WHICH_DATA_FLAG == "mnist_color":
                    x_test = x_test.view(
                        x_test.shape[0], n_input_channels,
                        x_test.shape[1],
                        x_test.shape[2]).float() / 255
                elif WHICH_DATA_FLAG == "cifar":
                    x_test = x_test / 255
                else:
                    x_test = x_test.view(
                        x_test.shape[0], n_input_channels,
                        x_test.shape[1], x_test.shape[2]).float()

                x_test = x_test.to(device)
                y_test = y_test.to(device)

                # forward pass
                X_decoded, logits, feature_dist, prot_dist = model(
                    x_test)

                # compute loss and accuracy and save
                test_accuracy = compute_acc(logits, y_test)
                test_loss = loss_function(
                    X_decoded, x_test, logits, y_test,
                    feature_dist, prot_dist)
                test_accs.append(test_accuracy)
                test_losses.append(test_loss)

            print('\nTest loss:', test_loss.item())
            print('Test acc:', test_accuracy)

    # validation
    with torch.no_grad():
        for (x_valid, y_valid) in valid_dl:
            if WHICH_DATA_FLAG == "mnist_original" \
               or WHICH_DATA_FLAG == "mnist_color":
                x_valid = x_valid.view(
                    x_valid.shape[0],
                    n_input_channels,
                    x_valid.shape[1],
                    x_valid.shape[2]).float() / 255
            elif WHICH_DATA_FLAG == "cifar":
                x_valid = x_valid / 255
            else:
                x_valid = x_valid.view(
                    x_valid.shape[0],
                    n_input_channels,
                    x_valid.shape[1],
                    x_valid.shape[2]).float()

            x_valid = x_valid.to(device)
            y_valid = y_valid.to(device)

            X_decoded, logits, feature_dist, prot_dist = model(
                x_valid)

            # compute losses and accuracy and save
            valid_accuracy = compute_acc(logits, y_valid)
            valid_loss = loss_function(
                X_decoded, x_valid, logits, y_valid, feature_dist,
                prot_dist, print_flag=False)
            valid_accs.append(valid_accuracy)
            valid_losses.append(valid_loss)

        print('\nValid loss:', valid_loss.item())
        print('Valid acc:', valid_accuracy)

# Save results
metrics_folder = os.path.join(model_folder, "metrics")
makedirs(metrics_folder)

with open(metrics_folder + '/train_accs.p', 'wb') as f:
    pickle.dump(train_accs, f)

with open(metrics_folder + '/test_accs.p', 'wb') as f:
    pickle.dump(test_accs, f)

with open(metrics_folder + '/valid_accs.p', 'wb') as f:
    pickle.dump(valid_accs, f)

with open(metrics_folder + '/train_losses.p', 'wb') as f:
    pickle.dump(train_losses, f)

with open(metrics_folder + '/test_losses.p', 'wb') as f:
    pickle.dump(test_losses, f)

with open(metrics_folder + '/valid_losses.p', 'wb') as f:
    pickle.dump(valid_losses, f)
