# -*- coding: utf-8 -*-
# python 3
# @Author: TomLotze
# @Date:   2020-01-17 12:38:54
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-01-17 15:28:23



def print_distances(input_image, prototypes):
    """
    This function takes an input image and a list of prototypes and prints the
    distances to all the prototypes
    """

    # use the function that

    lodist = list_of_distances(prototypes, input)

    return lodist



def list_of_norms(X):
    '''
    X is a list of vectors X = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    '''
    return tf.reduce_sum(np.pow(X, 2), axis=1)



def list_of_distances(X, Y):
    '''
    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    '''
    XX = torch.reshape(list_of_norms(X), shape=(-1, 1))
    YY = torch.reshape(list_of_norms(Y), shape=(1, -1))
    output = XX + YY - 2 * torch.mm(X, torch.transpose(Y))

    return output