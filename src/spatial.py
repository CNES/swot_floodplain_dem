# -*- coding: utf8 -*-
'''
Library provided extra methods

Copyright (c) 2018, CNES
'''

import numpy as np


def compute_binary_mask(in_sizex, in_sizey, in_x, in_y):
    '''
    Creates a 2D binary matrix from X and Y 1D vectors
    i.e. for each ind: mat[in_x[ind], in_y[ind]] = 1

    :param in_sizex: number of pixels in X dimension
    :type in_sizex: int
    :param in_sizey: number of pixels in Y dimension
    :type in_sizey: int
    :param in_x: X indices of "1" pixels
    :type in_x: 1D vector of int
    :param in_y: Y indices of "1" pixels
    :type in_y: 1D vector of int

    :return: 2D matrix with "1" for each (in_x_i, in_y_i) and 0 elsewhere
    :rtype: 2D binary matrix of int 0/1
    '''

    # 0 - Deal with exceptions
    # 0.1 - Input vectors size must be the same
    if in_x.size != in_y.size:
        raise ValueError("computeBinMat(in_x, in_y): in_x and in_y must be the same size; currently: in_x=%d and in_y=%d"
             % (in_x.size, in_y.size))
    else:
        nb_pts = in_x.size

    # 0.2 - max(X) < in_sizex
    if np.max(in_x) >= in_sizex:
        raise ValueError("computeBinMat(in_x, in_y) : elements of in_x must be less than in_sizex")
    # 0.3 - max(X) < in_sizex
    if np.max(in_y) >= in_sizey:
        raise ValueError("computeBinMat(in_x, in_y) : elements of in_y must be less than in_sizey")

    # 1 - Init output binary image
    OUT_binIm = np.zeros((in_sizex, in_sizey))

    # 2 - Put 1 for every pixels defined by the input vectors
    OUT_binIm[in_x, in_y] = 1

    return OUT_binIm