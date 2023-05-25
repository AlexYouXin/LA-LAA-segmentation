import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import time
import os
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.kde import KernelDensity
import skimage.io

def locate_LA_LAA(input):
    input = input[0]
    LAA = torch.zeros_like(input)
    LAA[input == 1] = 1
    count = torch.count_nonzero(LAA)
    LA = torch.zeros_like(input)
    LA[input == 2] = 1

    return LAA, LA, count

def centroid_calculate(LAA):

    count = torch.count_nonzero(LAA)
    index = torch.nonzero(LAA)
    # sum by coloumn directions
    centroid = torch.sum(index, 0) / count
    return centroid, index


def index_LA(LA):
    index = torch.nonzero(LA)
    return index


# input: after softmax and argmax
# target: GT
class the_connectivity_loss(nn.module):
    def __init__(self, n_classes, scale=20):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.scale = scale

    def forward(self, inputs, target):
        b, L, H, W = inputs.size()
        LAA, LA, count = locate_LA_LAA(inputs)

        if count == 0:
            return 0
        else:
            # calculate centroid of LAA
            centroid, index = centroid_calculate(LAA)     
            # centroid: # [[a, b, c]], 1 * 3
            # index: count * 3
            distance_matrix = pairwise_distances(centroid, index, metric='euclidean')
            if count > 200:
                distance_matrix, indices = torch.sort(distance_matrix)
                distance_matrix[torch.int(count * 0.005):] = 0
	    index = index[indices]

            # max_distance = torch.max(distance_matrix)
            max_distance_index = torch.argmax(distance_matrix)

            # vertex
            vertex_z = index[max_distance_index][0]
            vertex_y = index[max_distance_index][1]
            vertex_x = index[max_distance_index][2]

            # each coordinate of centroid
            centroid_z = centroid[0]
            centroid_y = centroid[1]
            centroid_x = centroid[2]

            z_length = torch.abs(centroid_z - vertex_z)
            y_length = torch.abs(centroid_y - vertex_y)
            x_length = torch.abs(centroid_x - vertex_x)


            vertex_z_down = centroid_z - z_length
            vertex_y_down = centroid_y - y_length
            vertex_x_down = centroid_x - x_length
            vertex_y_up = centroid_y + y_length
            vertex_x_up = centroid_x + x_length

            # vertex 1
            vertex1 = torch.tensor([[vertex_z_down, vertex_y_down, vertex_x_down]])
            # vertex 2
            vertex2 = torch.tensor([[vertex_z_down, vertex_y_down, vertex_x_up]])
            # vertex 3
            vertex3 = torch.tensor([[vertex_z_down, vertex_y_up, vertex_x_down]])
            # vertex 4
            vertex4 = torch.tensor([[vertex_z_down, vertex_y_up, vertex_x_up]])

            LA_index = index_LA(LA)

            distance_LA_1 = pairwise_distances(vertex1, LA_index, metric='euclidean')
            min_distance_1 = torch.min(distance_LA_1)
            distance_LA_2 = pairwise_distances(vertex2, LA_index, metric='euclidean')
            min_distance_2 = torch.min(distance_LA_2)
            distance_LA_3 = pairwise_distances(vertex3, LA_index, metric='euclidean')
            min_distance_3 = torch.min(distance_LA_3)
            distance_LA_4 = pairwise_distances(vertex4, LA_index, metric='euclidean')
            min_distance_4 = torch.min(distance_LA_4)

            c_loss = torch.sigmoid((min_distance_1 + min_distance_2 +min_distance_3 + min_distance_4) / self.scale) - 0.5
            return c_loss
