import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import cv2
import time
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.extmath import cartesian
import skimage.io

torch.set_default_dtype(torch.float32)

def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

# to record the index of each voxel point
def fixed_index(obj):
    z, y, x = obj.size()
    index_tensor = torch.zeros((3, z, y, x))
    '''
    for i in range(z):
	for j in range(y):
            for k in range(x):
	        index_tensor[:, i, j, k] = torch.tensor((i, j, k))
    '''
    x_list = torch.arange(0, x)
    y_list = torch.arange(0, y)
    z_list = torch.arange(0, z)
    corr = torch.meshgrid(z_list, y_list, x_list)
    # print(corr[0])
    # print(corr[1])
    # print(corr[2])
    a = torch.stack([corr[0], corr[1], corr[2]], dim=0)
    return index_tensor
		
		
# To localize the LA and LAA respectively
# count: number of voxels in LAA
def locate_LA_LAA(input):
    '''
    Input: input
    Output: LAA, LA, count
    '''
    LAA = torch.zeros_like(input)
    LAA[input == 1] = 1
    count = torch.count_nonzero(LAA)
    LA = torch.zeros_like(input)
    LA[input == 2] = 1

    return LAA, LA, count

def centroid_calculate_LAA(LAA):
    # tensor calculation -> differentiable
    count = torch.count_nonzero(LAA)
    index_tensor = fixed_index(LAA)
    index_tensor = index_tensor * LAA    
    centroid = torch.sum(index_tensor, dim=(1, 2, 3)) / count
    LAA_index = index_tensor.flatten(1).transpose(-1, -2)
    return centroid, LAA_index


def index_LA(LA):
    index_tensor = fixed_index(LA)
    index_tensor = index_tensor * LAA
    LA_index = index_tensor.flatten(1).transpose(-1, -2)
    return LA_index

# inputs: hard label
# target: GT
class the_connectivity_loss(nn.Module):
    def __init__(self, n_classes, scale=20):
        super(the_connectivity_loss, self).__init__()
        self.n_classes = n_classes
        self.scale = scale

    def forward(self, inputs, target):
        b, L, H, W = inputs.size()
    	loss = 0.0
    	for i range(b):
            LAA, LA, count = locate_LA_LAA(inputs[i])

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
    	    max_distance_value, max_distance_index = torch.max(distance_matrix)

            # vertex
    	    # index reused in tensor -> differentiable
            vertex_z = index[max_distance_index, 0]
            vertex_y = index[max_distance_index, 1]
            vertex_x = index[max_distance_index, 2]

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
            loss += c_loss
        return loss
