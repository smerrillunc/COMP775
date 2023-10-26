import torch
import torch.nn as nn
from pointnet_util import farthest_point_sample, index_points, square_distance, minkowski_distance, cosine_sim
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import time

def discretize_points(xyz, cube_dim):
    """
    Description: This function groups N x 3 array of xyz
    indexes and returns a corresponding array of indexes to which each
    point should be assigned in a 3d with a total of cube_dim^3 blocks    inputs:
        xyz: an N x 3 list of xyz points
        cube_dim: the dimension of the 3d cube to discretize x into    outputs:
        xyz_groupings: an N x 3 list which provides the index of the cube
        for which each xyz coordinate should be mapped
    """    # get mins/maxes in range
    min_tens = torch.min(xyz, axis=1)[0] # shape = B x 3
    max_tens = torch.max(xyz, axis=1)[0]   # number of points in 3d cube
    #print(max_tens.shape)
    max_x, max_y, max_z = max_tens[:,0], max_tens[:,1], max_tens[:,2]  # number of points in 3d cube
    min_x, min_y, min_z = min_tens[:,0], min_tens[:,1], min_tens[:,2]  # number of points in 3d cube
    #print(max_x.shape)
    num_x_points, num_y_points, num_z_points = cube_dim, cube_dim, cube_dim    #####################################################################
    # Leaving this here as a sanity check for our more efficient output #
    #####################################################################
    # create cube barriers
    #x = torch.linspace(min_x, max_x, num_x_points)
    #y = torch.linspace(min_y, max_y, num_y_points)
    #z = torch.linspace(min_z, max_z, num_z_points)    # x_indexes = torch.expand_dims(torch.searchsorted(x, xyz[:,0]), axis=1)
    # y_indexes = torch.expand_dims(torch.searchsorted(y, xyz[:,1]), axis=1)
    # z_indexes = torch.expand_dims(torch.searchsorted(z, xyz[:,2]), axis=1)
    #####################################################################    # can find which cube they belong to mathematically
    # print(xyz[:,:,0].shape, min_x.shape)
    x_indexes = torch.floor((xyz[:,:,0] - min_x[:,None])/(max_x[:,None]-min_x[:,None])*num_x_points)
    y_indexes = torch.floor((xyz[:,:,1] - min_y[:,None])/(max_y[:,None]-min_y[:,None])*num_y_points)
    z_indexes = torch.floor((xyz[:,:,2] - min_z[:,None])/(max_z[:,None]-min_z[:,None])*num_z_points)    # stacking these together, we get groupings

    x_indexes = torch.unsqueeze(torch.clamp(x_indexes, max=num_x_points - 1), axis=1)
    y_indexes = torch.unsqueeze(torch.clamp(y_indexes, max=num_y_points - 1), axis=1)
    z_indexes = torch.unsqueeze(torch.clamp(z_indexes, max=num_z_points - 1), axis=1)

    xyz_groupings = torch.hstack([x_indexes, y_indexes, z_indexes])
    return xyz_groupings

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint

    W = 8 # box width
    GRID_SIZE = W*W*W
    indices = discretize_points(xyz, W) # B x 3 x N
    indices = indices.permute(0, 2, 1) # B x N x 3
    # (0,0) (0,1) (0,2)
    # (1,0) 
    #
    # 0 1 2
    # 3 4 5
    # 6 7 
    #print(indices.shape)
    #print(B,N,C)
    indices = indices[:,:,0] * W * W + indices[:,:,1] * W + indices[:,:,2]
    #print(indices.shape)
    indices = indices.long() # B x N

    min_tens = torch.min(xyz, axis=1)[0] # shape = B x 3
    max_tens = torch.max(xyz, axis=1)[0]   # number of points in 3d cube
    max_x, max_y, max_z = max_tens[:,0], max_tens[:,1], max_tens[:,2]  # number of points in 3d cube
    min_x, min_y, min_z = min_tens[:,0], min_tens[:,1], min_tens[:,2]  # number of points in 3d cube

    # [0, 6, 7, 8, ...]

    def find_first_occurence(x):
        unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        return perm

    # select one point from each voxel
    selected = [find_first_occurence(idx) for idx in indices]

    fps_idx = torch.stack([torch.randperm(N)[:npoint] for _ in range(B)]).cuda()
    for i in range(len(fps_idx)):
        s = selected[i]
        fps_idx[i,0:len(s)] = s

    #fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)

    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #points_mask = (x != float('-inf'))
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        #x_q[!points_mask] = 0.
        #x_k[!points_mask] = 0.
        #x_v[!points_mask] = 0.
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.Identity() # nn.BatchNorm1d(channels)
        self.bn2 = nn.Identity() # nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        #print("shape", x.shape)

        #print(x.shape)
        return x


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim


        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=256)
        #self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention(channels=256) # modified

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        #self.linear1 = nn.Linear(512, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        #import time
        #t_m1 = time.time()

        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)

        #t_sg1 = time.time()
        # fix npoint
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        #t_sg1_e = time.time()
        #print("SG1:", t_sg1_e - t_sg1)

        feature_0 = self.gather_local_0(new_feature)
        # no second sample and group architecture here
        feature_1 = feature_0

        #t_m2 = time.time()
        #print("Reducing:", t_m2 - t_m1)
        
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        #t_m3 = time.time()
        #print("Attn:", t_m3 - t_m2)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        #t_m4 = time.time()
        #print("Head:", t_m4 - t_m3)

        return x

