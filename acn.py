from torch.utils.cpp_extension import load
import torch


def ring_point(radius_in, radius_out, nsample, xyz1, xyz2, idx2):
    '''
    Input:
        radius_in: float32, ball search inner radius
        radius_out: float32, ball search outter radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
        idx2: (batch_size, npoint) int32, indecies of query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    return

def gather_point(inp,idx):
    '''
    input:
        batch_size * ndataset * 3   float32
        batch_size * npoints        int32
    returns:
        batch_size * npoints * 3    float32
    '''
    return


# THIS IS THE SAME AS INDEX POINTS
def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return

def order_neighbors(k, idx, input_xyz, query_xyz, query_normals):
    '''
    Order neighbors in counterclockwise manner.

    Input:
        input_xyz: (batch_size, ndataset, c) float32 array, input points
        query_xyz: (batch_size, npoint, c) float32 array, query points
        idx: (batch_size, npoint, k) int32 array, indecies of the k neighbor points
    Output:
        outi: (batch_size, npoint, k) int32 array, points orderred courterclockwise
        proj: (batch_size, npoint, k, 3) float32 array, projected neighbors on the local tangent plane
        angles: (batch_size, npoint, k) float32 array, values represents angles [0, 360)
    '''
    return



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 0), stride=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

def acnn_module_rings(xyz, points, normals, npoint, radius_list, nsample_list, mlp_list, use_xyz=True):
    ''' A-CNN module with rings
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            normals: (batch_size, ndataset, 3) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius_list: list of float32 -- search radiuses (inner and outer) represent ring in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    # data_format = 'NCHW' if use_nchw else 'NHWC'
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]

    # I think we can use our index_point function as is here directly, but we'll need to double check
    # new_xyz = gather_point(xyz, fps_idx)  # (batch_size, npoint, 3)
    # new_normals = gather_point(normals, fps_idx)  # (batch_size, npoint, 3)
    new_xyz = index_points(xyz, fps_idx)  # (batch_size, npoint, 3)
    new_normals = index_points(normals, fps_idx)  # (batch_size, npoint, 3)

    new_points_list = []
    for i in range(len(radius_list)):
        radius_in = radius_list[i][0]
        radius_out = radius_list[i][1]
        nsample = nsample_list[i]
        idx, _ = ring_point(radius_in, radius_out, nsample, xyz, new_xyz, fps_idx)
        idx, _, _ = order_neighbors(nsample, idx, xyz, new_xyz, new_normals)

        #grouped_xyz = group_point(xyz, idx)
        grouped_xyz = index_points(xyz, idx)

        #grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
        grouped_xyz -= new_xyz.unsqueeze(2).expand(-1, -1, nsample, -1)

        if points is not None:
            #grouped_points = group_point(points, idx)
            grouped_points = index_points(points, idx)
            if use_xyz:
                #grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        else:
            grouped_points = grouped_xyz
        for j, num_out_channel in enumerate(mlp_list[i]):
            #grouped_points = tf.concat([grouped_points, grouped_points[:, :, :2, :]], axis=2)
            grouped_points = torch.cat([grouped_points, grouped_points[:, :, :2, :]], dim=2)

            # NOTE I REMOVED BATCHNORM WEIGHT DECAY
            conv_block = ConvBlock(in_channels=grouped_points.shape[-1], out_channels=num_out_channel)
            grouped_points = conv_block(grouped_points)

            #grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 3],
            #                                padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
            #                                scope='conv%d_%d' % (i, j), bn_decay=bn_decay)

        new_points, _ = torch.max(grouped_points, dim=2)
        new_points_list.append(new_points)


    #new_points_concat = tf.concat(new_points_list, axis=-1)
    new_points_concat = torch.cat(new_points_list, dim=-1)
    return new_xyz, new_points_concat, new_normals

