import torch
import torch.nn as nn
# from train_cls import ParserArgs
from pointnet_util import farthest_point_sample, index_points, square_distance, minkowski_distance, cosine_sim, generalized_distance, query_ball_point
from .modules import ISAB  # Import ISAB from modules.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def voxelize(npoint, nsample, xyz, points, W) -> torch.Tensor:
    B, N, C = xyz.shape
    S = npoint

    GRID_SIZE = W*W*W
    indices = discretize_points(xyz, W) # B x 3 x N
    indices = indices.permute(0, 2, 1) # B x N x 3
    # maps
    # (0,0) (0,1) (0,2)
    # (1,0) 
    # to
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

    # if there are less than W*W*W points, we fill the rest with randomly selected points
    fps_idx = torch.stack([torch.randperm(N)[:npoint] for _ in range(B)]).to(device)
    for i in range(len(fps_idx)):
        s = selected[i]
        # print(fps_idx.shape,len(s))
        fps_idx[i,0:len(s)] = s

    #fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    
    return fps_idx

def sample_and_group(npoint: torch.Tensor, nsample: int, xyz: torch.Tensor, points: torch.Tensor, sampling_method: str, distance_function, local_features="diff", voxel_width=8, radii=[0.1]):
    B, N, C = xyz.shape
    S = npoint

    # SHAPE REFERENCE
    # B = Batch size (16)
    # N = Number poitns input 1024 for first layer
    # C = Cardinality of x, y, z = 3 always
    # S/npoints: points to downsample to
    # nsample: number of neighbors 32


    # SAMPLING METHOD 1: DEFAULT FPS
    if sampling_method == 'fps':
        fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    # SAMPLING METHOD 2: RANDOM SAMPLING
    # NYI/TODO
    elif sampling_method == 'random':
        fps_idx = torch.stack([torch.randperm(xyz.size()[1]).to(device)[:npoint] for _ in range(xyz.shape[0])])
    elif sampling_method == 'voxel':
        fps_idx = voxelize(npoint, nsample, xyz, points, voxel_width)
    else:
        raise KeyError("Invalid sampling method")
    
        

    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)


    #### Ball query with MSG
    if distance_function =='bq':
        print("Ball Query Start")
        # relative scale to use for radii
        dists = square_distance(new_xyz, xyz)
        scale = float(torch.max(dists) - torch.min(dists))
        # check different radii
        # radii = [0.1]
        # initialize empty tensor
        grouped_points = index_points(points, query_ball_point(radii[0]*scale, nsample, xyz, new_xyz))
        local_feat = grouped_points - new_points.view(B, S, 1, -1)

        for i in range(1, len(radii)):
            grouped_points = index_points(points, query_ball_point(radii[i]*scale, nsample, xyz, new_xyz))
            tmp = grouped_points - new_points.view(B, S, 1, -1)
            local_feat = torch.cat((tmp, local_feat), dim=-1)

        new_points = torch.cat([local_feat, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
        return new_xyz, new_points
    ####




    ################################################################################
    # DISTANCE METRIC 1: DEFUALT SQUARED DISTANCE
    if distance_function == "square":
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
    ################################################################################

    ################################################################################
    # DISTANCE METRIC 2: manhattan distance
    if distance_function == "mink_1":
        dists = minkowski_distance(new_xyz, xyz, p=1)
    ################################################################################

    ################################################################################
    # DISTANCE METRIC 3: euclidean distance
    # dists = minkowski_distance(new_xyz, xyz, p=2)
    ################################################################################

    ################################################################################
    # DISTANCE METRIC 4: p = 3
    if distance_function == "mink_3":
        dists = minkowski_distance(new_xyz, xyz, p=3)
    ################################################################################

    ################################################################################
    # DISTANCE METRIC 5: cosine similarity just on xyz
    if distance_function == "cos_sim_1":
        dists = cosine_sim(new_xyz, xyz)
    ################################################################################

    ################################################################################
    # DISTANCE METRIC 6: ccosine similary weighting the xyz portion of similarity by a
    # and the point features aspect of similiarty by b
    ################################################################################
    if distance_function == "cos_sim_2":
        a = 0.75
        b = 0.25
        dists = a * cosine_sim(new_xyz, xyz) + b * cosine_sim(new_points[..., 3:], points[..., 3:])

    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    # the k closest points to each of the "new downsampled points"
    # B, S, K, C
    grouped_points = index_points(points, idx)


    #### NOTE DEPENDING ON WHAT CLASS OF METHOd WE CHOOSE WE'll HAVE TO CHANGE THE SHAPES OF THE
    #### GATHER LOCAL LAYERS IN THE MAIN MODEL.  THIS IS BECAUSE WE ARE CONSIDERING TWO DIFFERENT CARDINALITIES
    #### OF LOCAL FEATURE SIZE.

    ##### THIS CLASS OF LOCAL FEATURE METHODS WILL NOT CHANGE THE CARDINALITY OF THE INPUT #####
    # Method 1: raw differences what the paper uses
    if local_features == "diff":
        local_feat = grouped_points - new_points.view(B, S, 1, -1)

    # Method 2: absolute value of the differences
    #local_feat = torch.abs(grouped_points - new_points.view(B, S, 1, -1))

    # Method 3: a distance based approach
    # we have many different values of p in this setting
    #local_feat = generalized_distance(grouped_points, new_points, B, S, p=0.25)
    #local_feat = generalized_distance(grouped_points, new_points, B, S, p=0.5)
    #local_feat = generalized_distance(grouped_points, new_points, B, S, p=0.75)
    if local_features == "dist_p_1":
        local_feat = generalized_distance(grouped_points, new_points, B, S, p=1.0)
    #local_feat = generalized_distance(grouped_points, new_points, B, S, p=2.0)
    #local_feat = generalized_distance(grouped_points, new_points, B, S, p=4.0)
    #new_points = torch.cat([local_feat, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    #print(local_feat.shape)

    ##### THIS CLASS OF LOCAL FEATURE METHODS WILL CHANGE THE CARDINALITY OF THE INPUT ####
    # Method 1: Summing the differences of the features of all neighbors
    if local_features == "diff_2":
        local_feat = torch.sum(grouped_points - new_points.view(B, S, 1, -1), keepdims=True, dim=-1)

    # Method 2: Summing the absolute value of the differences of the features
    #local_feat = torch.sum(torch.abs(grouped_points - new_points.view(B, S, 1, -1)), keepdims=True, dim=-1)

    # Method 3: Cosine Similarity of feature vects
    # a bit confusing vectorization but I tested it
    if local_features == "cos_sim":
        local_feat = torch.sum((grouped_points * new_points.view(B, S, 1, -1)), keepdims=True, dim=-1) / (
                                torch.linalg.vector_norm(grouped_points, keepdims=True,dim=-1) *
                                torch.linalg.vector_norm(new_points.view(B, S, 1, -1), keepdims=True, dim=-1)
                                )

    new_points = torch.cat([local_feat, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
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
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=256, cfg=None):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.use_isab = cfg.use_isab

        if cfg.use_isab > 0:
            # ISAB layer
            dim_in = cfg.num_points_attn  # Adjust according to your model's output
            dim_out = cfg.num_points_attn  # Adjust according to your model's output
            num_heads = 4  # Number of heads in multi-head attention
            num_inds = 32  # Number of inducing points in ISAB
            
            if cfg.use_isab == 1:
                self.isab = ISAB(dim_in, dim_out, num_heads, num_inds)
            if cfg.use_isab == 3:
                self.isab1 = ISAB(dim_in, dim_out, 1, num_inds)
                self.isab2 = ISAB(dim_in, dim_out, 1, num_inds)
                self.isab3 = ISAB(dim_in, dim_out, 1, num_inds)
                self.isab4 = ISAB(dim_in, dim_out, 1, num_inds)
            if cfg.use_isab == 2:
                # print("test")

                self.isab1 = ISAB(dim_in, dim_out, 1, num_inds)
                self.isab2 = ISAB(dim_in, dim_out, 1, num_inds)
                        
            if cfg.use_isab == 4:
                self.isab = ISAB(dim_in, dim_out, num_heads, num_inds, False, True)
            if cfg.use_isab == 5:
                # print("test")

                self.isab1 = ISAB(dim_in, dim_out, 1, num_inds,  False, True)
                self.isab2 = ISAB(dim_in, dim_out, 1, num_inds,  False, True)
            
            if cfg.use_isab == 6:
                self.isab1 = ISAB(dim_in, dim_out, 1, num_inds,  False, True)
                self.isab2 = ISAB(dim_in, dim_out, 1, num_inds,  False, True)
                self.isab3 = ISAB(dim_in, dim_out, 1, num_inds,  False, True)
                self.isab4 = ISAB(dim_in, dim_out, 1, num_inds,  False, True)

        else:
            self.sa1 = SA_Layer(channels)
            self.sa2 = SA_Layer(channels)
            self.sa3 = SA_Layer(channels)
            self.sa4 = SA_Layer(channels)
            
            self.isab = None

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()


        if self.use_isab in [1,4]:
            return self.isab(x)
        elif self.use_isab in [2,5]:
            x1 = self.isab1(x)
            x2 = self.isab2(x1)

            x = torch.cat((x1, x2), dim=1)
        elif self.use_isab in [3,6]:
            x1 = self.isab1(x)
            x2 = self.isab2(x1)
            x3 = self.isab2(x2)
            x4 = self.isab2(x3)


            x = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            x = self.relu(self.bn1(self.conv1(x))) # B, D, N
            x = self.relu(self.bn2(self.conv2(x)))
            x1 = self.sa1(x)
            x2 = self.sa2(x1)
            x3 = self.sa3(x2)
            x4 = self.sa4(x3)
            
            x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class MenghaoPointTransformerCls(nn.Module):
    def __init__(self, cfg):
        # print(cfg.num_points_attn)
        super().__init__()

        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.n_points_attn = cfg.num_points_attn
        
        self.sampling_method = cfg.sampling_method
        
        self.distance_function = cfg.distance_function
        self.downsample_layer_count = cfg.downsample_layer_count
        
        self.local_features = cfg.local_features
        
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.radii = list(map(float, cfg.radii))

        # NOTE CHANGE GATHER LOCAL LAYERS CORRESPONDING TO METHOD WE'RE USING ABOVE
        # USING SUMMARY METRICS (COSINE SIM/SUM/AVERAGING OVER THE FEATURES WE SHOULD
        # USE THIS
        # TODO add if statement
        if self.local_features in ["diff_2", "cos_sim"]:
            self.gather_local_0 = Local_op(in_channels=65, out_channels=128)
            self.gather_local_1 = Local_op(in_channels=129, out_channels=256)

        else:
            # IF WE WANT DISTANCES IN EACH CARDINAL DIRECTION OF DATA USE THIS
            if self.downsample_layer_count == 2:
                self.gather_local_0 = Local_op(in_channels=64+64*len(self.radii), out_channels=128)
                self.gather_local_1 = Local_op(in_channels=128+128*len(self.radii), out_channels=256)
            elif self.downsample_layer_count == 1:
                self.gather_local_1 = Local_op(in_channels=128+128*len(self.radii), out_channels=256)
        self.pt_last = StackedAttention(channels=256, cfg=cfg)

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

        print("isab:", cfg.use_isab)

        if cfg.use_isab > 0:
            # modify layer counts.
            if cfg.use_isab in [1,4]:
                self.conv_fuse = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(negative_slope=0.2))
                self.linear1 = nn.Linear(512, 512, bias=False)
            if cfg.use_isab in [2,5]:
                self.conv_fuse = nn.Sequential(nn.Conv1d(768, 512, kernel_size=1, bias=False),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(negative_slope=0.2))
                self.linear1 = nn.Linear(512, 512, bias=False)
            if cfg.use_isab in [3,6]:
                self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                nn.BatchNorm1d(1024),
                                nn.LeakyReLU(negative_slope=0.2))
                self.linear1 = nn.Linear(1024, 512, bias=False)

    def forward(self, x: torch.Tensor):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        
        if self.downsample_layer_count == 2:
            import math
            # This is not used if the voxel method is not used
            voxel_width_1 = int(math.floor((2*self.n_points_attn) ** (1/3)))
            new_xyz, new_feature = sample_and_group(npoint=self.n_points_attn * 2, nsample=32, xyz=xyz, points=x, sampling_method=self.sampling_method, distance_function=self.distance_function, local_features=self.local_features, voxel_width=voxel_width_1, radii=self.radii)
            feature_0 = self.gather_local_0(new_feature)
            feature = feature_0.permute(0, 2, 1)
            
            voxel_width_2 = int(math.floor((self.n_points_attn) ** (1/3)))
            new_xyz, new_feature = sample_and_group(npoint=self.n_points_attn, nsample=32, xyz=new_xyz, points=feature, sampling_method=self.sampling_method, distance_function=self.distance_function, local_features=self.local_features, voxel_width=voxel_width_2, radii=self.radii)
            feature_1 = self.gather_local_1(new_feature)
        elif self.downsample_layer_count == 1:
            import math
            voxel_width = int(math.floor(self.n_points_attn ** (1/3)))
            new_xyz, new_feature = sample_and_group(npoint=self.n_points_attn, nsample=32, xyz=xyz, points=x, sampling_method=self.sampling_method, distance_function=self.distance_function, local_features=self.local_features, voxel_width=voxel_width, radii=self.radii)
            feature_1 = self.gather_local_1(new_feature)
        else:
            raise KeyError()

        x = self.pt_last(feature_1)
        # print(x.shape,feature_1.shape)
        x = torch.cat([x, feature_1], dim=1)

        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x