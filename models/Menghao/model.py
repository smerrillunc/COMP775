import torch
import torch.nn as nn
from pointnet_util import farthest_point_sample, index_points, square_distance, minkowski_distance, cosine_sim, generalized_distance
from .modules import ISAB  # Import ISAB from modules.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_and_group(npoint, nsample, xyz, points, sampling_method, distance_function, local_features="diff"):
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
    else:
        raise KeyError("Invalid sampling method")

    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

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
    #local_feat = generalized_distance(grouped_points, new_points, B, S, p=1.0)
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
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

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

        return x


class MenghaoPointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.n_points_attn = cfg.num_points_attn
        
        self.sampling_method = cfg.sampling_method
        
        self.distance_function = cfg.distance_function
        
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # NOTE CHANGE GATHER LOCAL LAYERS CORRESPONDING TO METHOD WE'RE USING ABOVE
        # USING SUMMARY METRICS (COSINE SIM/SUM/AVERAGING OVER THE FEATURES WE SHOULD
        # USE THIS
        # TODO add if statement
        # self.gather_local_0 = Local_op(in_channels=65, out_channels=128)
        # self.gather_local_1 = Local_op(in_channels=129, out_channels=256)

        # IF WE WANT DISTANCES IN EACH CARDINAL DIRECTION OF DATA USE THIS
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = StackedAttention()

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
            # ISAB layer
            dim_in = self.n_points_attn  # Adjust according to your model's output
            dim_out = self.n_points_attn  # Adjust according to your model's output
            num_heads = 4  # Number of heads in multi-head attention
            num_inds = 32  # Number of inducing points in ISAB

            if cfg.use_isab == 1:
                self.isab = ISAB(dim_in, dim_out, num_heads, num_inds)
            if cfg.use_isab == 3:
                self.isab = ISAB(dim_in, dim_out, 1, num_inds)
            if cfg.use_isab == 2:
                print("test")

                self.isab = nn.Sequential(
                    ISAB(dim_in, dim_out, 1, num_inds),
                    ISAB(dim_in, dim_out, 1, num_inds),
                )

                
            self.conv_fuse = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(negative_slope=0.2))
            self.linear1 = nn.Linear(512, 512, bias=False)
        else:
            self.isab = None

    def forward(self, x: torch.Tensor):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=self.n_points_attn * 2, nsample=32, xyz=xyz, points=x, sampling_method=self.sampling_method, distance_function=self.distance_function)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=self.n_points_attn, nsample=32, xyz=new_xyz, points=feature, sampling_method=self.sampling_method, distance_function=self.distance_function)
        feature_1 = self.gather_local_1(new_feature)
        
        if self.isab is not None:
            # print(feature_1.shape)
            x = self.isab(feature_1)
        else:
            x = self.pt_last(feature_1)
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