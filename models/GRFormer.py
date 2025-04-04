

import torch
from torch import einsum
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling
from extensions.chamfer_dist import ChamferDistanceL2
from extensions.gridding_loss import GriddingLoss
from pointnet2_ops import pointnet2_utils
import torch.nn.functional as F
from .build import MODELS

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()
class MLP_CONV(torch.nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(torch.nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(torch.nn.BatchNorm1d(out_channel))
            layers.append(torch.nn.ReLU())
            last_channel = out_channel
        layers.append(torch.nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(torch.nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = torch.nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = torch.nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = torch.nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out
class SA_Layer(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = torch.nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = torch.nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = torch.nn.Conv1d(channels, channels, 1)
        self.trans_conv = torch.nn.Conv1d(channels, channels, 1)
        self.after_norm = torch.nn.BatchNorm1d(channels)
        self.act = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.pos_mlp = torch.nn.Conv1d(channels, channels, 1)


    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention.transpose(1, 2)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class cross_transformer(torch.nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = torch.nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = torch.nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.linear12 = torch.nn.Linear(dim_feedforward, d_model_out)
        # 层归一化
        self.norm12 = torch.nn.LayerNorm(d_model_out)
        self.norm13 = torch.nn.LayerNorm(d_model_out)

        self.dropout12 = torch.nn.Dropout(dropout)
        self.dropout13 = torch.nn.Dropout(dropout)
        # 激活函数
        self.activation1 = torch.nn.GELU()
        # 输入投影，将输入数据的维度从d_model变为d_model_out
        self.input_proj = torch.nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1
class SkipTransformer(torch.nn.Module):
    def __init__(self, in_channel, out_channel, dim=256, n_knn=20, up_factor=8 ,pos_hidden_dim=64, 
                 attn_hidden_multiplier=4, attn_channel=True):
        super(SkipTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        attn_out_channel = dim if attn_channel else 1
        
        self.conv_key = torch.nn.Conv1d(in_channel, dim, 1)
        self.conv_query = torch.nn.Conv1d(in_channel, dim, 1)
        self.mlp_v = MLP_Res(in_dim=in_channel* 2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_value = torch.nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(3, pos_hidden_dim, 1),
            torch.nn.BatchNorm2d(pos_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = [torch.nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            torch.nn.BatchNorm2d(dim * attn_hidden_multiplier),
            torch.nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(torch.nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor,1), (up_factor,1)))
        else:
            self.attn_mlp.append(torch.nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = torch.nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = torch.nn.Upsample(scale_factor=(up_factor,1)) if up_factor else torch.nn.Identity()
        self.upsample2 = torch.nn.Upsample(scale_factor=up_factor) if up_factor else torch.nn.Identity()
       
        # residual connection
        self.conv_end = torch.nn.Conv1d(dim, out_channel, 1)
       

    def forward(self, pos, key, query, include_self=True):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
            include_self: boolean

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        print("value:",value.size())
        key = self.conv_key(key) # B, dim, N
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous() # B,N,3
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped, include_self=include_self)

        key = pointnet2_utils.grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - pointnet2_utils.grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel) # b, dim, n, n_knn

        #attention
        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n*up_factor, n_knn
        attention = torch.softmax(attention, -1)

        # value = value.reshape((b, -1, n, 1)) + pos_embedding  #
        value= pointnet2_utils.grouping_operation(value, idx_knn) + pos_embedding  # b, dim, n, n_knn
        value =self.upsample1(value) # b, dim, n*up_factor, n_knn

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n*up_factor
        y = self.conv_end(agg) # b, out_dim, n*up_factor

        identity = self.upsample2(identity) # b, out_dim, n*up_factor

        return y + identity
class FeaExtract(torch.nn.Module):
    def __init__(self, channel=64):
        super(FeaExtract, self).__init__()
        self.channel = channel
        ### MLP(2048,64)
        self.conv1 = torch.nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = cross_transformer((channel)*8,channel*8)

        self.relu = torch.nn.GELU()

    def forward(self, points):
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)   # B,64,2048

        # GDP
        idx_0 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = pointnet2_utils.gather_operation(x0, idx_0) # B,64,512
        points = pointnet2_utils.gather_operation(points, idx_0) # B,3,512
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1) # B,128,512
        # SFA
        x1 = self.sa1_1(x1,x1).contiguous() #B,128,512
        # GDP
        idx_1 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = pointnet2_utils.gather_operation(x1, idx_1) # B,128,256
        points = pointnet2_utils.gather_operation(points, idx_1) # B,3,256
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N  B,128,256
        x2 = torch.cat([x_g1, x2], dim=1)  # B,256,256
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous() # B,256,256
        # GDP
        idx_2 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = pointnet2_utils.gather_operation(x2, idx_2)   # B,256,128
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4  B,256,128
        x3 = torch.cat([x_g2, x3], dim=1) # B,512,128
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous() # B,512,128
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1) # B,512,1
        # print("x_g:",x_g.size())

        return x_g
class CoraseGE(torch.nn.Module):
    def __init__(self, channel=64):
        super(CoraseGE, self).__init__()
        self.channel = channel
        ### MLP(2048,64)
        self.conv1 = torch.nn.Conv1d(1536, 512, kernel_size=1)
        self.relu = torch.nn.GELU()

        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out = torch.nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = torch.nn.Conv1d(channel*2, 64, kernel_size=1)
        self.ps = torch.nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = torch.nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = torch.nn.Conv1d(channel*8, channel*8, kernel_size=1)


    def forward(self, points, feature):
        batch_size, _, N = points.size()
        # MLP + Reshape
        x_g=self.conv1(feature)  # B, 512, 1
        x = self.relu(self.ps_adj(x_g)) # B,512,1
        x = self.relu(self.ps(x))  # B,64,128
        x = self.relu(self.ps_refuse(x)) # B,512,128
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*2,N//4)
        # Reshape
        coarse = self.conv_out(self.relu(self.conv_out1(x2_d))) # B,3,256
        # 与输入拼接
        coarse = torch.concat([coarse, points],dim=2)
        coarse_idx = pointnet2_utils.furthest_point_sample(coarse.transpose(2,1).contiguous(), N//4) 
        coarse = pointnet2_utils.gather_operation(coarse, coarse_idx)
        return coarse
class FineGE(torch.nn.Module):
    def __init__(self, num_pred=16384):
        super(FineGE, self).__init__()
        self.num_pred = num_pred
        self.fc1 = torch.nn.Sequential(
              torch.nn.Linear(1539, 1539),
              torch.nn.ReLU()
          )
        self.fc2 = torch.nn.Sequential(
              torch.nn.Linear(1539, 384),
              torch.nn.ReLU()
          )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(384, 96),
            torch.nn.ReLU()
          )
        self.fc4 = torch.nn.Linear(96, 12)

    def forward(self, features, sparse_cloud):

        point_features = self.fc1(features)
        # print(point_features.size())    # torch.Size([batch_size, num_pred//32, 1539])
        point_features = self.fc2(point_features)
        # print(point_features.size())    # torch.Size([batch_size, num_pred//32, 384])
        point_features = self.fc3(point_features)
        # print(point_features.size())    # torch.Size([batch_size, num_pred//32, 96])
        point_features = self.fc4(point_features)
        # print(point_features.size())    # torch.Size([batch_size, num_pred//32, 12])
        point_offset=point_features.view(-1, self.num_pred//8, 3)
        # point_offset = self.fc14(point_features).view(-1, num_crop, 3)
        fine_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 4, 1).reshape(-1,self.num_pred//8,3) + point_offset

        return fine_cloud


class DenseGE(torch.nn.Module):
    def __init__(self, dim_feat=512, up_factor=4, dim=256):
        super(DenseGE, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])
       
        self.sa=cross_transformer(128,256)
        self.skip_transformer = SkipTransformer(dim,dim, dim=64,up_factor=self.up_factor, attn_channel=True)
  
        self.up_sampler = torch.nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256+256, hidden_dim=256, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)  # (B, 128, N_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)   # (B, 128+128+1536, N_prev)
        Q = self.mlp_2(feat_1) # (B, 128, N_prev)

        H=self.sa(Q,Q) # B ,256, N_prev
        
        feat_child = self.skip_transformer(pcd_prev,H,H) # (B, 256, N_prev * up_factor)
       

      
        H_up = self.up_sampler(H)   # (B, 256, N_prev * up_factor)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))  # (B, 128, N_prev * up_factor)

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr)))  # (B, 3, N_prev * up_factor)
        pcd_new = self.up_sampler(pcd_prev)  # (B, 3, N_prev * up_factor)
        pcd_new = pcd_new + delta

        return pcd_new
@MODELS.register_module()
class GRFormer(torch.nn.Module):
    def __init__(self, config):
        super(GRFormer, self).__init__()
        self.num_pred = config.num_pred
        self.gridding_scale = [config.gridding_loss_scales]
        self.gridding_alpha = [config.gridding_loss_alphas]
        self.loss_lambda = 0.

        ### feature extraction
        self.feature_extractor = FeaExtract()
        ### Gridding、3DCNN、GriddingReverse
        self.gridding = Gridding(scale=64)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(16384, 2048),
            torch.nn.ReLU()
        )
   
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU()
        )
        self.cor_ge = CoraseGE()
        self.mlp_refine = DenseGE(dim_feat=1536, up_factor=4)
        self.refine = DenseGE(dim_feat=1536, up_factor=8)
       
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_1 = ChamferDistanceL2()
        self.loss_func_2 = GriddingLoss(
                        self.gridding_scale,
                        self.gridding_alpha)

    def get_loss(self, ret, gt, epoch=0):
        loss_coarse = self.loss_func_1(ret[0], gt) 
        loss_fine = self.loss_func_1(ret[1], gt)
        loss_dense = self.loss_func_1(ret[2], gt)
        return loss_coarse, loss_fine, loss_dense

    def forward(self, xyz):
        # NOTE: # Avoid overflow while gridding on ShapeNet55
        partial_cloud = xyz * 0.5
        # print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        ### feature extraction
        feature_ext=self.feature_extractor(partial_cloud.transpose(2,1).contiguous()) # B 512 1
        ### Gridding、3DCNN、GriddingReverse
        pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 64, 64, 64)
        # print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 64, 64, 64]) 1个特征通道，64*64*64的3D网格
        pt_features_32_l = self.conv1(pt_features_64_l)
        # print(pt_features_32_l.size())  # torch.Size([batch_size, 32, 32, 32, 32]) 增加特征通道到32，减少空间维度，32*32*32的3D网格
        pt_features_16_l = self.conv2(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_8_l = self.conv3(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_4_l = self.conv4(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 16384))
        # print(features.size())          # torch.Size([batch_size, 2048])
        pt_features_4_r=self.fc6(features)
        # print("CNN feature:", pt_features_4_r.size())          # torch.Size([batch_size, 1024])
        pt_features_4_r = pt_features_4_r.unsqueeze(-1)
        # print("CNN feature2:", pt_features_4_r.size())          # torch.Size([batch_size, 1024, 1 ])
        feat_cat=torch.cat([pt_features_4_r, feature_ext],dim=1) # B,1536,1
        
        sparse_cloud = self.cor_ge(partial_cloud.transpose(2,1).contiguous(), feat_cat)
        # print("cor_ge sparse_cloud:",sparse_cloud.size())      # torch.Size([batch_size, 3, 512])
        
        fine_cloud = self.mlp_refine(sparse_cloud, feat_cat)
        
        dense_cloud = self.refine(fine_cloud, feat_cat)
        dense_cloud = dense_cloud.transpose(2,1).contiguous() ### B,16384,3
        fine_cloud=fine_cloud.transpose(2,1).contiguous()
        sparse_cloud = sparse_cloud.transpose(2,1).contiguous()  # B,512,3
        ret = (sparse_cloud * 2.0, fine_cloud * 2.0, dense_cloud * 2.0)
               
        return ret
