import torch
import numpy as np
import math
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN


### pytorch version of query_ball_point,slower
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: float, radius of the ball.
        nsample: int, maximum number of points in the ball.
        xyz: torch.Tensor, input points, shape (B, N, 3).
        new_xyz: torch.Tensor, query points, shape (B, S, 3).
    Return:
        group_idx: torch.Tensor, indices of the grouped points, shape (B, S, nsample)
        pts_cnt: torch.Tensor, number of unique points in each group, shape (B, S)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    pts_cnt = torch.sum(~mask, dim=-1)
    return group_idx, pts_cnt
### pytorch version of group_point,slower
def group_point(points, idx):
    """
    Input:
        points: torch.Tensor, input points data, shape (B, N, C)
        idx: torch.Tensor, sample index data, shape (B, S, K)
    Return:
        grouped_points: torch.Tensor, grouped points data, shape (B, S, K, C)
    """
    B, N, C = points.shape
    _, S, K = idx.shape
    device = points.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    points = points.view(B*N, -1)[idx, :]
    points = points.view(B, S, K, C)
    return points

### 排斥损失
def get_repulsion_loss(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint, 3)
    pred = pred.contiguous()
    idx= pointnet2_utils.ball_query(radius, nsample, pred, pred) # (batch_size, npoint, nsample)

    grouped_pred = pointnet2_utils.grouping_operation(pred.transpose(1, 2).contiguous(), idx).permute(0,2,3,1)  # (batch_size, npoint, nsample, 3)
    print("group:",grouped_pred.shape)
    grouped_pred -= pred.unsqueeze(2)

    ## get the repulsion loss
    h = 0.03
    dist_square = torch.sum(grouped_pred ** 2, dim=-1)
    dist_square, idx = torch.topk(-dist_square, 5, dim=-1)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = torch.maximum(dist_square, torch.tensor(1e-12).to(pred.device))
    dist = torch.sqrt(dist_square)
    weight = torch.exp(-dist_square / h ** 2)
    repulsion_loss = torch.mean(radius - dist * weight)
    return repulsion_loss

def get_pairwise_distance(batch_features):
    """Compute pairwise distance of a point cloud using PyTorch.

    Args:
      batch_features: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    if batch_features.dim() == 2:  # just two dimensions
        batch_features = batch_features.unsqueeze(0)

    batch_features_transpose = batch_features.transpose(2, 1)

    batch_features_inner = torch.matmul(batch_features, batch_features_transpose)

    batch_features_inner = -2 * batch_features_inner
    batch_features_square = torch.sum(batch_features ** 2, dim=-1, keepdim=True)

    batch_features_square_transpose = batch_features_square.transpose(2, 1)

    return batch_features_square + batch_features_inner + batch_features_square_transpose

def get_knn_dis(queries, pc, k):
    """
    Compute k nearest neighbors distances for queries from point cloud pc using KNN_CUDA on GPU.

    Args:
      queries: tensor [M, C], assumed to be on GPU
      pc: tensor [P, C], assumed to be on GPU
      k: int, number of nearest neighbors to find

    Returns:
      distances: tensor, distances of the k nearest neighbors [M, k], on GPU
    """
    # 确保输入是(batch_size, num_points, 3)的形状，这里我们假设C=3且batch_size=1
    queries = queries.unsqueeze(0)  # [M, C] -> [1, M, C]
    pc = pc.unsqueeze(0)  # [P, C] -> [1, P, C]

    # 创建KNN对象并执行搜索
    knn = KNN(k=k, transpose_mode=False)
    dis, _ = knn(pc, queries)

    # dis是距离的平方，取平方根得到实际距离
    dis = torch.sqrt(dis)

    # 移除batch维度并返回结果
    return dis.squeeze(0).to(queries.device)
# def knn_point(k, xyz1, xyz2):
#     """
#     Input:
#         k: int, number of k in k-nn search
#         xyz1: torch.Tensor, (batch_size, ndataset, c) input points
#         xyz2: torch.Tensor, (batch_size, npoint, c) query points
#     Output:
#         val: torch.Tensor, (batch_size, npoint, k) L2 distances
#         idx: torch.Tensor, (batch_size, npoint, k) indices to input points
#     """
#     xyz1 = xyz1.unsqueeze(2)  # (batch_size, ndataset, 1, c)
#     xyz2 = xyz2.unsqueeze(1)  # (batch_size, 1, npoint, c)
#     dist = torch.sum((xyz1 - xyz2) ** 2, dim=-1)  # (batch_size, ndataset, npoint)
#     val, idx = dist.topk(k, dim=1, largest=False, sorted=True)  # (batch_size, npoint, k)
#     return -val, idx

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int, number of k in k-nn search
        xyz1: torch.Tensor, (batch_size, ndataset, c) input points
        xyz2: torch.Tensor, (batch_size, npoint, c) query points
    Output:
        val: torch.Tensor, (batch_size, npoint, k) L2 distances
        idx: torch.Tensor, (batch_size, npoint, k) indices to input points
    '''
    # 扩展维度
    xyz1 = xyz1.unsqueeze(2)  # (batch_size, ndataset, 1, c)
    xyz2 = xyz2.unsqueeze(1)  # (batch_size, 1, npoint, c)
    
    # 计算L2距离
    dist = torch.sum((xyz1 - xyz2) ** 2, dim=-1)  # (batch_size, ndataset, npoint)
    
    # 查找k近邻
    val, idx = dist.topk(k, dim=1, largest=False, sorted=True)  # (batch_size, npoint, k)
    
    return val, idx
def py_uniform_loss(points, idx, pts_cnt, radius):
    B, N, C = points.shape
    _, npoint, nsample = idx.shape
    uniform_vals = []
    for i in range(B):
        point = points[i]
        for j in range(npoint):
            number = pts_cnt[i, j].item()
            coverage = ((number - nsample) ** 2) / nsample
            if number < 5:
                uniform_vals.append(coverage)
                continue
            _idx = idx[i, j, :number]
            disk_point = point[_idx]
            if disk_point.shape[0] < 0:
                pair_dis = get_pairwise_distance(disk_point)  # (batch_size, num_points, num_points)
                nan_valid = torch.where(pair_dis < 1e-7)
                pair_dis[nan_valid] = 0
                pair_dis = torch.squeeze(pair_dis, axis=0)
                pair_dis = torch.sort(pair_dis, axis=1)[0]
                shortest_dis = torch.sqrt(pair_dis[:, 1])
            else:
                shortest_dis = get_knn_dis(disk_point, disk_point, 2)
                shortest_dis = shortest_dis[:, 1]
            disk_area = math.pi * (radius ** 2) / disk_point.shape[0]
            expect_d = torch.sqrt(2 * disk_area / 1.732)  # using hexagon
            dis = ((shortest_dis - expect_d) ** 2) / expect_d
            uniform_val = coverage * torch.mean(dis)

            uniform_vals.append(uniform_val)

    uniform_dis = torch.tensor(uniform_vals, dtype=torch.float32)
    uniform_dis = torch.mean(uniform_dis)
    return uniform_dis
### 均匀损失 whole Version，slower
def get_uniform_loss2(pcd, percentages=[0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015], radius=1.0):
    B, N, C = pcd.size()
    npoint = int(N * 0.05)
    loss = []
    for p in percentages:
        nsample = int(N * p)
        r = math.sqrt(p * radius)
        new_xyz_idx = pointnet2_utils.furthest_point_sample(pcd, npoint)
        new_xyz = pointnet2_utils.gather_operation(pcd.transpose(1, 2).contiguous(), new_xyz_idx).transpose(1,2).contiguous() # (batch_size, npoint, 3)
        idx, pts_cnt = pointnet2_utils.ball_query(r, nsample, pcd, new_xyz)  #(batch_size, npoint, nsample)

        uniform_val = py_uniform_loss(pcd, idx, pts_cnt, r)

        loss.append(uniform_val * math.sqrt(p * 100))
    return torch.sum(torch.stack(loss)) / len(percentages)
### 均匀损失  simplfied version, faster
def get_uniform_loss(pcd, percentages=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0):
    B, N, C = pcd.size()
    npoint = int(N * 0.05)
    loss = []
    pcd = pcd.contiguous()
    for p in percentages:
        nsample = int(N * p)
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) * p / nsample
        new_xyz_idx = pointnet2_utils.furthest_point_sample(pcd, npoint)
        new_xyz = pointnet2_utils.gather_operation(pcd.transpose(1, 2).contiguous(), new_xyz_idx).transpose(1, 2).contiguous()
        # idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)
        idx = pointnet2_utils.ball_query(r, nsample, pcd, new_xyz)
        expect_len = torch.sqrt(torch.tensor(disk_area))  # using square

        # grouped_pcd = group_point(pcd, idx)
        grouped_pcd = pointnet2_utils.grouping_operation(pcd.transpose(1, 2).contiguous(), idx).permute(0, 2, 3, 1)
        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, dim=1), dim=0)

        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = torch.sqrt(torch.abs(uniform_dis + 1e-8))
        uniform_dis = torch.mean(uniform_dis, dim=[-1])
        uniform_dis = torch.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = torch.reshape(uniform_dis, [-1])

        mean = torch.mean(uniform_dis)
        mean = mean * math.pow(p * 100, 2)

        loss.append(mean)
    return torch.sum(torch.stack(loss)) / len(percentages)