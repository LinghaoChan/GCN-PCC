import torch


def knn(X, k):
    # print(x.shape)
    inner_pro = -2*torch.matmul(X.transpose(2, 1), X)
    X_square = torch.sum(X**2, dim=1, keepdim=True)
    pairwise_distance = -X_square - inner_pro - X_square.transpose(2, 1)
    index = pairwise_distance.topk(k=k, dim=-1)[1]
    # print(idx[0][1])
    return index


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = feature.permute(0, 3, 1, 2).contiguous()
    return feature
