"""Numpy version of euclidean distance, shortest distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from qpth.qp import QPFunction

def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        
        return dist

#old dist
def shortest_dist(dist_mat):
    """Parallel version.
  Args:
    dist_mat: numpy array, available shape
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`
      1) scalar
      2) numpy array, with shape [N]
      3) numpy array with shape [*]
  """
    m, n = dist_mat.shape[:2]
  
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]

            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]
    # I ran into memory disaster when returning this reference! I still don't
    dist = dist[-1, -1].copy()

    return dist

def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = torch.tensor(cost_matrix)  
    cost_matrix = cost_matrix.detach().cpu().numpy()

    #print(cost_matrix.shape) #(8,8)

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    #print(cost_matrix.shape)
    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow
# new dist
def emd_inference_opencv_test(dist_mat,weight1,weight2):
    distance_list = []
    flow_list = []
    #print(dist_mat.shape)
    for i in range (dist_mat.shape[0]):   #32
        cost,flow=emd_inference_opencv(dist_mat[i],weight1[i],weight2[i]) #(8,8)
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    dist = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list,dim=0).cuda().double()
  
    return dist

def unaligned_dist(dist_mat):
    """Parallel version.
    Args:
      dist_mat: numpy array, available shape
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
      dist: three cases corresponding to `dist_mat`
        1) scalar
        2) numpy array, with shape [N]
        3) numpy array with shape [*]
    """

    m = dist_mat.shape[0]
    dist = np.zeros_like(dist_mat[0])
    for i in range(m):
        dist[i] = dist_mat[i][i]
    dist = np.sum(dist, axis=0).copy()
    
    return dist


def meta_local_dist(x, y, aligned):
    """
  Args:
    x: numpy array, with shape [m, d]
    y: numpy array, with shape [n, d]
  Returns:
    dist: scalar
  """
    eu_dist = compute_dist(x, y, 'euclidean')
    dist_mat = (np.exp(eu_dist) - 1.) / (np.exp(eu_dist) + 1.)
    #dist_mat = torch.Tensor(dis_mat)  # tu comment
    if aligned:
        #dist = shortest_dist(dist_mat[np.newaxis])[0]
        #dist = emd_inference_qpth(dist_mat[np.newaxis], weight1, weight2, form= form)[0]
        dist = emd_inference_opencv(dist_mat[np.newaxis],weight1,weight2)[0]
    else:
        dist = unaligned_dist(dist_mat[np.newaxis])[0]
    
    return dist


# Tooooooo slow!
def serial_local_dist(x, y):
    """
  Args:
    x: numpy array, with shape [M, m, d]
    y: numpy array, with shape [N, n, d]
  Returns:
    dist: numpy array, with shape [M, N]
  """
    M, N = x.shape[0], y.shape[0]
    dist_mat = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            dist_mat[i, j] = meta_local_dist(x[i], y[j])
    
    return dist_mat

def parallel_local_dist(x, y, aligned):
    """GPU-accelerated parallel version.
    Args:
        x: numpy array, with shape [M, m, d]
        y: numpy array, with shape [N, n, d]
    Returns:
        dist: numpy array, with shape [M, N]
    """
    # Convert numpy arrays to PyTorch tensors and move to GPU
    x_tensor = torch.from_numpy(x).float().cuda()
    y_tensor = torch.from_numpy(y).float().cuda()
    
    M, m, d = x.shape  # 158 8 2048
    N, n, d = y.shape  # 158 8 2048
    
    if aligned:
        # Initialize the distance matrix on GPU
        dist_mat = torch.zeros((M, N), device='cuda')
        
        if m == n:
            for i in range(M):
              
                diffs = x_tensor[i:i+1].expand(N, m, d) - y_tensor
                squared_diffs = torch.sum(diffs**2, dim=2)
                sequence_dists = torch.sqrt(squared_diffs)
                dist_mat[i] = torch.mean(sequence_dists, dim=1)
        else:
            # For sequences of different lengths, we need a different approach
            for i in range(M):
                for j in range(N):
                    xi_expanded = x_tensor[i].unsqueeze(1)  # [m, 1, d]
                    yj_expanded = y_tensor[j].unsqueeze(0)  # [1, n, d]
                    
                    # Calculate squared Euclidean distances
                    dists = torch.sum((xi_expanded - yj_expanded)**2, dim=2)  # [m, n]
                    dists = torch.sqrt(dists)
                    
                    # Take the mean of all pairwise distances
                    dist_mat[i, j] = torch.mean(dists)

        return dist_mat.cpu().numpy()
    else:
        # Reshape tensors
        x_reshaped = x_tensor.reshape(M * m, d)
        y_reshaped = y_tensor.reshape(N * n, d)
        
        x_norm = torch.sum(x_reshaped**2, dim=1, keepdim=True)
        y_norm = torch.sum(y_reshaped**2, dim=1, keepdim=True)
        
        xy = torch.mm(x_reshaped, y_reshaped.t())
        
        dist_squared = x_norm + y_norm.t() - 2.0 * xy
        dist_squared = torch.clamp(dist_squared, min=0.0)  # Ensure non-negative
        dist_mat = torch.sqrt(dist_squared)
        
        dist_mat = (torch.exp(dist_mat) - 1.0) / (torch.exp(dist_mat) + 1.0)
        
        dist_mat = dist_mat.reshape(M, m, N, n).permute(1, 3, 0, 2)
        
        # Implement unaligned_dist on GPU
        m_dim = dist_mat.shape[0]
        dist = torch.zeros_like(dist_mat[0])
        for i in range(m_dim):
            dist[i] = dist_mat[i, i]
        dist = torch.sum(dist, dim=0)
        
        return dist.cpu().numpy()

# def parallel_local_dist(x, y, aligned):
#     """Parallel version.
#   Args:
#     x: numpy array, with shape [M, m, d]
#     y: numpy array, with shape [N, n, d]
#   Returns:
#     dist: numpy array, with shape [M, N]
#   """
#     M, m, d = x.shape #158 8 2048
#     N, n, d = y.shape #158 8 2048
#     x = x.reshape([M * m, d])
#     y = y.reshape([N * n, d])
    
#     # Create weight tensors
#     k = torch.tensor([0.4,0.5,0.6,0.7,0.8,0.9,0.9,0.8,0.7,0.6,0.5,0.4])
#     weight1 = k.repeat(M, 1).cuda()  # Adjust size to match M
#     weight2 = k.repeat(N, 1).cuda()  # Adjust size to match N
    
#     # shape [M * m, N * n]
#     dist_mat = compute_dist(x, y, type='euclidean')
#     dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)
    
#     # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
#     dist_mat = dist_mat.reshape([M, m, N, n]).transpose([1, 3, 0, 2])
    
#     # shape [M, N]
#     if aligned:
#         # Block compute EMD distance
#         dist_mat = dist_mat.transpose([2, 3, 0, 1])  # [M, N, m, n]
#         dist_mat = dist_mat.reshape(M * N, m, n)  # Batch of cost matrices
#         weight1_batch = weight1[:m].repeat(M * N, 1)
#         weight2_batch = weight2[:n].repeat(M * N, 1)
#         dist = emd_inference_opencv_test(dist_mat, weight1_batch, weight2_batch)
#         dist = dist.cpu().numpy().reshape(M, N)

#         # Block compute EMD distance
#         #dist = shortest_dist(dist_mat)
#     else:
#         dist_mat = unaligned_dist(dist_mat)
#         dist = dist_mat
    
#     return dist
  
def local_dist(x, y, aligned):
    if (x.ndim == 2) and (y.ndim == 2):
        return meta_local_dist(x, y, aligned)
    elif (x.ndim == 3) and (y.ndim == 3):
        return parallel_local_dist(x, y, aligned)
    else:
        raise NotImplementedError('Input shape not supported.')


def low_memory_matrix_op(
        func,
        x, y,
        x_split_axis, y_split_axis,
        x_num_splits, y_num_splits,
        verbose=False, aligned=True):
    """
  For matrix operation like multiplication, in order not to flood the memory
  with huge data, split matrices into smaller parts (Divide and Conquer).

  Note:
    If still out of memory, increase `*_num_splits`.

  Args:
    func: a matrix function func(x, y) -> z with shape [M, N]
    x: numpy array, the dimension to split has length M
    y: numpy array, the dimension to split has length N
    x_split_axis: The axis to split x into parts
    y_split_axis: The axis to split y into parts
    x_num_splits: number of splits. 1 <= x_num_splits <= M
    y_num_splits: number of splits. 1 <= y_num_splits <= N
    verbose: whether to print the progress

  Returns:
    mat: numpy array, shape [M, N]
  """

    if verbose:
        import sys
        import time
        printed = False
        st = time.time()
        last_time = time.time()

    mat = [[] for _ in range(x_num_splits)]
    for i, part_x in enumerate(
            np.array_split(x, x_num_splits, axis=x_split_axis)):
    
        for j, part_y in enumerate(
                np.array_split(y, y_num_splits, axis=y_split_axis)):
            #print(part_x.shape)(158, 8, 2048)
            #print(part_y.shape)(158, 8, 2048)
            part_mat = func(part_x, part_y, aligned)
            mat[i].append(part_mat)
            if verbose:
                if not printed:
                    printed = True
                else:
                    # Clean the current line
                    sys.stdout.write("\033[F\033[K")
                print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
                    .format(i + 1, j + 1, x_num_splits, y_num_splits,
                            time.time() - last_time, time.time() - st))
                last_time = time.time()
        mat[i] = np.concatenate(mat[i], axis=1)
    mat = np.concatenate(mat, axis=0)
    return mat


def low_memory_local_dist(x, y, aligned=True):
    print('Computing local distance...')
    x_num_splits = int(len(x) / 200) + 1
    y_num_splits = int(len(y) / 200) + 1
    z = low_memory_matrix_op(local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True, aligned=aligned)
    return z