# -*- coding: utf-8 -*-
# Author: LinYuqi
# Email: lyqllfw@gmail.com
# Date: 2024-10-29
# Description: I-index to quantifying an urban location's irreplaceability
# paper: Wang X. et al. (2021). “I-index for quantifying an urban location's irreplaceability.” Computers, Environment and Urban Systems 90 (2021) 101711.
# Note:

import numpy as np
from dataset import generate_flow_distance_matrices

def _inner_calculate_nj(T_mat: np.ndarray)->np.ndarray:
    """
    计算每个目的地的总流量 Nj。

    参数:
    T (numpy.ndarray): OD 旅行矩阵 (m x n)。

    返回:
    numpy.ndarray: 每个目的地的总流量 Nj (长度 n)。
    """
    return np.sum(T_mat, axis=0)

def _inner_create_fj(T_mat: np.ndarray, D_mat: np.ndarray, j: int) -> np.ndarray:
    """
    为指定目的地创建流距离向量 Fj。

    参数:
    T_mat (numpy.ndarray): OD 旅行矩阵 (m x n)。
    D_mat (numpy.ndarray): 距离矩阵 (m x n)。
    j (int): 指定的目的地索引。

    返回:
    numpy.ndarray: 指定目的地的流距离向量 Fj。
    """
    m = T_mat.shape[0]
    fj = []
    for i in range(m):
        fj.extend([D_mat[i, j]] * int(T_mat[i, j]))  # 根据流量重复距离

    return np.array(fj)

def _inner_calculate_mj(T_mat: np.ndarray, D_mat: np.ndarray) -> np.ndarray:
    """
    计算每个目的地流距离的中位数 Mj。

    参数:
    D_mat (numpy.ndarray): 距离矩阵 (m x n)。
    T_mat (numpy.ndarray): OD 旅行矩阵 (m x n)。

    返回:
    numpy.ndarray: 每个目的地的流距离中位数 Mj (长度 n)。
    """
    n = D_mat.shape[1]
    mj_list = []

    for j in range(n):
        fj = _inner_create_fj(T_mat, D_mat, j)  # 获取流距离向量
        mj_list.append(np.median(fj))  # 计算中位数并存储

    return np.array(mj_list)

def _inner_calculate_i_index(T_mat: np.ndarray, D_mat: np.ndarray) -> np.ndarray:
    """
    计算每个目的地的 I-index，内嵌 alpha 的计算。
    """
    Nj = _inner_calculate_nj(T_mat)  # 计算每个目的地的总流量
    Mj = _inner_calculate_mj(T_mat, D_mat)  # 计算每个目的地的流距离中位数

    # 计算 alpha
    alpha = np.median(Mj) / np.median(Nj)

    n = T_mat.shape[1]
    i_index = np.zeros(n)

    for j in range(n):
        fj = _inner_create_fj(T_mat, D_mat, j)  # 获取流距离向量
        nj = int(Nj[j])  # 当前目的地的总流量

        # 使用 NumPy 向量化计算
        fj_sorted = np.sort(fj)[::-1]  # 按降序排序
        indices = np.arange(1, nj + 1)  # kj 的值
        S_kj = fj_sorted[:nj]  # 取前 nj 个值

        # 查找满足条件的最大 kj
        valid_kj = S_kj >= alpha * indices
        if np.any(valid_kj):
            i_index[j] = np.max(indices[valid_kj])  # 更新 I-index

    return i_index

def calculate_i_index(T_mat: np.ndarray, D_mat: np.ndarray)->np.ndarray:
    return _inner_calculate_i_index(T_mat,D_mat)

# 示例使用
if __name__ == "__main__":

    # T = np.array([[10, 5, 0],
    #               [3, 0, 8],
    #               [0, 12, 1]])
    #
    # D = np.array([[1, 2, 0],
    #               [4, 0, 3],
    #               [0, 5, 1]])
    num_flows = 160
    num_destinations = 4
    T, D = generate_flow_distance_matrices(num_flows, num_destinations)
    print(D)
    i_index = calculate_i_index(T,D)
    print(i_index)


