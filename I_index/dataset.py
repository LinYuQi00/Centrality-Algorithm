import numpy as np

def generate_flow_distance_matrices(num_flows, num_destinations):
    # 初始化流量矩阵和距离矩阵
    T = np.zeros((num_flows, num_destinations))
    D = np.zeros((num_flows, num_destinations))

    # Cluster #1: 50 flows
    for i in range(50):
        T[i, 0] = 1  # 到目的地0的流量
        radius = np.random.uniform(5, 15)
        angle = np.random.uniform(0, 2 * np.pi)
        origin_x = 25 + radius * np.cos(angle)
        origin_y = 75 + radius * np.sin(angle)
        D[i, 0] = np.sqrt((origin_x - 25)**2 + (origin_y - 75)**2)

    # Cluster #2: 50 flows
    for i in range(50, 100):
        T[i, 1] = 1  # 到目的地1的流量
        radius = np.random.uniform(0, 5)
        angle = np.random.uniform(0, 2 * np.pi)
        origin_x = 75 + radius * np.cos(angle)
        origin_y = 75 + radius * np.sin(angle)
        D[i, 1] = np.sqrt((origin_x - 75)**2 + (origin_y - 75)**2)

    # Cluster #3: 30 flows
    for i in range(100, 130):
        T[i, 2] = 1  # 到目的地2的流量
        radius = np.random.uniform(0, 5)
        angle = np.random.uniform(0, 2 * np.pi)
        origin_x = 25 + radius * np.cos(angle)
        origin_y = 25 + radius * np.sin(angle)
        D[i, 2] = np.sqrt((origin_x - 25)**2 + (origin_y - 25)**2)

    # Cluster #4: 30 flows (with one long flow)
    for i in range(130, 160):
        if i < 159:  # 前29个流量
            T[i, 3] = 1  # 到目的地3的流量
            radius = np.random.uniform(0, 5)
            angle = np.random.uniform(0, 2 * np.pi)
            origin_x = 75 + radius * np.cos(angle)
            origin_y = 25 + radius * np.sin(angle)
            D[i, 3] = np.sqrt((origin_x - 75)**2 + (origin_y - 25)**2)
        else:  # 最后一个极长流量
            T[i, 3] = 1  # 长流量
            long_flow_distance = abs(np.sum(D[:1])-np.sum(D[:3]))
            D[i, 3] = long_flow_distance  # 设定一个极长的距离

    return T, D

