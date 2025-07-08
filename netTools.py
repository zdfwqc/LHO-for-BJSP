import numpy as np
import torch
from torch_geometric.data import Data
def getJobFeature(orginData,PTmask,curSelected):
    numofMachines = len(orginData[0][0])
    numofJobs = len(curSelected)
    jobs_features = [[] for i in range(numofJobs)]
    for i in range(numofJobs):
        for j in range(numofMachines):
            if PTmask[curSelected[i]][j] == -1:
                # 已经完成的工序
                # 机器设为0是一个特殊值 正常机器不会为0
                jobs_features[i].append(0)
                jobs_features[i].append(0)
                jobs_features[i].append(0)
            elif PTmask[curSelected[i]][j] >0:
                # 未完全完成的工序 强制要求开头开始
                jobs_features[i].append(PTmask[curSelected[i]][j])
                jobs_features[i].append(orginData[1][curSelected[i]][j])
                jobs_features[i].append(1)
            else:
                # 正常的工序
                jobs_features[i].append(orginData[0][curSelected[i]][j])
                jobs_features[i].append(orginData[1][curSelected[i]][j])
                jobs_features[i].append(0)
    return jobs_features


def build_jsp_graph(feature_matrix, M):
    n = feature_matrix.shape[0]  # 工件数
    op_features = []  # 每个工序的3维特征
    op_id = 0
    edge_index = []
    machine_to_ops = {}

    for i in range(n):  # 遍历工件
        for k in range(M):
            t = feature_matrix[i, 3*k]
            m = feature_matrix[i, 3*k + 1]
            f = feature_matrix[i, 3*k + 2]
            op_features.append([t, m, f])

            # 同一工件的顺序约束
            if k > 0:
                edge_index.append([op_id - 1, op_id])  # i-th工件中的k-1 → k

            # 同机器的冲突
            m = int(m)
            if m not in machine_to_ops:
                machine_to_ops[m] = []
            machine_to_ops[m].append(op_id)

            op_id += 1

    # 加入机器冲突边（无向边）
    for ops in machine_to_ops.values():
        for i in range(len(ops)):
            for j in range(i+1, len(ops)):
                edge_index.append([ops[i], ops[j]])
                edge_index.append([ops[j], ops[i]])

    x = torch.tensor(op_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch = torch.zeros(x.size(0), dtype=torch.long)  # 假设是一个图

    return Data(x=x,edge_index=edge_index, batch=batch)
