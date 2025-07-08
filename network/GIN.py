import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
from netTools import build_jsp_graph
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class GINEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            mlp = MLP(input_dim if i == 0 else hidden_dim, hidden_dim)
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        graph_repr = global_mean_pool(x, batch)  # 聚合为图表示
        return self.output_proj(graph_repr)


class GINsolver(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = GINEncoder(input_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出一个值，如makespan
        )

    def forward(self, x):

        with torch.no_grad():


            data_list = [build_jsp_graph(fm, 5) for fm in x]
            graph = Batch.from_data_list(data_list)
        graph_embedding = self.encoder(graph.x, graph.edge_index, graph.batch)
        return self.net(graph_embedding)  # 输出大小为 (batch_size,)
