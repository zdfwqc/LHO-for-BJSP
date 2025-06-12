import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class DeepSetNet(nn.Module):
    def __init__(self, input_dim=10, phi_hidden=128, rho_hidden=128, output_dim=1, dropout_rate=0.3):
        """
        (batch_size, N, input_dim)
        input_dim: 输入维度
        phi_hidden: phi网络的隐藏层大小
        rho_hidden: rho网络的隐藏层大小
        output_dim: 输出维度
        dropout_rate: dropout比率
        """
        super(DeepSetNet, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 2*phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2*phi_hidden, phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden, 2*rho_hidden),
            nn.BatchNorm1d(2*rho_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2*rho_hidden, rho_hidden), 
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rho_hidden, output_dim)
        )

        
    def forward(self, x):
        # 确保输入维度正确 (batch_size, n, 15)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  
        phi_x = self.phi(x)  
        pooled = phi_x.mean(dim=1)  
        out = self.rho(pooled)
        return out