import torch
import torch.nn as nn
import torch.nn.functional as F

def concave_activation(x):
    return torch.sqrt(torch.clamp(x, min=1e-6))

class DSFDeepSet(nn.Module):
    def __init__(self, input_dim=15, embed_dim=128, hidden_dims=[64, 32]):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, embed_dim, bias=False),
            nn.ReLU(),
        )
        self.dsf_layers = nn.ModuleList()
        dims = [embed_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1], bias=False)
            nn.init.uniform_(layer.weight, a=0.01, b=0.1)
            self.dsf_layers.append(layer)

        self.output_layer = nn.Linear(dims[-1], 1, bias=False)
        nn.init.uniform_(self.output_layer.weight, a=0.01, b=0.1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # x: (batch_size, set_size, feature_dim)

        x = self.phi(x)  # shape: (batch, set_size, embed_dim)
        x = torch.sum(x, dim=1)  # permutation invariant sum-pooling: (batch, embed_dim)

        # DSF-style layers with concave activations
        for layer in self.dsf_layers:
            layer.weight.data.clamp_(min=0)
            x = concave_activation(layer(x))

        self.output_layer.weight.data.clamp_(min=0)
        out = concave_activation(self.output_layer(x))
        return out
