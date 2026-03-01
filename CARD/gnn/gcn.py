import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FeatureFusion(torch.nn.Module):
    def __init__(self,use_softmax=True):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1,dtype=float))
        self.beta = torch.nn.Parameter(torch.tensor(1,dtype=float))
        self.use_softmax = use_softmax

    def forward(self,feature_1,feature_2):
        if self.use_softmax:
            weight = torch.softmax(torch.stack([self.alpha,self.beta]),dim=0)
            alpha_w,beta_w = weight[0],weight[1]
        else:
            alpha_w = self.alpha
            beta_w = self.beta
        x = alpha_w*feature_1 + beta_w*feature_2
        return x
