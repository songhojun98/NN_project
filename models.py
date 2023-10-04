import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F


class GraphPoolingModel_layer_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphPoolingModel_layer_1, self).__init__()

        self.gcn1 = GCNConv(input_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)

        # max, mean, sum pooling을 concatenation
        self.fc1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = self.norm1(x)

        # Pooling
        x_mean = global_mean_pool(x, data.batch)
        x_add = global_add_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)

        # read-out
        x = torch.cat([x_mean, x_add, x_max], dim=1)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
    

class GraphPoolingModel_layer_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphPoolingModel_layer_2, self).__init__()

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # max, mean, sum pooling을 concatenation
        self.fc1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = self.norm1(x)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.norm2(x)

        # Pooling
        x_mean = global_mean_pool(x, data.batch)
        x_add = global_add_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)

        # read-out
        x = torch.cat([x_mean, x_add, x_max], dim=1)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
    

class GraphPoolingModel_layer_3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphPoolingModel_layer_3, self).__init__()

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # max, mean, sum pooling을 concatenation
        self.fc1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = self.norm1(x)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.norm2(x)
        x = F.relu(self.gcn3(x, edge_index))
        x = self.norm3(x)

        # Pooling
        x_mean = global_mean_pool(x, data.batch)
        x_add = global_add_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)

        # read-out
        x = torch.cat([x_mean, x_add, x_max], dim=1)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
