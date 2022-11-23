import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class ShaDow(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, batch_idx, root_n_id):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)

        root_idx = batch_idx.clone()
        root_idx[1:] -= batch_idx[:-1]
        root_idx[0] += 1
        # We merge both central node embeddings and subgraph embeddings:
        # x = torch.cat([x[root_idx == 1, :], global_mean_pool(x, batch_idx)], dim=-1)
        # x = self.lin(x)
        graph_embs = x[root_idx == 1, :]

        return graph_embs