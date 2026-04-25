"""
predict_wrapper_B.py
"""

import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data


class ChunkedNodeWrapper(nn.Module):
    """
    Runs mini-batch inference via NeighborLoader to avoid OOM on 2.9M nodes.
    predict.py calls model(x, edge_index) expecting [N, num_classes].
    """

    def __init__(self, model, num_nodes, batch_size=4096, num_neighbors=None):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [10, 10]

    def forward(self, x, edge_index):
        device = x.device
        self.model.to(device)
        self.model.eval()
        data = Data(x=x.cpu(), edge_index=edge_index.cpu(), num_nodes=self.num_nodes)
        loader = NeighborLoader(
            data,
            input_nodes=torch.arange(self.num_nodes),
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        out_list = [None] * self.num_nodes
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = self.model(batch.x, batch.edge_index)[: batch.batch_size]
                for j in range(batch.batch_size):
                    out_list[batch.n_id[j].item()] = logits[j].cpu()
        return torch.stack(out_list, dim=0).to(device)


class PredictWrapper(nn.Module):
    """
    predict.py calls model(x, edge_index, edge_pairs) where edge_pairs is [E, 2].
    This wrapper translates that to SAGELinkPredictor's decode format.
    """

    def __init__(self, model, x, edge_index):
        super().__init__()
        self.model = model
        self.register_buffer("x", x.cpu())
        self.register_buffer("edge_index", edge_index.cpu())

    def forward(self, x, edge_index, edge_pairs):
        device = next(self.model.parameters()).device
        z = self.model.encode(x.to(device), edge_index.to(device))
        edge_pairs = edge_pairs.to(device)
        return self.model.decode(z, edge_pairs)
