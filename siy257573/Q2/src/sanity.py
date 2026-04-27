import torch
from torch_geometric.data import Data
from torch_geometric.data.data import Data as DataAlias
import os

path = "/home/asus/COL761_A3/public_datasets/B/data.pt"  # adjust filename

size_gb = os.path.getsize(path) / (1024**3)
print(f"File size: {size_gb:.2f} GB")

# Option 1: allowlist the PyG class and use mmap (best for your RAM situation)
torch.serialization.add_safe_globals([Data, DataAlias])

data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)

print(f"Type: {type(data)}")
if isinstance(data, list):
    print(f"Number of graphs: {len(data)}")
    print(f"First graph: {data[0]}")
    print(f"First graph keys: {data[0].keys()}")
else:
    print(data)
