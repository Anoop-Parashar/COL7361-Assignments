import sys
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np

# Parse command-line arguments (fixed: using square brackets instead of parentheses)
PATH_GRAPHS = sys.argv[1]
PATH_DISCRIMINATIVE_GRAPHS = sys.argv[2]
PATH_FEATURES = sys.argv[3]

def load_graphs(path):
    """Load graphs from a text file in the specified format."""
    graphs = []
    G = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Start of a new graph
                if G is not None:
                    graphs.append(G)
                G = nx.Graph()
            elif line.startswith("v"):
                # Add vertex
                _, nid, label = line.split(sep=" ")
                G.add_node(int(nid), label=int(label))
            elif line.startswith("e"):
                # Add edge
                _, u, v, label = line.split()
                G.add_edge(int(u), int(v), label=int(label))
        # Don't forget the last graph
        if G is not None:
            graphs.append(G)
    return graphs

def contains_fragment(big, small):
    """Check if 'small' is a subgraph of 'big' using subgraph isomorphism."""
    nm = lambda a, b: a["label"] == b["label"]
    em = lambda a, b: a["label"] == b["label"]
    GM = isomorphism.GraphMatcher(big, small,
                                  node_match=nm,
                                  edge_match=em)
    return GM.subgraph_is_isomorphic()

# Load graphs
print("Loading graphs...")
graphs = load_graphs(PATH_GRAPHS)
print(f"Loaded {len(graphs)} graphs from {PATH_GRAPHS}")

print("Loading discriminative subgraphs...")
d_graphs = load_graphs(PATH_DISCRIMINATIVE_GRAPHS)
print(f"Loaded {len(d_graphs)} discriminative subgraphs from {PATH_DISCRIMINATIVE_GRAPHS}")

# Initialize feature matrix
features = np.zeros((len(graphs), len(d_graphs)), dtype=np.uint8)

# Compute features with detailed progress
print("\nComputing features...")
for i, g in enumerate(graphs):
    for j, f in enumerate(d_graphs):
        print(f"Processing graph {i+1}/{len(graphs)}, fragment {j+1}/{len(d_graphs)}", end='\r')
        features[i, j] = contains_fragment(g, f)
    # Print newline after each graph is fully processed for cleaner output
    print(f"Processing graph {i+1}/{len(graphs)}, fragment {len(d_graphs)}/{len(d_graphs)}")

# Save to the specified path
print(f"\nSaving features to {PATH_FEATURES}...")
np.save(PATH_FEATURES, features)
print("Done!")