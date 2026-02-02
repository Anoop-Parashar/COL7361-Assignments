import sys
import os
import igraph as ig
from multiprocessing import Pool, Manager
import numpy as np
from tqdm import tqdm

PATH_GRAPHS = sys.argv[1]
PATH_DISCRIMINATIVE_GRAPHS = sys.argv[2]
PATH_FEATURES = sys.argv[3]

NUM_CORES = int(os.cpu_count())

def load_graphs_igraph(path):
    graphs = []
    vertices = []
    edges = []
    vertex_labels = []
    edge_labels = []
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if vertices or edges:
                    g = ig.Graph(n=len(vertices))
                    g.vs['label'] = vertex_labels
                    if edges:
                        g.add_edges(edges)
                        g.es['label'] = edge_labels
                    graphs.append(g)
                vertices = []
                edges = []
                vertex_labels = []
                edge_labels = []
            elif line.startswith("v"):
                _, nid, label = line.split(sep=" ")
                vertices.append(int(nid))
                vertex_labels.append(int(label))
            elif line.startswith("e"):
                _, u, v, label = line.split()
                edges.append((int(u), int(v)))
                edge_labels.append(int(label))
        
    
        if vertices or edges:
            g = ig.Graph(n=len(vertices))
            g.vs['label'] = vertex_labels
            if edges:
                g.add_edges(edges)
                g.es['label'] = edge_labels
            graphs.append(g)
    
    return graphs

def contains_fragment_igraph(big, small):
    
    try:
        
        result = big.get_subisomorphisms_vf2(
            small,
            node_compat_fn=lambda g1, g2, v1, v2: g1.vs[v1]['label'] == g2.vs[v2]['label'],
            edge_compat_fn=lambda g1, g2, e1, e2: g1.es[e1]['label'] == g2.es[e2]['label']
        )
        return len(result) > 0
    except:
        return False

def process_chunk(args):
    
    chunk_indices, graphs_chunk, d_graphs = args
    
    
    chunk_results = []
    
    for local_idx, (global_idx, g) in enumerate(zip(chunk_indices, graphs_chunk)):
        
        row = []
        for f in d_graphs:
            result = contains_fragment_igraph(g, f)
            row.append(1 if result else 0)
        
        chunk_results.append((global_idx, row))
    
    return chunk_results


print("Loading graphs...")
graphs = load_graphs_igraph(PATH_GRAPHS)
print(f"Loaded {len(graphs)} graphs from {PATH_GRAPHS}")

print("Loading discriminative subgraphs...")
d_graphs = load_graphs_igraph(PATH_DISCRIMINATIVE_GRAPHS)
print(f"Loaded {len(d_graphs)} discriminative subgraphs from {PATH_DISCRIMINATIVE_GRAPHS}")


features = np.zeros((len(graphs), len(d_graphs)), dtype=np.uint8)

print(f"\nComputing features with {NUM_CORES} core(s)...")

if NUM_CORES > 1:
    
    chunk_size = len(graphs) // NUM_CORES
    chunks = []
    
    for i in range(NUM_CORES):
        start_idx = i * chunk_size
        if i == NUM_CORES - 1:
            
            end_idx = len(graphs)
        else:
            end_idx = (i + 1) * chunk_size
        
        chunk_indices = list(range(start_idx, end_idx))
        graphs_chunk = graphs[start_idx:end_idx]
        chunks.append((chunk_indices, graphs_chunk))
    
    
    worker_args = [
        (chunk_indices, graphs_chunk, d_graphs)
        for chunk_indices, graphs_chunk in chunks
    ]
    
   
    with Pool(processes=NUM_CORES) as pool:
        with tqdm(total=len(graphs), desc="Processing graphs", unit="graph") as pbar:
            all_results = []
            for result in pool.imap_unordered(process_chunk, worker_args):
                all_results.append(result)
                
                pbar.update(len(result))
    
    
    for chunk_results in all_results:
        for global_idx, row in chunk_results:
            features[global_idx, :] = row

else:
    
    for i, g in enumerate(tqdm(graphs, desc="Processing graphs", unit="graph")):
        for j, f in enumerate(d_graphs):
            features[i, j] = contains_fragment_igraph(g, f)


print(f"\nSaving features to {PATH_FEATURES}...")
np.save(PATH_FEATURES, features)
print("Done!")