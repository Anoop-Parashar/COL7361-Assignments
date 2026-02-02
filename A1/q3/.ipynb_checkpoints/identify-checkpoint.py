import networkx as nx
import sys
from tqdm import tqdm
import copy


def parse_db_graphs(graph_fp):
    graphs = []
    with open(graph_fp, "r") as f:
        for l in f:
            line = l.rstrip()

            # New graph
            if line.startswith("#"):
                g = nx.Graph()
                graphs.append(g)

            # Node
            if line.startswith("v"):
                node_info = line.split(" ")
                id = int(node_info[1])
                label = int(node_info[2])

                graphs[-1].add_node(id, label=label)

            # Edge
            if line.startswith("e"):
                edge_info = line.split(" ")
                source = int(edge_info[1])
                dest = int(edge_info[2])
                label = int(edge_info[3])

                graphs[-1].add_edge(source, dest, label=label)

    return graphs


def parse_graphs(graph_fp):
    graphs = []
    with open(graph_fp, "r") as f:
        for l in f:
            line = l.rstrip()

            # New graph
            if line.startswith("t"):
                sup = int(line.split("*")[-1].strip())
                g = nx.Graph(support=sup)
                graphs.append(g)

            # Node
            if line.startswith("v"):
                node_info = line.split(" ")
                id = int(node_info[1])
                label = int(node_info[2])

                graphs[-1].add_node(id, label=label)

            # Edge
            if line.startswith("e"):
                edge_info = line.split(" ")
                source = int(edge_info[1])
                dest = int(edge_info[2])
                label = int(edge_info[3])

                graphs[-1].add_edge(source, dest, label=label)

            if line.startswith("x"):
                split = line.split(" ")
                sup_ind = [int(i) for i in split[1:]]
                graphs[-1].graph["sup_indices"] = sup_ind

    # for graph in graphs:
    #     print(graph)
    #     for e in graph.edges.data():
    #         print(e[0], e[1], e[2])
    #
    # exit(0)

    return graphs


def get_k_disc_subgraphs(freq_graphs, db_graphs, k=50):
    freq_graphs.sort(reverse=True, key=lambda x: x.graph["support"])

    if len(freq_graphs) <= k:
        return freq_graphs

    if len(freq_graphs) > 2 * k:
        freq_graphs = freq_graphs[: 2 * k]

    print(f"Total frequent subgraphs:", len(freq_graphs))
    # for graph in freq_graphs:
    #     print(graph)
    #     print(graph.graph)

    # Keep the rarest frequent subgraph as the initial selected feature
    feature_set = [freq_graphs[-1]]
    remaining_graphs = copy.deepcopy(freq_graphs)
    remaining_graphs.pop(-1)

    for num in range(k):
        print(f"{'-'*15} Iteration {num} {'-'*15}")
        gammas = []
        for g in tqdm(
            remaining_graphs,
            total=len(remaining_graphs),
            desc="Computing discriminative ratio of each graph",
        ):

            # Compute Dx
            Dx = g.graph["support"]

            # Compute Df
            Df = set(range(len(db_graphs)))

            for f in feature_set:

                if nx.algorithms.isomorphism.GraphMatcher(
                    g, f
                ).subgraph_is_isomorphic():
                    super_graphs_of_f = set()

                    for idx, h in enumerate(db_graphs):
                        if nx.algorithms.isomorphism.GraphMatcher(
                            h, f
                        ).subgraph_is_isomorphic():
                            super_graphs_of_f.add(idx)

                    Df = Df.intersection(super_graphs_of_f)

            # Compute gamma = |Df| / |Dx|
            gamma = len(Df) / Dx
            gammas.append(gamma)

        # Select graph with highest gamma as the next discriminative feature
        next_feature_idx = None
        max_gamma = -1
        for i, gamma in enumerate(gammas):
            if gamma > max_gamma:
                max_gamma = gamma
                next_feature_idx = i

        print(f"Max gamma = {max_gamma} for {remaining_graphs[next_feature_idx]}")
        feature_set.append(remaining_graphs[next_feature_idx])
        remaining_graphs.pop(next_feature_idx)

    return feature_set


def save_disc_subgraphs(graphs):
    with open("features.txt", "w") as f:
        for g in graphs:
            # New graph
            f.write("#\n")

            # Nodes
            for id, label in g.nodes.data():
                f.write(f"v {id} {label['label']}\n")

            # Edges
            for source, dest, label in g.edges.data():
                f.write(f"e {source} {dest} {label['label']}\n")


def main():
    freq_graph_file_path = sys.argv[1]
    db_graph_file_path = sys.argv[2]

    # Read and parse all frequent graphs given by gSpan
    freq_graphs = parse_graphs(freq_graph_file_path)
    db_graphs = parse_db_graphs(db_graph_file_path)

    print("# Frequent graphs:", len(freq_graphs))
    print("# DB graphs:", len(db_graphs))

    # Identify top-k most discriminative frequent subgraphs
    print(f"{'-'*50}")
    print("Identfying top-k most discriminative subgraphs...")
    disc_subgraphs = get_k_disc_subgraphs(freq_graphs, db_graphs)

    # Write discriminative subgraphs to file
    save_disc_subgraphs(disc_subgraphs)

    print("Discriminative features successfully written to 'features.txt'!")


if __name__ == "__main__":
    main()
