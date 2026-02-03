import rustworkx as rx
import sys
from tqdm import tqdm
import copy
from multiprocessing import Pool


def parse_db_graphs(graph_fp):
    graphs = []
    with open(graph_fp, "r") as f:
        for l in f:
            line = l.rstrip()

            if line.startswith("#"):
                g = rx.PyGraph()
                graphs.append(g)

            elif line.startswith("v"):
                node_info = line.split(" ")
                node_id = int(node_info[1])
                label = int(node_info[2])

                while node_id >= len(graphs[-1].nodes()):
                    graphs[-1].add_node(None)
                graphs[-1][node_id] = label  # store label as node data

            elif line.startswith("e"):
                edge_info = line.split(" ")
                source = int(edge_info[1])
                dest = int(edge_info[2])
                label = int(edge_info[3])

                graphs[-1].add_edge(source, dest, label)

    return graphs


def parse_graphs(graph_fp):
    graphs = []
    with open(graph_fp, "r") as f:
        for l in f:
            line = l.rstrip()

            if line.startswith("t"):
                sup = int(line.split("*")[-1].strip())
                g = rx.PyGraph(attrs={"support": sup})
                graphs.append(g)

            elif line.startswith("v"):
                node_info = line.split(" ")
                node_id = int(node_info[1])
                label = int(node_info[2])

                while node_id >= len(graphs[-1].nodes()):
                    graphs[-1].add_node(None)
                graphs[-1][node_id] = label

            elif line.startswith("e"):
                edge_info = line.split(" ")
                source = int(edge_info[1])
                dest = int(edge_info[2])
                label = int(edge_info[3])
                graphs[-1].add_edge(source, dest, label)

            elif line.startswith("x"):
                split = line.split(" ")
                sup_ind = [int(i) for i in split[1:]]

    # for g in graphs:
    #     print(g.attrs["support"], g.nodes())
    return graphs


def get_k_disc_subgraphs(freq_graphs, db_graphs, k=50):
    freq_graphs.sort(reverse=True, key=lambda x: x.attrs["support"])

    if len(freq_graphs) > 100:
        freq_graphs = freq_graphs[:100]

    if len(freq_graphs) <= k:
        return freq_graphs

    # if len(freq_graphs) > 2 * k:
    #     freq_graphs = freq_graphs[: 2 * k]

    remaining_graphs = [g.copy() for g in freq_graphs]
    feature_set = [remaining_graphs[-1]]
    remaining_graphs.pop(-1)

    for num in range(k - 1):
        print(f"{'-'*15} Iteration {num} {'-'*15}")
        gammas = []
        for g in tqdm(
            remaining_graphs,
            total=len(remaining_graphs),
            desc="Computing discriminative ratio of each graph",
        ):
            Dx = g.attrs["support"]
            Df = set(range(len(db_graphs)))

            for f in feature_set:
                if rx.is_subgraph_isomorphic(
                    g,
                    f,
                    node_matcher=lambda a, b: a == b,
                    edge_matcher=lambda a, b: a == b,
                ):
                    super_graphs_of_f = set()
                    for idx, h in enumerate(db_graphs):
                        if rx.is_subgraph_isomorphic(
                            h,
                            f,
                            node_matcher=lambda a, b: a == b,
                            edge_matcher=lambda a, b: a == b,
                        ):
                            super_graphs_of_f.add(idx)
                    Df = Df.intersection(super_graphs_of_f)

            # Compute gamma
            gamma = len(Df) / Dx
            gammas.append(gamma)

        next_feature_idx = None
        max_gamma = -1

        for i, gamma in enumerate(gammas):
            if gamma > max_gamma:
                max_gamma = gamma
                next_feature_idx = i

        print(
            f"Max gamma = {max_gamma} for graph #{next_feature_idx} with support = {remaining_graphs[next_feature_idx].attrs['support']}"
        )
        feature_set.append(remaining_graphs[next_feature_idx])
        remaining_graphs.pop(next_feature_idx)

    return feature_set


def save_disc_subgraphs(graphs):
    with open(sys.argv[3], "w") as f:
        for g in graphs:
            f.write("#\n")
            for i, label in enumerate(g.nodes()):
                f.write(f"v {i} {label}\n")
            for idx, t in enumerate(g.edge_list()):
                label = g.edges()[idx]
                source = t[0]
                dest = t[1]
                f.write(f"e {source} {dest} {label}\n")


def main():
    freq_graph_file_path = sys.argv[1]
    db_graph_file_path = sys.argv[2]

    freq_graphs = parse_graphs(freq_graph_file_path)
    db_graphs = parse_db_graphs(db_graph_file_path)

    print("# Frequent graphs:", len(freq_graphs))
    print("# DB graphs:", len(db_graphs))

    print(f"{'-'*50}")
    print("Identifying top-k most discriminative subgraphs...")
    disc_subgraphs = get_k_disc_subgraphs(freq_graphs, db_graphs)

    save_disc_subgraphs(disc_subgraphs)
    print("Discriminative features successfully written to 'features.txt'!")


if __name__ == "__main__":
    main()
