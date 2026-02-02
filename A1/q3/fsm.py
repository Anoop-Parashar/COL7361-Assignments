import sys
from tqdm import tqdm
import networkx as nx
import rustworkx as rx


def parse_graphs(graph_fp):
    graphs = []
    with open(graph_fp, "r") as f:
        for l in f:
            line = l.rstrip()

            if line.startswith("#"):
                g = rx.PyDiGraph()
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


def remove_duplicate_graphs(graphs):
    unique = []

    for g in tqdm(graphs, total=len(graphs)):
        found_duplicate = False

        for h in unique:
            if rx.is_isomorphic(
                h,
                g,
                node_matcher=lambda a, b: a == b,
                edge_matcher=lambda a, b: a == b,
            ):
                found_duplicate = True

                break

        if not found_duplicate:
            unique.append(g)

    return unique


def run_fsm(graphs):

    g = graphs[0]

    with open("gspan_graphs.txt", "w") as f:

        for i, g in tqdm(enumerate(graphs), total=len(graphs)):

            # Define a new graph in the text file
            f.write(f"t # {i}\n")

            # Add all nodes
            for idx, label in enumerate(g.nodes()):
                f.write(f"v {idx} {label}\n")

            # Add all edges
            visited_edges = []
            for idx, t in enumerate(g.edge_list()):
                label = g.edges()[idx]
                source = t[0]
                dest = t[1]
                visited_edges.append((source, dest))

                if (dest, source) not in visited_edges:
                    f.write(f"e {source} {dest} {label}\n")


def main():
    if len(sys.argv) <= 1:
        raise NotImplementedError(
            "[!] Command line arguments need to be passed properly"
        )

    # Read graphs
    graphs = parse_graphs(sys.argv[1])

    graphs = remove_duplicate_graphs(
        graphs
    )  # NOTE: Remember to uncomment this for the final run

    # Convert graphs into format expected by Gaston
    run_fsm(graphs)

    print("Converted DB graphs to the required format")


if __name__ == "__main__":
    main()
