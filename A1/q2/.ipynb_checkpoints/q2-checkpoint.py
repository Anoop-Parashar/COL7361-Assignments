import matplotlib
import sys


class Edge:
    def __init__(self, source, dest, edge_type):
        self.source = source
        self.dest = dest
        self.edge_type = edge_type

    def __repr__(self):
        return f"Edge of type {self.edge_type} from {self.source} to {self.dest}"


class Graph:
    def __init__(self, id):
        self.id = id
        self.num_nodes = 0
        self.num_edges = 0
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)
        self.num_nodes += 1

    def add_edge(self, e):
        self.edges.append(e)
        self.num_edges += 1

    def __repr__(self):
        return f"Graph with ID {self.id} containing {self.num_nodes} nodes and {self.num_edges} edges"


def parse_dataset(dataset_path):
    # Reading-in all the graphs in the dataset
    graphs = []

    with open(dataset_path, "r") as f:

        while True:
            l = f.readline()

            # EOF
            if not l:
                break

            # Empty line but not EOF
            elif l == "\n":
                continue

            l = l.rstrip()
            if l.startswith("#"):
                g = Graph(l.strip("#"))

                # Read number of nodes in g
                l = int(f.readline().rstrip())
                # Add all nodes to the graph
                for _ in range(l):
                    l = f.readline().rstrip()

                    g.add_node(l)

                # Do the same for edges
                l = int(f.readline().rstrip())
                for _ in range(l):
                    l = f.readline().rstrip().split(" ")

                    g.add_edge(Edge(l[0], l[1], l[2]))

                graphs.append(g)

    return graphs


def run_gaston(graphs):
    node_labels = set()
    for g in graphs:
        for node in g.nodes:
            node_labels.add(node)

    node_labels = sorted(list(node_labels))

    str_label_to_int = {}
    for idx, label in enumerate(node_labels):
        str_label_to_int[label] = idx

    # First, create file in the format expected by gSpan
    with open("gaston_graphs.txt", "w") as f:

        for i, g in enumerate(graphs):

            # Define a new graph in the text file
            f.write(f"t # {g.id}\n")

            # Add all nodes
            for j, node in enumerate(g.nodes):
                n = str_label_to_int[node]
                f.write(f"v {j} {n}\n")

            # Add all edges
            for k, e in enumerate(g.edges):
                f.write(f"e {e.source} {e.dest} {e.edge_type}\n")

        # Now, we can run gSpan on this file to get mine frequent subgraphs


def run_fsg(graphs):
    # Create file in the format expected by fsg
    with open("fsg_graphs.txt", "w") as f:

        for i, g in enumerate(graphs):

            # Define a new graph in the text file
            f.write("t\n")

            # Add all nodes
            for j, label in enumerate(g.nodes):
                f.write(f"v {j} {label}\n")

            # Add all edges
            for k, e in enumerate(g.edges):
                f.write(f"u {e.source} {e.dest} {str(e.edge_type)}\n")


def run_gspan(graphs):
    node_labels = set()
    for g in graphs:
        for node in g.nodes:
            node_labels.add(node)

    node_labels = sorted(list(node_labels))

    str_label_to_int = {}
    for idx, label in enumerate(node_labels):
        str_label_to_int[label] = idx

    # First, create file in the format expected by gSpan
    with open("gspan_graphs.txt", "w") as f:

        for i, g in enumerate(graphs):

            # Define a new graph in the text file
            f.write(f"t # {g.id}\n")

            # Add all nodes
            for j, node in enumerate(g.nodes):
                n = str_label_to_int[node]
                f.write(f"v {j} {n}\n")

            # Add all edges
            for k, e in enumerate(g.edges):
                f.write(f"e {e.source} {e.dest} {e.edge_type}\n")

        # Now, we can run gSpan on this file to get mine frequent subgraphs


def main():
    if len(sys.argv) <= 1:
        raise NotImplementedError(
            "[!] Command line arguments need to be passed properly"
        )
    gspan_exe_path = sys.argv[1]
    fsg_exe_path = sys.argv[2]
    gaston_exe_path = sys.argv[3]
    dataset_path = sys.argv[4]
    output_path = sys.argv[5]

    graphs = parse_dataset(dataset_path)

    run_gspan(graphs)
    run_fsg(graphs)
    run_gaston(graphs)


if __name__ == "__main__":
    main()
