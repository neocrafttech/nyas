import ast
import sys

import matplotlib.pyplot as plt
import networkx as nx


def parse_line(line):
    line = line.strip()
    if not line or ":" not in line:
        return None, None

    point_str, neighbors_str = line.split(":", 1)
    point = int(point_str.strip())

    neighbors = ast.literal_eval(neighbors_str.strip())

    return point, neighbors


def draw_graph(filepath):
    G = nx.Graph()

    with open(filepath, "r") as f:
        for line in f:
            point, neighbors = parse_line(line)
            if point is None:
                continue
            for n in neighbors:
                G.add_edge(point, n)

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, width=1.5)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.axis("off")
        plt.tight_layout()
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python draw_graph.py <datafile>")
        sys.exit(1)

    filepath = sys.argv[1]

    draw_graph(filepath)


if __name__ == "__main__":
    main()
