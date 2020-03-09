import networkx as nx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to input gexf")
parser.add_argument("output", type=str, help="Path to output gexf")
parser.add_argument("--weighted", type=bool, default=False, help="Weighted normalization (use edge weights)")

def normalize_weights(input, output, weighted):
        G = nx.read_gexf(input)

        for node in G:
            if weighted:
                norm = G.in_degree(node, weight="weight")
            else:
                norm = G.out_degree(node)
                
            print("norm", norm)
            edges = G.in_edges(node)
            for e in edges:
                print(e)
                print(">>>", e, G.get_edge_data(e[0],e[1])["weight"])
                G.get_edge_data(e[0],e[1])["weight"] = G.get_edge_data(e[0],e[1])["weight"]/norm
                print("<<<", e, G.get_edge_data(e[0],e[1])["weight"])

        nx.write_gexf(G, "%s" % output)

args = parser.parse_args()
normalize_weights(args.input, args.output)
