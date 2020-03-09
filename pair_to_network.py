import networkx as nx
from datetime import datetime
import argparse


# Generates network from a subset of pairs not normalized, weights are absolute number of copies
def build_network(all_pairs, path):
    G = nx.DiGraph()

    weight_dict = {}

    for pair in all_pairs:
        try:
            source0 = pair[0].split("--", 1)[0]
            source1 = pair[1].split("--", 1)[0]
            e = (source0,source1)
            if e in weight_dict:
                weight_dict[e] += 1
            else:
                weight_dict[e] = 1

            weight_edges = [(key[0], key[1], float(weight_dict[key])) \
                                            for key in weight_dict.keys()]

            G.add_weighted_edges_from(weight_edges)
        except:
            print("ERROR", pair)


    nx.write_gexf(G, "%s" % path)


parser = argparse.ArgumentParser()
parser.add_argument("pair_file", type=str, help="Path to input pair file")
parser.add_argument("output_file", type=str, help="Path to output gexf file")
parser.add_argument("--start_date", default=None, type=str, help="Start date YYYY-mm-dd")
parser.add_argument("--end_date", default=None, type=str, help="End date YYYY-mm-dd")

args = parser.parse_args()

with open(args.pair_file) as fin:
    pairs = list(map(lambda l: l.strip().split(",", 1), fin.readlines()))

if not args.start_date and not args.end_date:
    build_network(pairs, args.output_file)
else:
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    filtered_pairs = list()
    for p in pairs:
        try:
            date = datetime.strptime(p[1].split("--")[1], "%Y-%m-%d")
            if args.start_date and date >= start_date and args.end_date and date <= end_date:
                filtered_pairs.append(p)
        except Exception as e:
            print(e)
    print(filtered_pairs)
    build_network(filtered_pairs, args.output_file)
