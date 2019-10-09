from datetime import datetime
import networkx as nx
import os
import math

def get_source_to_published_counts():
    source_to_count = {}
    path = "D:\\RPI Research\\News Producer Networks\Data\\articles_7232018\\articles\\"
    for dirName, subdirList, fileList in os.walk(path):
        for fn in fileList:
            s = dirName.split("\\")[-1]
            if s in source_to_count.keys():
                source_to_count[s]+=1
            else:
                source_to_count[s] = 1
    return source_to_count

def normalize(source_to_count, weight_edges):
    normalized_edge_weights = []
    for edge in weight_edges:
        source1 = edge[0]
        source2 = edge[1]
        normalized_edge_weights.append((source1, source2, float(edge[2])/source_to_count[source2]))
    return normalized_edge_weights

#flags
overall_network = False
time_slice = False
topic1_slice = False
overall_network_nonnormal = True
overall_network_lognormal = False

pair_data_file = "Article_Matches_Perfect_MAGAHATKIDS.csv"

#general network
if overall_network:
    A = nx.DiGraph()
    edges = []
    weight_dict = {}
    with open(pair_data_file) as pair_data:
        pair_data.readline()
        for line in pair_data:
            line = line.strip().split(",")
            source0 = line[0].split("--")[0]
            source1 = line[1].split("--")[0]
            e = (source0, source1)
            if e in edges:
                weight_dict[e]+=1
            else:
                edges.append(e)
                weight_dict[e] = 1

    weight_edges = [(key[0], key[1], weight_dict[key]) for key in weight_dict.keys()]
    source_to_count = get_source_to_published_counts()
    normalized_weight_edges = normalize(source_to_count, weight_edges)
    for n in normalized_weight_edges:
        print n
    A.add_weighted_edges_from(normalized_weight_edges)
    nx.write_gexf(A, "Feb2018toNov2018_weightNormalize.gexf")

if overall_network_nonnormal:
    A = nx.DiGraph()
    edges = []
    weight_dict = {}
    with open(pair_data_file) as pair_data:
        pair_data.readline()
        for line in pair_data:
            line = line.strip().split(",")
            source0 = line[0].split("--")[0]
            source1 = line[1].split("--")[0]
            e = (source0, source1)
            if e in edges:
                weight_dict[e] += 1
            else:
                edges.append(e)
                weight_dict[e] = 1

    weight_edges = [(key[0], key[1], weight_dict[key]) for key in weight_dict.keys()]
    source_to_count = get_source_to_published_counts()
    A.add_weighted_edges_from(weight_edges)
    nx.write_gexf(A, "MAGAHATKIDS_verbatim.gexf")

if overall_network_lognormal:
    A = nx.DiGraph()
    edges = []
    weight_dict = {}
    with open(pair_data_file) as pair_data:
        pair_data.readline()
        for line in pair_data:
            line = line.strip().split(",")
            source0 = line[0].split("--")[0]
            source1 = line[1].split("--")[0]
            e = (source0, source1)
            if e in edges:
                weight_dict[e] += 1
            else:
                edges.append(e)
                weight_dict[e] = 1

    weight_edges = [(key[0], key[1], math.log10(weight_dict[key])) for key in weight_dict.keys()]
    source_to_count = get_source_to_published_counts()
    A.add_weighted_edges_from(weight_edges)
    nx.write_gexf(A, "Feb2018toNov2018_lognormal_new.gexf")

if time_slice:
    date_slices = ['02', '03', '04', '05', '06', '07']
    A = nx.DiGraph()
    edges = []
    weight_dict = {}
    for cur_date_range in date_slices:
        with open(pair_data_file) as pair_data:
            pair_data.readline()
            for line in pair_data:
                line = line.strip().split(",")
                date_month = line[0].split("--")[1].split("-")[1]
                if date_month == cur_date_range:
                    source0 = line[0].split("--")[0]
                    source1 = line[1].split("--")[0]
                    e = (source0, source1)
                    if e in edges:
                        weight_dict[e] += 1
                    else:
                        edges.append(e)
                        weight_dict[e] = 1

        weight_edges = [(key[0], key[1], weight_dict[key]) for key in weight_dict.keys()]
        A.add_weighted_edges_from(weight_edges)
        nx.write_gexf(A, cur_date_range+".gexf")

if topic1_slice:
    topics_to_use = ['DACA']#['FBI', 'Comey', 'McCabe'] #['White Helmets', 'Syria']#['Russia' 'Russian', 'Mueller', 'Putin']#["Obama", "Clinton", "Barack Obama", "Hillary", "Hillary Clinton", "Hilary", "Michelle Obama", "Benghazi"]
    A = nx.DiGraph()
    edges = []
    weight_dict = {}
    with open(pair_data_file) as pair_data:
        pair_data.readline()
        for line in pair_data:
            line = line.strip().split(",")
            topic1 = line[2]
            topic2 = line[3]
            topic3 = line[4]
            if topic1 in topics_to_use or topic2 in topics_to_use or topic3 in topics_to_use:
                source0 = line[0].split("--")[0]
                source1 = line[1].split("--")[0]
                e = (source0, source1)
                if e in edges:
                    weight_dict[e] += 1
                else:
                    edges.append(e)
                    weight_dict[e] = 1

        weight_edges = [(key[0], key[1], weight_dict[key]) for key in weight_dict.keys()]
        A.add_weighted_edges_from(weight_edges)
        nx.write_gexf(A, "Feb2018toJuly2018_duplicatefilter.gexf")