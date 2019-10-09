import networkx as nx
import copy

def get_communities():
    c2s = {}
    s2c = {}
    fn = "../Analysis Code/Feb2018toNov2018_lognormal_communities [Nodes]_FINAL.csv"
    with open(fn) as data:
        header = data.readline().strip().split(",")
        for line in data:
            l = line.strip().split(",")
            source = l[0]
            community = l[2]
            if community in c2s.keys():
                c2s[community].append(source)
            else:
                c2s[community] = [source]
            s2c[source] = community
    return c2s, s2c


def get_community_only_network(G, s2c, comm):
    nodes_to_remove = []
    G_copy = copy.copy(G)
    for node in G_copy.nodes():
        try:
            if s2c[node] != comm:
                nodes_to_remove.append(node)
        except:
            continue

    [G_copy.remove_node(nr) for nr in nodes_to_remove]
    return G_copy


fn = "Feb2018toNov2018_lognormal_communities_new.gexf"
G = nx.read_gexf(fn)
c2s, s2c = get_communities()
G_community = get_community_only_network(G, s2c, '12')



print "Core <-------------------------"
core =  nx.k_core(G_community)

print len(G_community.nodes()), len(core.nodes())

for node in core.nodes():
    print node

# print "Periphery <-------------------------"
# per = nx.periphery(G_community)
#
# print len(G_community.nodes()), len(per.nodes())
#
# for node in per.nodes():
#     print node