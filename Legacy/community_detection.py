import networkx as nx
from modularity_maximization import partition
from modularity_maximization.utils import get_modularity


networkfile = "Feb2018toNov2018_new.gexf"
G = nx.read_gexf(networkfile)

comm_dict = partition(G)
print get_modularity(G, comm_dict)
print comm_dict
for node in G.nodes():
    print node
nx.set_node_attributes(G, name='community', values=comm_dict)
# for comm in set(comm_dict.values()):
#     print("Community %d"%comm)
#     print(', '.join([node for node in comm_dict if comm_dict[node] == comm]))
#
#     for node in comm_dict:
#         if comm_dict[node] == comm:

nx.write_gexf(G, "Feb2018toNov2018_communities_new.gexf")

