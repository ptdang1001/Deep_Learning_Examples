import matplotlib.pyplot as plt
import networkx as nx

# draw the relationship graph
'''
input:    
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
'''
def plot_graph(G, weight_name=None):
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None

    if weight_name:
        weights = [int(G[u][v][weight_name]) for u, v in edges]
        labels = nx.get_edge_attributes(G, weight_name)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nodelist1 = []
        nodelist2 = []
        for i in range(34):
            if G.nodes[i]['club'] == 'Mr. Hi':
                nodelist1.append(i)
            else:
                nodelist2.append(i)
        # nx.draw_networkx(G, pos, edges=edges);
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist1, node_size=300, node_color='r', alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist2, node_size=300, node_color='b', alpha=0.8)
        nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.4)

