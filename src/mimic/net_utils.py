from numpy import vectorize, e
import networkx as nx
import matplotlib.pyplot as  plt

# @vectorize
def sigmoid(weighted_sum, gamma=1):
    return 1 / (1 + e ** (-gamma * weighted_sum))

# @vectorize
def pd_sigmoid(output):
    out = sigmoid(output)
    return out * (1 - out)


def build_network_graph(N):
    G = nx.DiGraph()
    edges = []

    # fully connects.. use 0 weights to repr?
    for l_i, l in enumerate(N.layers):
        for n_i, node in enumerate(range(l.width)):
            G.add_node(f'{l_i}.{n_i}', pos=(l_i, n_i), activation=l.nodes[n_i])

            if l_i < len(N.layers) - 1:
                for n_j, node2 in enumerate(range(N.layers[l_i + 1].width)):
                    edges.append((f'{l_i}.{n_i}', f'{l_i + 1}.{n_j}', N.layers[l_i].weights[n_i][n_j]))

    G.add_weighted_edges_from(edges)
    return G


def show_network(N):
    G = build_network_graph(N)

    plt.figure(figsize=(12,8))

    nodelist = G.nodes()
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx_nodes(G,pos,
                            nodelist=nodelist,
                            node_size=400,
                            node_color='black',
                            alpha=0.7)

    widths = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G,pos,
                            edgelist = widths.keys(),
                            width=list([x + 1 for x in widths.values()]),
                            edge_color='lightblue',
                            alpha=0.6,
                            arrows=True)

    activations = nx.get_node_attributes(G, 'activation')
    nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist, [f"{x:.2f}" for x in activations.values()])),
                            font_color='white')

    nx.draw_networkx_edge_labels(G,
                            pos=pos, 
                            edge_labels=dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)]), 
                            alpha=0.5,
                            label_pos=.3)

    plt.box(False)
    plt.show()
    