from numpy import vectorize, e
import networkx as nx
import matplotlib.pyplot as  plt

# @vectorize
def sigmoid(weighted_sum, gamma=1):
    return 1 / (1 + e ** (-gamma * weighted_sum))

# @vectorize
def pd_sigmoid(output):
    return output * (1 - output)


def show_graph(self):
    G = nx.DiGraph()
    edges = []

    # fully connects.. use 0 weights to repr?
    for l_i, l in enumerate(self.layers):
        for n_i, node in enumerate(range(l.width)):
            G.add_node(f'{l_i}.{n_i}', pos=(l_i, n_i))

            if l_i < len(self.layers) - 1:
                for n_j, node2 in enumerate(range(self.layers[l_i + 1].width)):
                    edges.append((f'{l_i}.{n_i}', f'{l_i + 1}.{n_j}', self.layers[l_i].weights[n_i][n_j]))

    G.add_weighted_edges_from(edges)

    widths = nx.get_edge_attributes(G, 'weight')
    nodelist = G.nodes()

    plt.figure(figsize=(12,8))

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G,pos,
                            nodelist=nodelist,
                            node_size=1200,
                            node_color='black',
                            alpha=0.7)
    nx.draw_networkx_edges(G,pos,
                            edgelist = widths.keys(),
                            width=list([x + 1 for x in widths.values()]),
                            edge_color='lightblue',
                            alpha=0.6,
                            arrows=True)
    nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist,nodelist)),
                            font_color='white')
    nx.draw_networkx_edge_labels(G,
                            pos=pos, 
                            edge_labels=dict([((u,v,), f"{' '}{d['weight']:.2f}") for u,v,d in G.edges(data=True)]), 
                            alpha=0.5,
                            label_pos=.3)
    plt.box(False)

    plt.show()

    return G