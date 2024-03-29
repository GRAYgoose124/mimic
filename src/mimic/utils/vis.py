"""Visualization utilities for the mimic package."""
import logging
import tkinter
import networkx as nx
import matplotlib.pyplot as plt

from numpy import vectorize, e


log = logging.getLogger(__name__)


def draw_linear_layer_network(M, filename=None, show=False, save=False):
    G = M.to_networkx()

    plt.figure(figsize=(12, 8))

    nodelist = G.nodes()

    pos = nx.get_node_attributes(G, "pos")
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodelist, node_size=400, node_color="black", alpha=0.7
    )

    widths = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=widths.keys(),
        width=list([x + 1 for x in widths.values()]),
        edge_color="lightblue",
        alpha=0.6,
        arrows=True,
    )

    activations = nx.get_node_attributes(G, "activations")
    nx.draw_networkx_labels(
        G,
        pos=pos,
        labels=dict(zip(nodelist, [f"{x:.2f}" for x in activations.values()])),
        font_color="white",
        font_size=10,
    )

    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=dict(
            [
                (
                    (
                        u,
                        v,
                    ),
                    f"{d['weight']:.2f}",
                )
                for u, v, d in G.edges(data=True)
            ]
        ),
        alpha=0.5,
        label_pos=0.3,
    )

    plt.box(False)

    if save:
        if filename is None:
            filename = "../output/network.png"

        plt.savefig(filename)
        log.info(f"Saved network visualization to {filename}")

    if show:
        plt.show()
