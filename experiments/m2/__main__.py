import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .kan import KAN

def plot_demo(X, y, X_test, y_pred):
    """Plot the neural network fitting uniform data"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label="Training data")
    plt.plot(X_test, y_pred, "r-", label="Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Optimized Neural Network Fitting Uniform Data")
    plt.legend()
    plt.show()


def networkx_weights_demo(kan):
    """Display the weights of the neural network as a networkx graph with weighted edges"""
    G = nx.DiGraph()

    # Add nodes for each neuron
    for i, layer in enumerate(kan.layers):
        for j in range(layer[0].shape[0]):
            G.add_node(f"L{i}N{j}", layer=i, neuron=j)

    # Add output layer nodes
    output_layer = len(kan.layers)
    for j in range(kan.layers[-1][0].shape[1]):
        G.add_node(f"L{output_layer}N{j}", layer=output_layer, neuron=j)

    # Add edges with weights
    for i, layer in enumerate(kan.layers):
        weights, _ = layer
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                G.add_edge(f"L{i}N{j}", f"L{i+1}N{k}", weight=weights[j, k])

    # Draw the graph
    # pos = nx.spring_layout(G)
    pos = nx.spectral_layout(G)
    # pos = nx.fruchterman_reingold_layout(G, k=0.1, iterations=100)
    # pos = nx.multipartite_layout(G, subset_key="layer")

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        width=2,
        alpha=0.7,
    )
    plt.title("Neural Network Architecture")
    plt.show()


if __name__ == "__main__":
    # Generate uniform data
    np.random.seed(42)
    X = np.random.uniform(0, 1, (10000, 1))
    y = 0.5 * np.sin(4 * np.pi * X) + 0.5 + 0.1 * np.random.randn(10000, 1)

    # Create and train the neural network
    kan = KAN([1, 20, 20, 1])
    kan.train(X, y, learning_rate=0.01, epochs=100, batch_size=64)

    # Generate predictions
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_pred = kan.predict(X_test)

    networkx_weights_demo(kan)
    plot_demo(X, y, X_test, y_pred)
