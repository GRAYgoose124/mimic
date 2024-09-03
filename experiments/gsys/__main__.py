import numpy as np
import matplotlib.pyplot as plt

from .gsystem import GSystem
from .amorphous import AmorphousNetwork


if __name__ == "__main__":
    axiom = 'N'
    rules = {
        'N': ['NN', 'N[N]', '[N]N', 'N[N]N'],
        '[': ['['],
        ']': [']'],
    }

    g_system = GSystem(axiom, rules, iterations=6)
    g_system.create_graph()
    g_system.visualize()

    print("Layer sizes:", g_system.get_layer_sizes())
    print("Feed-forward connections:", g_system.get_feed_forward_connections())

    # Create and use the amorphous network
    amorphous_net = AmorphousNetwork(g_system)

    # Generate example data
    X = np.linspace(0, 10, 1000).reshape(-1, 1)
    y = np.sin(X)

    # Add noise to the data
    y += np.random.randn(*y.shape) * 0.1

    #shuffle (X, y) keeping pairs together by index
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Normalize the data
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    # Show the original data
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Original Data')
    plt.title("Original Data")
    plt.legend()
    plt.show()

    # Train the network
    amorphous_net.train(X, y, learning_rate=0.005, epochs=5000)

    # Make predictions
    predictions = amorphous_net.forward(X)
    print("Predictions shape:", predictions.shape)

    # Compare predictions with actual outputs
    mse = np.mean(np.square(predictions - y))
    print("Mean squared error:", mse)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='True')
    plt.scatter(X, predictions, alpha=0.5, label='Predicted')
    plt.legend()
    plt.title("Amorphous Network Predictions")
    plt.show()