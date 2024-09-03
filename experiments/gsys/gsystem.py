import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



class GSystem:
    def __init__(self, axiom, rules, iterations, seed=None):
        self.axiom = axiom
        self.rules = rules
        self.iterations = iterations
        self.graph = nx.DiGraph()
        
    def apply_rules(self, symbol):
        if symbol in self.rules:
            return random.choice(self.rules[symbol])
        return symbol
    
    def generate(self):
        current = self.axiom
        for _ in range(self.iterations):
            next_gen = []
            for symbol in current:
                next_gen.extend(self.apply_rules(symbol))
            current = next_gen
        return current
    
    def create_graph(self):
        sequence = self.generate()
        stack = [0]
        node_counter = 1
        self.graph.add_node(0)  # Add the initial node
        
        for symbol in sequence:
            if symbol == '[':
                stack.append(node_counter - 1)
            elif symbol == ']':
                stack.pop()
            elif symbol == 'N':
                self.graph.add_node(node_counter)
                self.graph.add_edge(stack[-1], node_counter)
                stack[-1] = node_counter
                node_counter += 1
        
        # Ensure the graph is connected
        if not nx.is_weakly_connected(self.graph):
            components = list(nx.weakly_connected_components(self.graph))
            for i in range(1, len(components)):
                self.graph.add_edge(random.choice(list(components[0])), 
                                    random.choice(list(components[i])))
    
    def visualize(self):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold', arrows=True)
        plt.title("G-System Generated Graph")
        plt.axis('off')
        plt.show()

    def get_layer_structure(self):
        return list(nx.topological_generations(self.graph))

    def get_layer_sizes(self):
        return [len(layer) for layer in self.get_layer_structure()]

    def get_adjacency_matrix(self):
        return nx.adjacency_matrix(self.graph).todense()

    def get_feed_forward_connections(self):
        layer_structure = self.get_layer_structure()
        connections = []
        for i in range(len(layer_structure) - 1):
            for node in layer_structure[i]:
                for next_node in layer_structure[i+1]:
                    if self.graph.has_edge(node, next_node):
                        connections.append((node, next_node))
        return connections
