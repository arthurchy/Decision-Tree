import graphviz
from graphviz import Digraph

graphviz.set_jupyter_format('png')  # If using Jupyter

dot = graphviz.Digraph()
dot.attr(engine='dot')  # Ensure correct engine

class DecisionTreeGraphviz:
    def __init__(self, tree):
        self.tree = tree
        self.graph = Digraph(format='png')

    def _add_node(self, node, node_id):
        if node.score is None:
            label = f"Leaf\nSamples: {node.samples}\nLabels: {node.labels}"
            self.graph.node(str(node_id), label, shape='box', style='filled', color='lightblue')
        else:
            label = f"x[{node.feature}] <= {node.split_value:.4f}\nScore: {node.score:.4f}\nSamples: {node.samples}"
            self.graph.node(str(node_id), label, shape='ellipse', style='filled', color='lightgreen')

    def _build_graph(self, node, node_id):
        if node is None:
            return
        
        self._add_node(node, node_id)

        if node.left:
            left_id = f"{node_id}L"
            self._build_graph(node.left, left_id)
            self.graph.edge(str(node_id), str(left_id), label="True")

        if node.right:
            right_id = f"{node_id}R"
            self._build_graph(node.right, right_id)
            self.graph.edge(str(node_id), str(right_id), label="False")

    def visualize(self, output_file='tree'):
        """
        Visualize the decision tree and render it.
        """
        self._build_graph(self.tree, "root")
        self.graph.render(output_file, view=True)