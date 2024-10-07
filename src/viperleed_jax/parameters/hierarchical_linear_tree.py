import numpy as np
import jax.numpy as jnp

from .linear_transformer import LinearTransformer

class HLNode:
    def __init__(self, dof):
        """Initialize a node with a given number of degrees of freedom (dof)."""
        self.dof = dof
        self.children = []  # List to hold child nodes
        self.parent = None  # Reference to parent node

    def add_child(self, child_node):
        """Add a child node to the current node."""
        self.children.append(child_node)
        child_node.parent = self

    def __repr__(self):
        return f"HLNode(dof={self.dof})"


class HLLeafNode(HLNode):
    def __init__(self, dof, atom_site_element, site_element):
        """
        Initialize a leaf node with degrees of freedom and specific attributes.
        - `dof`: Number of degrees of freedom for the node.
        - `atom_site_element`: The atomic site element represented by this node.
        - `site_element`: The specific site element attribute for this node.
        """
        super().__init__(dof)
        self.atom_site_element = atom_site_element
        self.site_element = site_element

    def __repr__(self):
        return f"HLLeafNode(dof={self.dof}, atom_site_element={self.atom_site_element}, site_element={self.site_element})"


class HLConstraintNode(HLNode):
    def __init__(self, dof, n_free_params):
        """
        Initialize a constraint node that reduces the degrees of freedom.
        - `dof`: Number of degrees of freedom for the parent node.
        - `n_free_params`: The number of free parameters before the constraint is applied.
        """
        super().__init__(dof)
        self.n_free_params = n_free_params

    def __repr__(self):
        return f"HLConstraintNode(dof={self.dof}, n_free_params={self.n_free_params})"


class HLEdge:
    def __init__(self, parent, child, transformer):
        """
        Define a linear edge between parent and child nodes using a LinearTransformer.
        - `parent`: Parent node in the tree.
        - `child`: Child node in the tree.
        - `transformer`: LinearTransformer object defining the transformation.
        """
        self.parent = parent
        self.child = child
        self.transformer = transformer

    def __repr__(self):
        return f"HLEdge(Child dof={self.child.dof} -> Parent dof={self.parent.dof}, {self.transformer})"


class Layer:
    """A class representing a single layer in the Hierarchical Linear Tree."""

    def __init__(self, name, nodes):
        """
        Initialize a layer with a name and a set of nodes.
        - `name`: Name of the layer.
        - `nodes`: A list of `HLConstraintNode` instances that make up this layer.
        """
        self.name = name
        self.nodes = nodes

    def __repr__(self):
        return f"Layer(name={self.name}, nodes={len(self.nodes)} nodes)"


class HierarchicalLinearTree:
    def __init__(self, leaf_nodes):
        """
        Initialize the tree with a list of leaf nodes.
        - `leaf_nodes`: A list of `HLLeafNode` instances to serve as the initial leaves.
        """
        self.nodes = []  # Store nodes in a list
        self.edges = []  # Store all edges
        self.layers = []  # Store all layers

        # Ensure all provided leaf nodes are instances of HLLeafNode
        for leaf in leaf_nodes:
            if not isinstance(leaf, HLLeafNode):
                raise TypeError(
                    f"All leaf nodes must be instances of HLLeafNode. Invalid node: {leaf}"
                )
            self.nodes.append(leaf)

    def add_layer(self, transformers, layer_name):
        """
        Add a new layer to the tree, connecting all root nodes to new parent constraint nodes.
        The dimensions of the transformers must be validated against the child nodes' degrees of freedom.

        Parameters:
        - `transformers`: List of LinearTransformer objects, one for each current root node.
        - `layer_name`: Name of the new layer.
        """
        # Ensure that the number of transformers matches the number of root nodes
        if len(transformers) != len(self.roots):
            raise ValueError(
                "Number of transformers must match the number of root nodes."
            )

        # Calculate parent node dof and total free parameters
        parent_dof = transformers[0].weights.shape[0]
        total_free_params = sum(root.dof for root in self.roots)

        for transformer, child in zip(transformers, self.roots):
            if transformer.weights.shape[1] != child.dof:
                raise ValueError(
                    f"Transformer columns ({transformer.weights.shape[1]}) must match child node dof ({child.dof})."
                )
            if transformer.weights.shape[0] != parent_dof:
                raise ValueError(
                    "All transformers must have the same number of rows (parent node dof)."
                )

        # Create a new parent constraint node for each child
        new_nodes = []
        for child, transformer in zip(self.roots, transformers):
            parent_node = HLConstraintNode(
                parent_dof, n_free_params=total_free_params
            )
            self.nodes.append(parent_node)
            new_nodes.append(parent_node)

            # Connect each child to the new parent node
            self._add_edge(child, parent_node, transformer)

        # Create and add the new layer
        layer = Layer(name=layer_name, nodes=new_nodes)
        self.layers.append(layer)

    def _add_edge(self, child, parent, transformer):
        """Create a linear edge between child and parent nodes using a LinearTransformer."""
        parent.add_child(child)
        edge = HLEdge(parent, child, transformer)
        self.edges.append(edge)

    @property
    def roots(self):
        """Dynamically calculate the root nodes (nodes without parents)."""
        return [node for node in self.nodes if node.parent is None]

    @property
    def n_roots(self):
        """Return the number of root nodes."""
        return len(self.roots)

    def display_structure(self):
        """Display the structure of the tree with node degrees of freedom (dof), layers, and leaf attributes."""
        print(
            f"Tree Structure with {self.n_roots} root node(s) and {len(self.layers)} layer(s):"
        )
        for idx, node in enumerate(self.nodes):
            if isinstance(node, HLLeafNode):
                node_info = f" [Leaf: atom_site_element={node.atom_site_element}, site_element={node.site_element}]"
            elif isinstance(node, HLConstraintNode):
                node_info = (
                    f" [Constraint Node: n_free_params={node.n_free_params}]"
                )
            else:
                node_info = ""
            root_info = " (Root)" if node in self.roots else ""
            print(f"Node {idx}: {node.dof} dof{root_info}{node_info}")

        print("\nLayers:")
        for layer in self.layers:
            print(f"{layer.name}: {len(layer.nodes)} nodes")
