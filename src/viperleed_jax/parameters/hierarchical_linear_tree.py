import numpy as np
import jax.numpy as jnp


class LinearTransformer:
    """
    Linear transformation that applies a weight and a bias to an input.
    Can optionally reshape the output.
    """

    def __init__(self, weights, biases, out_reshape=None):
        self.weights = jnp.array(weights)
        self.n_free_params = self.weights.shape[1]
        self.biases = jnp.array(biases)
        self.out_reshape = out_reshape

    def __call__(self, free_params):
        if self.n_free_params == 0:
            return self.biases
        if isinstance(free_params, float):
            free_params = jnp.array([free_params])
        free_params = jnp.array(free_params)
        if len(free_params) != self.n_free_params:
            raise ValueError("Free parameters have wrong shape")
        result = self.weights @ free_params + self.biases
        if self.out_reshape is not None:
            result = result.reshape(self.out_reshape)
        return result

    def __repr__(self):
        return f"LinearTransformer(weights={self.weights.shape}, biases={self.biases.shape}, out_reshape={self.out_reshape})"


class HLNode:
    def __init__(self, dof):
        """
        Initialize a node with a given number of degrees of freedom (dof).
        - `dof`: Number of degrees of freedom for the node.
        """
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


class HierarchicalLinearTree:
    def __init__(self, leaf_nodes):
        """
        Initialize the tree with a list of leaf nodes.
        - `leaf_nodes`: A list of `HLLeafNode` instances to serve as the initial leaves.
        """
        self.nodes = []  # Store nodes in a list
        self.edges = []  # Store all edges

        # Ensure all provided leaf nodes are instances of HLLeafNode
        for leaf in leaf_nodes:
            if not isinstance(leaf, HLLeafNode):
                raise TypeError(
                    f"All leaf nodes must be instances of HLLeafNode. Invalid node: {leaf}"
                )
            self.nodes.append(leaf)

    def add_parent(self, child_indices, transformers):
        """
        Add a new parent node that connects to the specified child nodes by their indices.
        The dimensions of the transformers must be validated against the child nodes' degrees of freedom.

        Parameters:
        - `child_indices`: List of indices of the child nodes (must be roots).
        - `transformers`: List of LinearTransformer objects for each child.

        Returns:
        - The newly created parent node.
        """
        if len(child_indices) != len(transformers):
            raise ValueError(
                "Number of child nodes and transformers must match."
            )

        # Retrieve child nodes by their indices
        children = [self.roots[i] for i in child_indices]

        # Calculate the dof for the new parent node
        parent_dof = transformers[0].weights.shape[0]
        for transformer, child in zip(transformers, children):
            if transformer.weights.shape[1] != child.dof:
                raise ValueError(
                    f"Transformer columns ({transformer.weights.shape[1]}) must match child node dof ({child.dof})."
                )
            if transformer.weights.shape[0] != parent_dof:
                raise ValueError(
                    "All transformers must have the same number of rows (parent node dof)."
                )

        # Create the new parent node
        parent_node = HLNode(parent_dof)
        self.nodes.append(parent_node)

        # Connect the parent to each child
        for child, transformer in zip(children, transformers):
            self._add_edge(child, parent_node, transformer)

        return parent_node

    def _add_edge(self, child, parent, transformer):
        """
        Create a linear edge between child and parent nodes using a LinearTransformer.
        This method is private and should not be called directly.
        """
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
        """Display the structure of the tree with node degrees of freedom (dof) and leaf attributes."""
        print(f"Tree Structure with {self.n_roots} root node(s):")
        for idx, node in enumerate(self.nodes):
            leaf_info = (
                f" [Leaf: atom_site_element={node.atom_site_element}, site_element={node.site_element}]"
                if isinstance(node, HLLeafNode)
                else ""
            )
            root_info = " (Root)" if node in self.roots else ""
            print(f"Node {idx}: {node.dof} dof{root_info}{leaf_info}")


# Example Usage
leaf1 = HLLeafNode(dof=2, atom_site_element="H", site_element="1")
leaf2 = HLLeafNode(dof=1, atom_site_element="O", site_element="2")
leaf3 = HLLeafNode(dof=3, atom_site_element="C", site_element="3")
tree = HierarchicalLinearTree([leaf1, leaf2, leaf3])

# Use LinearTransformer objects to add a parent node
transformer1 = LinearTransformer(
    weights=np.array([[1, 0], [0, 1]]), biases=np.array([1, -1])
)
transformer2 = LinearTransformer(weights=np.array([[1]]), biases=np.array([2]))

# Adding parent node
tree.add_parent(
    child_indices=[0, 1],  # Indices of Leaf1 and Leaf2
    transformers=[transformer1, transformer2],
)

transformer3 = LinearTransformer(
    weights=np.array([[1, 1, 1], [0, 0, 1]]), biases=np.array([0, 0])
)
transformer4 = LinearTransformer(
    weights=np.array([[1, 0], [0, 1]]), biases=np.array([1, 1])
)

# Adding another parent node
tree.add_parent(
    child_indices=[2, 3],  # Indices of Leaf3 and Parent1
    transformers=[transformer3, transformer4],
)

# Display the structure with degrees of freedom and leaf attributes
tree.display_structure()
