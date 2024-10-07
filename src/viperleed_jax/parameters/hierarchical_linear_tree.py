import numpy as np


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


class HLEdge:
    def __init__(self, parent, child, weight_matrix, bias_vector):
        """
        Define a linear edge between parent and child nodes.
        - `weight_matrix`: A matrix to multiply the child's degrees of freedom.
        - `bias_vector`: A bias vector to add after multiplication.
        """
        self.parent = parent
        self.child = child
        self.weight_matrix = weight_matrix
        self.bias_vector = bias_vector

    def __repr__(self):
        return f"HLEdge(Child dof={self.child.dof} -> Parent dof={self.parent.dof}, W={self.weight_matrix.shape}, b={self.bias_vector.shape})"


class HierarchicalLinearTree:
    def __init__(self, leaf_dofs):
        """
        Initialize the tree with a list of leaf nodes specified by their degrees of freedom.
        - `leaf_dofs`: A list of integers where each entry represents the degrees of freedom (dof) for a leaf node.
        """
        self.nodes = []  # Store nodes in a list
        self.edges = []  # Store all edges
        self.roots = []  # Keep track of root nodes (initially, all are leaves)

        # Add all specified leaf nodes to the tree
        for dof in leaf_dofs:
            leaf = HLNode(dof)
            self.nodes.append(leaf)
            self.roots.append(leaf)

    def add_parent(self, child_indices, weight_matrices, bias_vectors):
        """
        Add a new parent node that connects to the specified child nodes by their indices.
        The dimensions of the weight matrices must be validated against the child nodes' degrees of freedom.

        Parameters:
        - `child_indices`: List of indices of the child nodes (must be roots).
        - `weight_matrices`: List of weight matrices for each child.
        - `bias_vectors`: List of bias vectors for each child.

        Returns:
        - The newly created parent node.
        """
        if len(child_indices) != len(weight_matrices) or len(
            weight_matrices
        ) != len(bias_vectors):
            raise ValueError(
                "Number of child nodes, weight matrices, and bias vectors must match."
            )

        # Retrieve child nodes by their indices
        children = [self.roots[i] for i in child_indices]

        # Calculate the dof for the new parent node
        parent_dof = weight_matrices[0].shape[0]
        for matrix, child in zip(weight_matrices, children):
            if matrix.shape[1] != child.dof:
                raise ValueError(
                    f"Weight matrix columns ({matrix.shape[1]}) must match child node dof ({child.dof})."
                )
            if matrix.shape[0] != parent_dof:
                raise ValueError(
                    "All weight matrices must have the same number of rows (parent node dof)."
                )

        # Create the new parent node
        parent_node = HLNode(parent_dof)
        self.nodes.append(parent_node)
        self.roots.append(parent_node)

        # Connect the parent to each child and remove children from root list
        for child, W, b in zip(children, weight_matrices, bias_vectors):
            self.add_edge(child, parent_node, W, b)
            self.roots.remove(child)

        return parent_node

    def add_edge(self, child, parent, weight_matrix, bias_vector):
        """
        Create a linear edge between child and parent nodes.
        Ensure that the dimensions match between the child's dof and the weight matrix.
        """
        # Add the child to the parent's children list
        parent.add_child(child)

        # Create the edge and store it
        edge = HLEdge(parent, child, weight_matrix, bias_vector)
        self.edges.append(edge)

    @property
    def n_roots(self):
        """Return the number of root nodes."""
        return len(self.roots)

    def display_structure(self):
        """Display the structure of the tree with node degrees of freedom (dof)."""
        print(f"Tree Structure with {self.n_roots} root node(s):")
        for idx, node in enumerate(self.nodes):
            root_info = " (Root)" if node in self.roots else ""
            print(f"Node {idx}: {node.dof} dof{root_info}")


# Example Usage
# Initialize the tree with leaf nodes specified by their degrees of freedom
leaf_dofs = [2, 1, 3]  # Three leaf nodes with 2, 1, and 3 degrees of freedom
tree = HierarchicalLinearTree(leaf_dofs)

# Add parent nodes by specifying child node indices (must be roots at this point)
tree.add_parent(
    child_indices=[0, 1],  # Indices of Leaf1 and Leaf2
    weight_matrices=[np.array([[1, 0], [0, 1]]), np.array([[1]])],
    bias_vectors=[np.array([1, -1]), np.array([2])],
)

tree.add_parent(
    child_indices=[2, 3],  # Indices of Leaf3 and Parent1
    weight_matrices=[
        np.array([[1, 1, 1], [0, 0, 1]]),
        np.array([[1, 0], [0, 1]]),
    ],
    bias_vectors=[np.array([0, 0]), np.array([1, 1])],
)

# Display the structure with degrees of freedom
tree.display_structure()
