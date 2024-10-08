import numpy as np
import jax.numpy as jnp
import anytree
from anytree import Node


from .linear_transformer import LinearTransformer

# Enable checks for the anytree library – we don't deal with huge trees so this
# should not be a performance issue.
anytree.config.ASSERTIONS = True


class HLNode(Node):
    """Base class for hierarchical linear tree nodes."""
    separator = "/"

    def __init__(self, dof, name=None, parent=None, children=None):
        self.dof = dof  # Number of degrees of freedom
        self._transformer = None  # LinearTransformer object
        self.name = f"({self.dof}) {name}" if name else f"({self.dof})"
        self.parent = parent
        if self.children:
            self.children = children

    def set_transformer(self, transformer):

        # check if the transformer is valid
        if not isinstance(transformer, LinearTransformer):
            raise TypeError(
                f"Transformer must be an instance of LinearTransformer. "
                f"Invalid transformer: {transformer}"
            )

        if transformer.out_dim != self.dof:
            raise ValueError(
                f"Transformer output dimension ({transformer.out_dim}) "
                f"must match child dof ({self.dof})."
            )
        self._transformer = transformer

    @property
    def transformer(self):
        if self._transformer is None:
            raise ValueError("Node does not have a transformer.")
        return self._transformer

    def _pre_attach(self, parent):

        # check that the parent is a ConstraintNode
        if not isinstance(parent, HLConstraintNode):
            raise TypeError(
                f"Parent must be an instance of HLConstraintNode. "
                f"Invalid parent: {parent}"
            )

        # check that the transformer dimensions match
        if self.transformer.in_dim != parent.dof:
            raise ValueError(
                f"Transformer input dimension ({self.transformer.in_dim}) "
                f"must match parent dof ({parent.dof})."
            )

    def _pre_detach(self, parent):
        raise RuntimeError(
            "The hierarchical linear tree does not support " "detaching nodes."
        )


class HLLeafNode(HLNode):

    def __init__(self, dof, name=None, parent=None, children=[]):
        if children:
            raise ValueError("Leaf nodes cannot have children.")
        super().__init__(dof=dof, name=name, parent=parent)


class HLConstraintNode(HLNode):
    """Base class for hierarchical linear tree constraint nodes.

    Any non-leaf node in the tree is a constraint node. Constraint nodes must
    be initialized with a list of child nodes and their corresponding
    transformers."""

    def __init__(self, dof, name=None, children=[], transformers=None):
        _children = list(children)
        super().__init__(dof=dof, name=name)  # Initialize the base class

        if len(_children) == 0:
            raise ValueError("Constraint nodes must be initialized with "
                             "at least one child node.")
        # check that children are roots
        if any(not child.is_root for child in _children):
            raise ValueError("Children must be root nodes.")

        # if no transformers are provided, check if all children already
        # have transformers
        if transformers is None:
            if any(child.transformer is None for child in _children):
                raise ValueError("All children must have transformers.")
            transformers = [child.transformer for child in _children]

        # check that the number of children and transformers match
        if len(_children) != len(transformers):
            raise ValueError(
                f"Number of children ({len(_children)}) must match "
                f"number of transformers ({len(transformers)})."
            )

        # if everything is fine, set the transformers
        for child, transformer in zip(_children, transformers):
            child.set_transformer(transformer)

        # if that went well, set the children
        for child in _children:
            child.parent = self


def create_subtree_root(subtree_nodes, name):
    subtree_root_nodes = [node for node in subtree_nodes if node.is_root]
    if not subtree_root_nodes:
        raise ValueError("No root nodes found in subtree.")
    root_dof = sum(node.dof for node in subtree_root_nodes)
    transformers = []
    for node in subtree_root_nodes:
        weights = np.zeros((node.dof, root_dof))
        weights[:, : node.dof] = np.identity(node.dof)
        bias = np.zeros(node.dof)
        transformers.append(LinearTransformer(weights, bias, (node.dof,)))
    return HLConstraintNode(
        dof=root_dof,
        name=name,
        children=subtree_root_nodes,
        transformers=transformers,
    )
