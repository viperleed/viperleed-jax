from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp
import anytree
from anytree import Node, RenderTree


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
            raise ValueError(
                "Constraint nodes must be initialized with "
                "at least one child node."
            )
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



class ParameterHLSubtree(ABC):
    """Base class representing a subtree in the hierarchical linear tree.

    Subtrees are used to group nodes for a group of parameters (vib, geo, occ,
    V0r). This allows implementing constraints more easily and makes it possible
    to extract additional information from the tree (e.g. propagator
    transformations).
    """

    def __init__(self, slab, atom_site_elements, site_elements):
        self.slab = slab
        self.atom_site_elements = atom_site_elements
        self.site_elements = site_elements
        self.nodes = []

        self._subtree_root_has_been_created = False
        self.build_subtree()

    def __repr__(self):
        if not self._subtree_root_has_been_created:
            return f"{self.__class__.__name__}({self.name})"
        return RenderTree(self.subtree_root).by_attr()

    @property
    def roots(self):
        return [node for node in self.nodes if node.is_root]

    @property
    def leafs(self):
        return [node for node in self.nodes if node.is_leaf]

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def subtree_root_name(self):
        pass

    @abstractmethod
    def build_subtree(self):
        """Method to build the subtree for the parameter group."""
        pass


    def create_subtree_root(self):
        """Create a root node that aggregates all root nodes in the subtree.y"""
        if self._subtree_root_has_been_created:
            raise ValueError("Subtree root has already been created.")
        self._subtree_root_has_been_created = True
        if not self.roots:
            raise ValueError("No root nodes found in subtree.")
        root_dof = sum(node.dof for node in self.roots)
        transformers = []
        for node in self.roots:
            weights = np.zeros((node.dof, root_dof))
            weights[:, : node.dof] = np.identity(node.dof)
            bias = np.zeros(node.dof)
            transformers.append(LinearTransformer(weights, bias, (node.dof,)))
        self.subtree_root = HLConstraintNode(
            dof=root_dof,
            name=self.subtree_root_name,
            children=self.roots,
            transformers=transformers,
    )
