"""Module vib_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-09'

import numpy as np

from .displacement_tree_layers import DisplacementTreeLayers
from .linear_transformer import AffineTransformer, LinearMap
from .nodes import (
    AtomicLinearNode,
    ImplicitLinearConstraintNode,
    LinearConstraintNode,
)
from .reduced_space import Zonotope
from .tree import (
    DisplacementTree,
)


class VibLeafNode(AtomicLinearNode):
    """Represents a leaf node with vibrational parameters."""

    def __init__(self, atom):
        dof = 1
        super().__init__(dof=dof, atom=atom)
        self._name = f'vib (At_{self.num},{self.site},{self.element})'
        self.ref_vib_amp = atom.atom.site.vibamp[self.element]


class VibConstraintNode(LinearConstraintNode):
    """Represents a constraint node for vibrational parameters."""

    def __init__(self, children, name, layer, dof=1, transformers=None):
        if dof != 1:
            raise ValueError('Vibrational constraints must have dof=1.')

        if transformers is None:
            # default to identity
            transformers = [LinearMap(np.eye(1), (1,)) for _ in children]
        super().__init__(
            dof=dof,
            name=name,
            children=children,
            layer=layer,
            transformers=transformers,
        )


class VibSymmetryConstraint(VibConstraintNode):
    """Class for linking vibrations of symmetry equivalent atoms."""

    def __init__(self, children):
        # check that all children are leaf nodes and share a site-element
        if not all(isinstance(child, VibLeafNode) for child in children):
            raise ValueError('Children must be VibLeaf nodes.')
        if not all(
            child.site_element == children[0].site_element for child in children
        ):
            raise ValueError('Children must have the same site-element.')
        self.ref_vib_amp = children[0].ref_vib_amp
        self.site_element = children[0].site_element
        dof = 1

        super().__init__(
            dof=dof,
            children=children,
            transformers=[
                AffineTransformer(
                    weights=np.array([[1.0]]), biases=np.array([0.0])
                )
                for _ in children
            ],
            name='Symmetry',
            layer=DisplacementTreeLayers.Symmetry,
        )


class VibTree(DisplacementTree):
    def __init__(self, atom_basis):
        super().__init__(
            atom_basis,
            name='Vibrational Parameters',
            root_node_name='vib root',
            perturbation_type='vib',
        )

    def _initialize_tree(self):
        leaf_nodes = [VibLeafNode(ase) for ase in self.atom_basis]

        self.nodes.extend(leaf_nodes)

        # vibrations need to fulfill symmetry constraints
        for link in self.atom_basis.atom_number_symmetry_links:
            # put all linked atoms in the same symmetry group

            nodes_to_link = [node for node in leaf_nodes if node.num in link]
            if not nodes_to_link:
                continue
            symmetry_node = VibSymmetryConstraint(
                children=nodes_to_link,
            )
            self.nodes.append(symmetry_node)

        unlinked_vib_leaves = [node for node in leaf_nodes if not node.parent]
        for node in unlinked_vib_leaves:
            dummy_symmetry_node = VibSymmetryConstraint(
                children=[node],
            )
            self.nodes.append(dummy_symmetry_node)

    @property
    def _ref_vib_amplitudes(self):
        """Return the reference vibrational amplitudes for all leaves."""
        return np.array([leaf.ref_vib_amp for leaf in self.leaves])

    def apply_bounds(self, vib_delta_line):
        super().apply_bounds(vib_delta_line)

        # resolve targets
        _, target_roots_and_primary_leaves = self._get_leaves_and_roots(
            vib_delta_line.targets
        )

        vib_range = np.array(
            [[vib_delta_line.range.start, vib_delta_line.range.stop]]
        ).T

        leaf_range_zonotope = Zonotope(
            basis=np.array([[1.0]]),  # 1D zonotope
            ranges=vib_range,
            offset=None,
        )

        for root, primary_leaf in target_roots_and_primary_leaves.items():
            root_to_leaf_transformer = root.transformer_to_descendent(
                primary_leaf
            )
            leaf_to_root_transformer = root_to_leaf_transformer.pseudo_inverse()
            root_range_zonotope = leaf_range_zonotope.apply_affine(
                leaf_to_root_transformer
            )
            implicit_constraint_node = ImplicitLinearConstraintNode(
                child=root,
                name=vib_delta_line.raw_line,
                child_zonotope=root_range_zonotope,
            )
            self.nodes.append(implicit_constraint_node)

    def _post_process_values(self, raw_values):
        # add reference calculation vibrational amplitudes to raw values
        # (vib deltas)
        return raw_values + self._ref_vib_amplitudes
