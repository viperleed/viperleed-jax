"""Module vib_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-09'
__copyright__ = 'Copyright (c) 2023-2025 ViPErLEED developers'
__license__ = 'GPLv3+'

import numpy as np

from viperleed_jax.lib.math import EPS
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
from viperleed.calc import LOGGER as logger


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
        self._leaf_node = VibLeafNode
        self._symmetry_node = VibSymmetryConstraint

        super().__init__(
            atom_basis,
            name='Vibrational Parameters',
            root_node_name='vib root',
            perturbation_type='vib',
        )

    @property
    def _ref_vib_amplitudes(self):
        """Return the reference vibrational amplitudes for all leaves."""
        return np.array([leaf.ref_vib_amp for leaf in self.leaves])

    def _zonotope_from_bounds_line(self, vib_delta_line, primary_leaf):
        vib_range = np.array(
            [[vib_delta_line.range.start, vib_delta_line.range.stop]]
        ).T

        # check if the range would go below zero amplitude
        primary_leaf_ref_vib_amp = primary_leaf.ref_vib_amp
        if any(primary_leaf_ref_vib_amp + vib_range < 0):
            msg = (
                f'Vibrational range {vib_delta_line} would lead to '
                'below zero amplitude. Please adjust the range.'
            )
            logger.error(msg)
            raise ValueError(msg)

        return Zonotope(
            basis=np.array([[1.0]]),  # 1D zonotope
            ranges=vib_range,
            offset=None,
        )

    def _post_process_values(self, raw_values):
        # add reference calculation vibrational amplitudes to raw values
        # (vib deltas)
        return raw_values + self._ref_vib_amplitudes

    def is_centered(self):
        """Check if the vibrational tree is centered."""
        super().is_centered()
        centered_vibrations = self(np.array([0.5] * self.root.dof))
        return (
            np.sum(np.abs(centered_vibrations - self._ref_vib_amplitudes)) < EPS
        )

    @property
    def ref_calc_values(self):
        """Return the reference calculation values for all leaves."""
        return np.array([leaf.ref_vib_amp for leaf in self.leaves])
