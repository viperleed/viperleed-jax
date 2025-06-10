"""Module occ_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-08'

import numpy as np

from .displacement_tree_layers import DisplacementTreeLayers
from .tree import (
    DisplacementTree,
)
from .linear_transformer import AffineTransformer
from .nodes import AtomicLinearNode, LinearConstraintNode


class OccLeafNode(AtomicLinearNode):
    """Represents a leaf node with occupational parameters."""

    def __init__(self, atom):
        dof = 1
        super().__init__(dof=dof, atom=atom)
        self._name = f'occ (At_{self.num},{self.site},{self.element})'

        # apply reference occupation as non-enforced bounds
        # TODO: get non 100% reference occupation? Where is that stored?
        self._bounds.update_offset(offset=1.0, enforce=False)

    def _update_bounds(self, line):
        # occupational leaves are 1D, so bounds are scalars
        range = line.range
        self._bounds.update_range((range.start, range.stop), enforce=True)

    def update_offsets(self, line):
        offset = line.value
        self._bounds.update_offset(offset=offset, enforce=True)


class OccConstraintNode(LinearConstraintNode):
    """Represents a constraint node for occupational parameters."""

    def __init__(self, dof, children, name, layer, transformers=None):
        self.dof = dof

        if transformers is None:
            raise ValueError('Transformers must be provided.')
        super().__init__(
            dof=dof,
            name=name,
            children=children,
            layer=layer,
            transformers=transformers,
        )


class OccSymmetryConstraint(OccConstraintNode):
    """Constraint for enforcing symmetry in occupation."""

    def __init__(self, children, name):
        # Check that all children have the same dof
        if len({child.dof for child in children}) != 1:
            raise ValueError('Children must have the same dof.')

        dof = children[0].dof

        transformers = []
        for _ in children:
            weights = np.identity(dof)
            bias = np.zeros(dof)
            transformers.append(AffineTransformer(weights, bias, (dof,)))
        super().__init__(
            dof=dof,
            name=name,
            children=children,
            transformers=transformers,
            layer=DisplacementTreeLayers.Symmetry,
        )


class OccLinkedConstraint(OccConstraintNode):
    """Class for explicit links of occupational parameters."""

    def __init__(self, children, name):
        # check that all children have the same dof
        if len({child.dof for child in children}) != 1:
            raise ValueError('Children must have the same dof.')
        dof = children[0].dof

        # transformers can be identity
        transformers = [
            AffineTransformer(np.eye(dof), np.zeros(dof), (dof,))
            for _ in children
        ]
        super().__init__(
            dof=dof,
            children=children,
            transformers=transformers,
            name=f"CONSTRAIN '{name}'",
            layer=DisplacementTreeLayers.User_Constraints,
        )


class OccTree(DisplacementTree):
    r"""Tree for occupational parameters.

    TODO

    Parameters
    ----------
    atom_basis: AtomBasis
        The atom basis to be used for the tree.

    Notes
    -----
    In addition to symmetry, chemical parameters (occupations) MUST fulfill the
    physical constraint that the sum of occupations for each scattering site
    must be less than or equal to 1. This is currently NOT enforced in the tree,
    and MUST be enforced in the calculator to ensure physically valid
    configurations.
    The reason we do not enforce this constraint in the tree at this time is
    that the condition \sum{c_i} \leq 1 is inherently non-linear, and
    therefore not easily compatible with the current implementation that builds
    on the assumption of affine transformations.

    To enforce the constraint, we need to distinguish between two cases:
    1. The sum of occupations is less than or equal to 1: In this case, the
       occupations are already valid and do not need to be modified.
    2. The sum of occupations is greater than 1: In this case, we need to
       project the occupations onto the hyperplane defined by the constraint
       \sum{c_i} = 1.
    """

    def __init__(self, atom_basis):
        super().__init__(
            atom_basis,
            name='Occupational Parameters',
            root_node_name='occ root',
            perturbation_type='occ',
        )

    def _initialize_tree(self):
        # initially, every atom-site-element has a free chemical weight
        # to allow for (partial) vacancies
        occ_leaf_nodes = [OccLeafNode(ase) for ase in self.atom_basis]
        self.nodes.extend(occ_leaf_nodes)

        # occupational parameters need to fulfill symmetry constraints

        for link in self.atom_basis.atom_number_symmetry_links:
            # put all linked atoms in the same symmetry group

            nodes_to_link = [node for node in occ_leaf_nodes if node.num in link]
            if not nodes_to_link:
                continue
            symmetry_node = OccSymmetryConstraint(
                children=nodes_to_link, name='Symmetry'
            )
            self.nodes.append(symmetry_node)

        unlinked_site_el_nodes = [node for node in occ_leaf_nodes if node.is_root]
        for node in unlinked_site_el_nodes:
            symmetry_node = OccSymmetryConstraint(
                children=[node], name='Symmetry'
            )
            self.nodes.append(symmetry_node)

    def apply_explicit_constraint(self, constraint_line):
        _, selected_roots = self._select_constraint(constraint_line)

        if not all(
            node.dof == selected_roots[0].dof for node in selected_roots
        ):
            raise ValueError(
                'All root nodes must have the same number of free parameters.'
            )
        # create a constraint node for the selected roots
        self.nodes.append(
            OccLinkedConstraint(
                children=selected_roots, name=constraint_line.line
            )
        )
