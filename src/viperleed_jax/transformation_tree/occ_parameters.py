"""Module occ_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-08'
__copyright__ = 'Copyright (c) 2023-2025 ViPErLEED developers'
__license__ = 'GPLv3+'

import numpy as np

from .displacement_tree_layers import DisplacementTreeLayers
from .linear_transformer import AffineTransformer
from .nodes import (
    AtomicLinearNode,
    ImplicitLinearConstraintNode,
    LinearConstraintNode,
)
from .reduced_space import Zonotope
from .tree import (
    ConstructionOrder,
    DisplacementTree,
)


class OccLeafNode(AtomicLinearNode):
    """Represents a leaf node with occupational parameters."""

    def __init__(self, atom):
        dof = 1
        super().__init__(dof=dof, atom=atom)
        self._name = f'occ (At_{self.num},{self.site},{self.element})'
        self.ref_occ = atom.atom.site.occ[self.element]
        # self.occ_range = (0.0, 1.0)


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

    def __init__(self, children):
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
            name='Symmetry',
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


class OccTotalOccupationConstraint(OccConstraintNode):
    """Constraint enforcing a fixed total occupation across children.

    For n children, introduces n-1 free parameters; the last child's
    occupation is determined by the fixed total.
    """

    def __init__(self, children, total_occupation, name):
        # Require identical dof and scalar nodes
        if len({child.dof for child in children}) != 1:
            raise ValueError('Children must have the same dof.')
        if children[0].dof != 1:
            raise ValueError(
                'Total occupation constraint only supports dof == 1.'
            )

        self.total_occupation = total_occupation

        n_children = len(children)
        if n_children < 2:
            raise ValueError(
                'At least two children are required for a total occupation constraint.'
            )

        if not (0.0 <= total_occupation <= 1.0):
            raise ValueError('Total occupation must be between 0 and 1.')
        if total_occupation == 0.0:
            raise ValueError('Total occupation cannot be zero.')

        dof = n_children - 1

        # Build the transformer matrices
        # Each of the first n-1 children gets one parameter directly
        # The last child gets the negative sum of all previous parameters
        # such that rho_n = 1 - sum_{i=1}^{n-1} rho_i
        base_weights = np.eye(dof)
        base_biases = np.zeros(dof)
        transformers = [
            AffineTransformer(
                base_weights[i : i + 1, :], np.array([base_biases[i]])
            )
            for i in range(dof)
        ]

        last_weights = -np.ones((1, dof)) / dof
        last_bias = np.array([1.0])
        transformers.append(AffineTransformer(last_weights, last_bias))

        super().__init__(
            dof=dof,
            children=children,
            transformers=transformers,
            name=name,
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
        self._leaf_node = OccLeafNode
        self._symmetry_node = OccSymmetryConstraint

        super().__init__(
            atom_basis,
            name='Occupational Parameters',
            root_node_name='occ root',
            perturbation_type='occ',
        )

    def _zonotope_from_bounds_line(self, occ_delta_line, primary_leaf):
        # extract element ranges from the delta line
        element_ranges = {
            elem_token.symbol: range_token
            for (elem_token, range_token) in occ_delta_line.element_ranges
        }
        # Leaf nodes represent the occupation with a single element,
        # so we can use 1D zonotopes
        occ_range = np.array(
            [
                [
                    element_ranges[primary_leaf.element].start,
                    element_ranges[primary_leaf.element].stop,
                ]
            ]
        ).T

        return Zonotope(
            basis=np.array([[1.0]]),  # 1D zonotope
            ranges=occ_range,
            offset=None,
        )

    def apply_explicit_constraint(self, constraint_line):
        """Apply an explicit constraint to the occupational parameters.

        If the constraint is a normal linear constraint, dispatch to the
        superclass method. If the constraint is a total occupation constraint,
        handle it separately.
        """
        if not constraint_line.is_total_occupation:
            super().apply_explicit_constraint(constraint_line)
            return

        # get total occupation form the constraint line tokens
        total_occupation = constraint_line.linear_operation.total_occupation

        # handle total occupation constraint
        self._check_construction_order(ConstructionOrder.EXPLICIT_CONSTRAINT)

        # select the atoms that are to be linked
        to_link_mask = self.atom_basis.target_selection_mask(
            constraint_line.targets
        )
        leaves_to_link = self.leaves[to_link_mask]
        roots_to_link = [leaf.root for leaf in leaves_to_link]
        # remove duplicate roots
        roots_to_link = list({root: None for root in roots_to_link}.keys())

        while roots_to_link:
            # take the first root and find all leaves that are children
            primary_root = roots_to_link[0]

            # select all roots that have a leaf with the same atom number
            atom_nums = [leaf.atom.num for leaf in primary_root.leaves]
            shared_occ_roots = [
                root
                for root in roots_to_link
                if any(leaf.atom.num in atom_nums for leaf in root.leaves)
            ]

            if len(shared_occ_roots) == 1:
                # all nodes are already linked somehow...
                msg = (
                    f'The total occupation constraint '
                    f'"{constraint_line.raw_line}" is trying to link (at '
                    'least) two occupations that are already linked.'
                )
                raise ValueError(msg)

            # none of the roots should have dof > 1
            # failsafe in case we implement dof > 1 in the future
            if any(root.dof > 1 for root in shared_occ_roots):
                raise ValueError(
                    'Cannot link occupations with dof > 1. '
                    'This is not supported by the current implementation.'
                )

            # create the total-occupation constraint node for the primary root
            linked_constraint_node = OccTotalOccupationConstraint(
                children=shared_occ_roots,
                total_occupation=total_occupation,  # store total occupation
                name=constraint_line.raw_line,
            )

            # remove linked roots from the list of roots to link
            roots_to_link = [
                root for root in roots_to_link if root not in shared_occ_roots
            ]
            # add the linked constraint node to the tree
            self.nodes.append(linked_constraint_node)

    @property
    def _ref_occupations(self):
        """Return the reference occupations for all leaves."""
        return np.array([leaf.ref_occ for leaf in self.leaves])

    def _post_process_values(self, raw_values):
        # For any nodes that are not dynamic, we return the reference
        # occupations instead of the raw values.
        return (
            self.leaf_is_dynamic * raw_values
            + ~self.leaf_is_dynamic * self._ref_occupations
        )

    def _centered_occupations(self):
        """Return the centered occupations based on the parameters."""
        return self(np.array([0.5] * self.root.dof))

    def is_centered(self):
        """Check if the occupational tree is centered."""
        super().is_centered()
        centered_occupations = self._centered_occupations()
        return np.allclose(self._ref_occupations, centered_occupations)

    @property
    def ref_calc_values(self):
        return self._ref_occupations
