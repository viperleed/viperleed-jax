"""Module occ_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-08'

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

    def apply_bounds(self, occ_delta_line):
        super().apply_bounds(occ_delta_line)

        # resolve targets
        _, target_roots_and_primary_leaves = self._get_leaves_and_roots(
            occ_delta_line.targets
        )
        # extract element ranges from the delta line
        element_ranges = {
            elem_token.symbol: range_token
            for (elem_token, range_token) in occ_delta_line.element_ranges
        }

        for root, primary_leaf in target_roots_and_primary_leaves.items():
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

            leaf_range_zonotope = Zonotope(
                basis=np.array([[1.0]]),  # 1D zonotope
                ranges=occ_range,
                offset=None,
            )

            root_to_leaf_transformer = root.transformer_to_descendent(
                primary_leaf
            )
            leaf_to_root_transformer = root_to_leaf_transformer.pseudo_inverse()
            root_range_zonotope = leaf_range_zonotope.apply_affine(
                leaf_to_root_transformer
            )
            implicit_constraint_node = ImplicitLinearConstraintNode(
                child=root,
                name=occ_delta_line.raw_line,
                child_zonotope=root_range_zonotope,
            )
            self.nodes.append(implicit_constraint_node)

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
        roots_to_link = list(set(roots_to_link))

        while roots_to_link:
            # take the first root and find all leaves that are children of this root
            primary_root = roots_to_link[0]

            # select all roots that have a leaf with the same atom number
            atom_nums = [leaf.atom.num for leaf in primary_root.children]
            shared_occ_roots = [
                root
                for root in roots_to_link
                if any(leaf.atom.num in atom_nums for leaf in root.children)
            ]

            # none of the roots should have dof > 1
            # failsafe in case we implement dof > 1 in the future
            if any(root.dof > 1 for root in shared_occ_roots):
                raise ValueError(
                    'Cannot link occupations with dof > 1. '
                    'This is not supported by the current implementation.'
                )

            # TODO: could be made into a custom Constraint Node class
            # create the linked constraint node for the primary root
            linked_constraint_node = LinearConstraintNode(
                dof=len(shared_occ_roots) - 1,
                children=shared_occ_roots,
                transformers=_fixed_occ_constraint_linear_map(
                    len(shared_occ_roots), total_occupation
                ),
                name=constraint_line.raw_line,
                layer=DisplacementTreeLayers.User_Constraints,
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
        return raw_values + ~self.leaf_is_dynamic * self._ref_occupations

    def _centered_occupations(self):
        """Return the centered occupations based on the parameters."""
        return self(np.array([0.5] * self.root.dof))

    @property
    def is_centered(self):
        """Check if the occupational tree is centered."""
        super().is_centered
        centered_occupations = self._centered_occupations()
        return np.allclose(self._ref_occupations, centered_occupations)


def _fixed_occ_constraint_linear_map(n_children, total_occ):
    """Create a linear map for the fixed occupation constraint."""
    if n_children < 2:
        raise ValueError(
            'At least two children are required for a total occupation constraint.'
        )

    if total_occ < 0 or total_occ > 1:
        raise ValueError('Total occupation must be between 0 and 1.')

    transformers = []
    for i in range(n_children - 1):
        arr = np.zeros([1, n_children - 1])
        arr[0, i] = 1.0 / total_occ
        bias = np.zeros([1])
        transformers.append(AffineTransformer(arr, bias))
    # The last child is minus the sum of the others

    arr = -np.ones([1, n_children - 1]) / (n_children - 1) / total_occ
    bias = np.array([total_occ])
    transformers.append(AffineTransformer(arr, bias))

    return transformers
