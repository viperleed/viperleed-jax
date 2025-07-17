"""Module geo_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-08-30'


import numpy as np

from .displacement_tree_layers import DisplacementTreeLayers
from .linear_transformer import LinearMap
from .nodes import (
    AtomicLinearNode,
    ImplicitLinearConstraintNode,
    LinearConstraintNode,
)
from .reduced_space import Zonotope
from .tree import (
    DisplacementTree,
)


class GeoLeafNode(AtomicLinearNode):
    """Represents a leaf node with geometric parameters."""

    def __init__(self, atom):
        dof = 3
        super().__init__(dof=dof, atom=atom)
        self.symrefm = atom.atom.symrefm
        self._name = f'geo (At_{self.num},{self.site},{self.element})'


class GeoConstraintNode(LinearConstraintNode):
    """Base constraint node for geometric parameters."""

    def __init__(
        self,
        dof,
        children,
        transformers,
        layer,
        name='unnamed',
    ):
        self.dof = dof

        if transformers is None:
            raise ValueError(
                'Transformers must be provided for '
                'geometric constraint nodes.'
            )
        super().__init__(
            dof=dof,
            name=name,
            children=children,
            transformers=transformers,
            layer=layer,
        )


class GeoSymmetryConstraint(GeoConstraintNode):
    """Constraint node for symmetry constraints on geometric parameters.

    Symmetry constraints are the first layer of constraints applied to the
    geometric leaf nodes. They are link atoms based on symmetry relations and
    always reduce the number of free parameters to a maximum of 3.
    All transformers must therefore have in_dim=3 and the bias must be zero.
    Further properties of the transformer are determined by the symmetry and
    the allowed directions of movement.
    The viperleed.calc symmetry recognition gives atoms the attributes `symrefm`
    and `freedir`. `freedir` determines the allowed directions of movement, and
    `symrefm` is the symmetry operation that links the atoms that are symmetry
    linked.
    """

    def __init__(self, children):
        # transformers may not be provided, as they must be determined from the
        # symmetry operations
        if not children:
            raise ValueError('Symmetry constraints must have children')

        # make sure that all children are leaf nodes
        if not all(isinstance(child, GeoLeafNode) for child in children):
            raise ValueError(
                'Symmetry constraints can only be applied to '
                'geometric leaf nodes.'
            )

        # check that all children are in the same linklist
        linklist = children[0].atom.atom.linklist
        if not all(child.atom.atom in linklist for child in children):
            raise ValueError(
                'Symmetry linked atoms must be in the same ' 'linklist'
            )

        # irrespective of the symmetry the transformer bias so we use a map
        transformers = []

        # how to proceed is determined by the freedir attribute
        # NB: this is a weird attribute and will be changed in the future
        # Currently freedir can be either an int (specifically 0 or 1, implying
        # z-only or completely free movement) or a 1D array of shape (2,)
        # (implying 1D in-plane movement in addition to z)
        freedir = children[0].atom.atom.freedir
        if isinstance(freedir, int) and freedir == 0:
            # check that all children have the same freedir
            if not all(child.atom.atom.freedir == 0 for child in children):
                raise ValueError(
                    'All symmetry linked atoms must have the same '
                    'freedir attribute.'
                )
            # z-only movement
            dof = 1
            name = 'Symmetry (z-only)'
            for child in children:
                # set the symmetry linking matrix and direct transfer of z
                weights = np.array([1.0, 0.0, 0.0]).reshape((3, 1))
                transformers.append(LinearMap(weights, (3,)))

        elif isinstance(freedir, int) and freedir == 1:
            # check that all children have the same freedir
            if not all(child.atom.atom.freedir == 1 for child in children):
                raise ValueError(
                    'All symmetry linked atoms must have the same '
                    'freedir attribute.'
                )
            # free in-plane movement in addition to z
            dof = 3
            name = 'Symmetry (free)'

            for child in children:
                # set the symmetry linking matrix and direct transfer of z
                weights = np.identity(3)
                weights[1:3, 1:3] = child.symrefm
                transformers.append(LinearMap(weights, (3,)))

        elif freedir.shape == (2,):
            # check that all children have the same freedir
            if not all(
                child.atom.atom.freedir.shape == (2,) for child in children
            ):
                raise ValueError(
                    'All symmetry linked atoms must have the same '
                    'freedir attribute.'
                )
            # 1D in-plane movement in addition to z
            dof = 2
            name = 'Symmetry (1D in-plane)'

            # plane unit cell
            ab_cell = children[0].atom.atom.slab.ab_cell

            for child in children:
                at_freedir = child.atom.atom.freedir
                movement_vector = ab_cell.T @ at_freedir
                movement_vector = child.symrefm @ movement_vector
                movement_vector = movement_vector / np.linalg.norm(
                    movement_vector
                )
                weights = np.array([[1.0, 0.0], [0.0, np.nan], [0.0, np.nan]])
                weights[1:3, 1] = movement_vector
                transformers.append(LinearMap(weights, (3,)))
        else:
            raise ValueError(
                'freedir attribute must be 0, 1, or have shape (2,)'
            )

        super().__init__(
            dof=dof,
            children=children,
            transformers=transformers,
            name=name,
            layer=DisplacementTreeLayers.Symmetry,
        )


class GeoLinkedConstraint(GeoConstraintNode):
    """Class for explicit links of geometric parameters."""

    # TODO: this needs to be adapted if we allow partial directional linking
    # e.g. linking z coordinates of two atoms, but not x and y
    # TODO: furthermore: linking of atoms in the same direction, but with
    #       different magnitudes would also be problematic! – check math for this
    #       case. I'm not sure if the propagators are linear with respect to the
    #       length of the displacement vector
    # If either of these is implemented, the children will not share a propagator
    def __init__(self, children, name):
        # check that all children have the same dof
        if len({child.dof for child in children}) != 1:
            raise ValueError('Children must have the same dof.')
        dof = children[0].dof

        # transformers can be identity
        transformers = [LinearMap(np.eye(dof), (dof,)) for _ in children]
        super().__init__(
            dof=dof,
            children=children,
            transformers=transformers,
            name=f"CONSTRAIN '{name}'",
            layer=DisplacementTreeLayers.User_Constraints,
        )


class GeoTree(DisplacementTree):
    def __init__(self, atom_basis):
        super().__init__(
            atom_basis,
            name='Geometric Parameters',
            root_node_name='geo root',
            perturbation_type='geo',
        )

        # set the leaf and constraint node classes
        self._leaf_node = GeoLeafNode
        self._constraint_node = GeoConstraintNode

        unlinked_site_el_nodes = [node for node in self.leaves if node.is_root]
        for node in unlinked_site_el_nodes:
            self.nodes.append(GeoSymmetryConstraint(children=[node]))

    def apply_bounds(self, geo_delta_line):
        super().apply_bounds(geo_delta_line)

        # resolve targets
        _, target_roots_and_primary_leaves = self._get_leaves_and_roots(
            geo_delta_line.targets
        )

        # get vector and range information from the GEO_DELTA line
        n_vectors = len(geo_delta_line.direction.vectors_zxy)

        ranges = np.vstack(
            [
                np.full(n_vectors, geo_delta_line.range.start),
                np.full(n_vectors, geo_delta_line.range.stop),
            ]
        )

        leaf_range_zonotope = Zonotope(
            basis=geo_delta_line.direction.vectors_zxy,
            ranges=ranges,
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
                name=geo_delta_line.raw_line,
                child_zonotope=root_range_zonotope,
            )
            self.nodes.append(implicit_constraint_node)

    def _post_process_values(self, raw_values):
        # reshape to (n_atoms, 3)
        return super()._post_process_values(raw_values).reshape(-1, 3)
