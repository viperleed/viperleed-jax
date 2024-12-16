"""Module geo_parameters."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-08-30'


import numpy as np

from .displacement_tree_layers import DisplacementTreeLayers
from .functionals import LinearTreeFunctional
from .linear_transformer import LinearMap
from .nodes import AtomicLinearNode, LinearConstraintNode
from .tree import (
    DisplacementTree,
)


class DisplacementFunctional(LinearTreeFunctional):
    def __init__(
        self,
    ):
        super().__init__(name='displacement', transformer_class=LinearMap)


class GeoLeafNode(AtomicLinearNode):
    """Represents a leaf node with geometric parameters."""

    _Z_DIR_ID = 0  # TODO: unify and move to a common place

    def __init__(self, base_scatterer):
        dof = 3
        super().__init__(dof=dof, base_scatterer=base_scatterer)
        self.symrefm = base_scatterer.atom.symrefm
        self._name = f'geo (At_{self.num},{self.site},{self.element})'

    def _update_bounds(self, line):
        # geometric leaf bounds are 3D
        range = line.range
        direction = line.direction

        if direction._fractional:
            raise NotImplementedError('TODO')

        if (
            direction.num_free_directions == 1 and direction._vectors[0][2] == 1
        ):  # TODO: index 2 here needs to be changed to LEED convention
            # z-only movement
            start = np.array([range.start, 0.0, 0.0])
            stop = np.array([range.stop, 0.0, 0.0])
            user_set = [True, False, False]
        else:
            raise NotImplementedError('TODO')
        self._bounds.update_range((start, stop), enforce=user_set)

    def update_offsets(self, line):
        # geometric leaf bounds are 3D
        direction = line.direction

        if direction._fractional:
            raise NotImplementedError('TODO')

        if (
            direction.num_free_directions == 1
            and direction._vectors[0][2]
            == 1  # TODO: index 2 here needs to be changed to LEED convention
        ):
            # z-only movement
            offset = np.array([line.value, 0.0, 0.0])
            user_set = np.array([True, False, False])
        else:
            raise NotImplementedError('TODO')
        self._bounds.update_range(_range=None, offset=offset, enforce=user_set)


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
        linklist = children[0].base_scatterer.atom.linklist
        if not all(
            [child.base_scatterer.atom in linklist for child in children]
        ):
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
        freedir = children[0].base_scatterer.atom.freedir
        if isinstance(freedir, int) and freedir == 0:
            # check that all children have the same freedir
            if not all(
                child.base_scatterer.atom.freedir == 0 for child in children
            ):
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
            if not all(
                child.base_scatterer.atom.freedir == 1 for child in children
            ):
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
                child.base_scatterer.atom.freedir.shape == (2,)
                for child in children
            ):
                raise ValueError(
                    'All symmetry linked atoms must have the same '
                    'freedir attribute.'
                )
            # 1D in-plane movement in addition to z
            dof = 2
            name = 'Symmetry (1D in-plane)'

            # plane unit cell
            ab_cell = children[0].base_scatterer.atom.slab.ab_cell

            for child in children:
                at_freedir = child.base_scatterer.atom.freedir
                movement_vector = ab_cell.T @ at_freedir
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

    # TODO: if we implement linking of nodes with different dof (directional),
    # this needs to be adapted
    # TODO: this also needs to be adapted if we allow partial directional linking
    # e.g. linking z coordinates of two atoms, but not x and y
    # TODO: furthermore: linking of atoms in the same direction, but with
    #       different magnitudes would also be problematic! – check math for this
    #       case. I'm not sure if the propagators are linear with respect to the
    #       length of the displacement vector
    # If either of these is implemented, the children will not share a propagator
    def __init__(self, children, name):
        # check that all children have the same dof
        if len(set(child.dof for child in children)) != 1:
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
        )
        self.displacement_functional = DisplacementFunctional()
        self.functionals.append(self.displacement_functional)


    def _initialize_tree(self):
        # create leaf nodes
        geo_leaf_nodes = [
            GeoLeafNode(base_scatterer)
            for base_scatterer in self.atom_basis
        ]
        self.nodes.extend(geo_leaf_nodes)

        # apply symmetry constraints
        for siteel in self.site_elements:
            site_el_params = [
                node for node in self.leaves if node.site_element == siteel
            ]

        for link in self.atom_basis.symmetry_links:
            # put all linked atoms in the same symmetry group
            nodes_to_link = [
                node for node in self.leaves if node.base_scatterer in link
            ]
            if nodes_to_link:
                self.nodes.append(GeoSymmetryConstraint(children=nodes_to_link))

        unlinked_site_el_nodes = [node for node in self.leaves if node.is_root]
        for node in unlinked_site_el_nodes:
            self.nodes.append(GeoSymmetryConstraint(children=[node]))

    def apply_explicit_constraint(self, constraint_line):
        # self._check_constraint_line_type(constraint_line, "geo")
        *_, selected_roots = self._select_constraint(constraint_line)

        if constraint_line.direction is None:
            # complete linking; requires all root nodes to have the same dof
            if not all(
                node.dof == selected_roots[0].dof for node in selected_roots
            ):
                raise ValueError(
                    'All root nodes must have the same number of free parameters.'
                )
            # create a constraint node for the selected roots
            self.nodes.append(
                GeoLinkedConstraint(
                    children=selected_roots, name=constraint_line.line
                )
            )
        else:
            raise NotImplementedError(
                'Directional geo constraints are not yet supported.'
            )

    #############################
    # Geometry specific methods #
    #############################
    def all_displacements_transformer(self):
        """Return a transformer that gives the displacements for all base
        scatterers."""
        collapsed_transformer = self.collapsed_transformer_scatterer_order
        collapsed_transformer.out_reshape = (-1, 3)
        return collapsed_transformer

    def dynamic_displacements_transformers(self):
        """Return a list of transformers that give the reference displacements
        for the dynamic propagators."""
        return [
            self.root.transformer_to_descendent(node)
            for node in self.displacement_functional.dynamic_reference_nodes
        ]

    @property
    def static_propagator_inputs(self):
        """Return the displacements for the static reference propagators."""
        return self.displacement_functional.static_reference_nodes_values

    @property
    def propagator_map(self):
        """Return a mapping of base scatterers to propagators."""
        return self.displacement_functional.static_dynamic_map

    @property
    def leaf_plane_symmetry_operations(self):
        """Return the in-plane symmetry operations for each leaf in respect to the
        reference displacement (the one for which the propagator is calculated).
        """
        return tuple(
            sym_op.weights[1:, 1:]
            for sym_op in self.displacement_functional._arg_transformers
        )

    @property
    def n_dynamic_propagators(self):
        return self.displacement_functional.n_dynamic_values

    @property
    def n_static_propagators(self):
        return self.displacement_functional.n_static_values
