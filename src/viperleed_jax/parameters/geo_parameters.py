import jax
from jax import numpy as jnp
import numpy as np
from anytree.walker import Walker

from viperleed_jax.parameters.base_parameters import (
    BaseParam,
    Params,
    ConstrainedDeltaParam,
    Bound,
)
from viperleed_jax import atomic_units
from viperleed_jax.files.displacements.lines import ConstraintLine

from .linear_transformer import LinearTransformer
from .hierarchical_linear_tree import HLLeafNode, HLConstraintNode
from .hierarchical_linear_tree import ParameterHLSubtree
from .hierarchical_linear_tree import HLTreeLayers
from .linear_transformer import LinearTransformer, stack_transformers


class GeoHLLeafNode(HLLeafNode):
    """Represents a leaf node with geometric parameters."""
    _Z_DIR_ID = 0 # TODO: unify and move to a common place

    def __init__(self, base_scatterer):
        dof = 3
        self.symrefm = base_scatterer.atom.symrefm
        self.layer = base_scatterer.atom.layer.num
        self.element = base_scatterer.site_element.element
        self.site = base_scatterer.site_element.site
        self.site_element = base_scatterer.site_element
        self.base_scatterer = base_scatterer
        self.num = base_scatterer.num
        self.name = f"geo (At_{self.num},{self.site},{self.element})"
        super().__init__(dof=dof, name=self.name)

    @property
    def symmetry_linking_matrix(self):
        raise NotImplementedError  # TODO adapat from below

    def update_bounds(self, line):
        # geometric leaf bounds are 3D
        range = line.range
        direction = line.direction

        if direction._fractional:
            raise NotImplementedError("TODO")

        if direction.num_free_directions == 1 and direction._vectors[0][2] == 1:  # TODO: index 2 here needs to be changed to LEED convention
            # z-only movement
            start = np.array([range.start, 0., 0.])
            stop = np.array([range.stop, 0., 0.])
            user_set = [True, False, False]
        else:
            raise NotImplementedError("TODO")
        self._bounds.update_range((start, stop), user_set=user_set)

    def update_offsets(self, line):
        # geometric leaf bounds are 3D
        direction = line.direction

        if direction._fractional:
            raise NotImplementedError("TODO")

        if (
            direction.num_free_directions == 1
            and direction._vectors[0][2]
            == 1  # TODO: index 2 here needs to be changed to LEED convention
        ):
            # z-only movement
            offset = np.array([line.value, 0., 0.])
            user_set = np.array([True, False, False])
        else:
            raise NotImplementedError("TODO")
        self._bounds.update_range(range=None, offset=offset, user_set=user_set)


class GeoHLConstraintNode(HLConstraintNode):
    """Base constraint node for geometric parameters."""

    def __init__(self, dof, children, transformers, layer, name="unnamed"):
        self.dof = dof

        if transformers is None:
            raise ValueError(
                "Transformers must be provided for "
                "geometric constraint nodes."
            )
        super().__init__(
            dof=dof, name=name, children=children,
            transformers=transformers, layer=layer
        )


class GeoSymmetryHLConstraint(GeoHLConstraintNode):
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
    linked."""

    def __init__(self, children):
        # transformers may not be provided, as they must be determined from the
        # symmetry operations
        if not children:
            raise ValueError("Symmetry constraints must have children")

        # make sure that all children are leaf nodes
        if not all(isinstance(child, GeoHLLeafNode) for child in children):
            raise ValueError(
                "Symmetry constraints can only be applied to "
                "geometric leaf nodes."
            )

        # check that all children are in the same linklist
        linklist = children[0].base_scatterer.atom.linklist
        if not all([child.base_scatterer.atom in linklist for child in children]):
            raise ValueError(
                "Symmetry linked atoms must be in the same " "linklist"
            )

        # irrespective of the symmetry the transformer bias is zero
        bias = np.zeros(3)
        transformers = []

        # how to proceed is determined by the freedir attribute
        # NB: this is a weird attribute and will be changed in the future
        # Currently freedir can be either an int (specifically 0 or 1, implying
        # z-only or completely free movement) or a 1D array of shape (2,)
        # (implying 1D in-plane movement in addition to z)
        if children[0].base_scatterer.atom.freedir == 0:
            # check that all children have the same freedir
            if not all(
                child.base_scatterer.atom.freedir == 0 for child in children
            ):
                raise ValueError(
                    "All symmetry linked atoms must have the same "
                    "freedir attribute."
                )
            # z-only movement
            dof = 1
            name = "Symmetry (z-only)"
            for child in children:
                # set the symmetry linking matrix and direct transfer of z
                weights = np.array([1.0, 0., 0.]).reshape((3,1))
                transformers.append(LinearTransformer(weights, bias, (3,)))

        elif children[0].base_scatterer.atom.freedir == 1:
            # check that all children have the same freedir
            if not all(
                child.base_scatterer.atom.freedir == 1 for child in children
            ):
                raise ValueError(
                    "All symmetry linked atoms must have the same "
                    "freedir attribute."
                )
            # free in-plane movement in addition to z
            dof = 3
            name = "Symmetry (free)"

            for child in children:
                # set the symmetry linking matrix and direct transfer of z
                weights = np.identity(3)
                weights[1:3, 1:3] = child.symrefm
                bias = np.zeros(3)
                transformers.append(LinearTransformer(weights, bias, (3,)))

        elif children[0].base_scatterer.atom.freedir.shape == (2,):
            # check that all children have the same freedir
            if not all(
                child.base_scatterer.atom.freedir.shape == (2,)
                for child in children
            ):
                raise ValueError(
                    "All symmetry linked atoms must have the same "
                    "freedir attribute."
                )
            # 1D in-plane movement in addition to z
            dof = 2
            name = "Symmetry (1D in-plane)"
            # TODO: sort this out (discuss using fractional coordinates)
            raise NotImplementedError
        else:
            raise ValueError(
                "freedir attribute must be 0, 1, or have shape (2,)"
            )

        # TODO: sort out how the propagators transform
        # self.symmetry_operations = {
        #     child: geo_sym_linking(child)
        #     for child in children
        # }

        super().__init__(
            dof=dof,
            children=children,
            transformers=transformers,
            name=name,
            layer=HLTreeLayers.Symmetry,
        )


class GeoLinkedHLConstraint(GeoHLConstraintNode):
    """Class for explicit links of geometric parameters."""
    # TODO: if we implement linking of nodes with different dof (directional),
    # this needs to be adapted
    def __init__(self, children, name):
        # check that all children have the same dof
        if len(set(child.dof for child in children)) != 1:
            raise ValueError("Children must have the same dof.")
        dof = children[0].dof

        # transformers can be identity
        transformers = [
            LinearTransformer(np.eye(dof), np.zeros(dof), (dof,))
            for _ in children
        ]
        super().__init__(
            dof=dof, children=children, transformers=transformers,
            name=f"CONSTRAIN '{name}'",
            layer=HLTreeLayers.User_Constraints,
        )

class GeoHLSubtree(ParameterHLSubtree):
    def __init__(self, base_scatterers):
        super().__init__(base_scatterers)

    @property
    def name(self):
        return "Geometric Parameters"

    @property
    def subtree_root_name(self):
        return "geo root"

    def build_subtree(self):

        # create leaf nodes
        geo_leaf_nodes = [
            GeoHLLeafNode(base_scatterer)
            for base_scatterer in self.base_scatterers
        ]
        self.nodes.extend(geo_leaf_nodes)

        # apply symmetry constraints
        for siteel in self.site_elements:
            site_el_params = [
                node for node in self.leaves if node.site_element == siteel
            ]

        for link in self.base_scatterers.symmetry_links:
            # put all linked atoms in the same symmetry group
            nodes_to_link = [
                node for node in self.leaves
                if node.base_scatterer in link
            ]
            if nodes_to_link:
                self.nodes.append(GeoSymmetryHLConstraint(children=nodes_to_link))

        unlinked_site_el_nodes = [node for node in self.leaves if node.is_root]
        for node in unlinked_site_el_nodes:
            self.nodes.append(GeoSymmetryHLConstraint(children=[node]))

        # # add offset nodes
        # self._add_offset_nodes("geo offset (unused)")

    def apply_explicit_constraint(self, constraint_line):
        #self._check_constraint_line_type(constraint_line, "geo")
        *_, selected_roots = self._select_constraint(constraint_line)

        if constraint_line.direction is None:
            # complete linking; requires all root nodes to have the same dof
            if not all(node.dof == selected_roots[0].dof for node in selected_roots):
                raise ValueError(
                    "All root nodes must have the same number of free parameters."
                )
            # create a constraint node for the selected roots
            self.nodes.append(GeoLinkedHLConstraint(children=selected_roots,
                                                    name=constraint_line.line))
        else:
            raise NotImplementedError(
                "Directional geo constraints are not yet supported.")


class GeoBaseParam(BaseParam):
    def __init__(self, base_scatterer):
        self.n_free_params = 3  # x, y, z
        self.symrefm = base_scatterer.atom.symrefm
        self.layer = base_scatterer.atom.layer.num
        super().__init__(base_scatterer)

    @property
    def symmetry_linking(self):
        if self.parent is None:
            return self.symrefm
        symmetry_op = np.array([[1.0, 0.0], [0.0, 1.0]])
        up = self
        while up.parent is not None:
            symmetry_op = np.dot(
                symmetry_op, up.parent.symmetry_operations[up]
            )  # TODO: left or right multiply?
            up = up.parent
        return symmetry_op


# Isotropic geometric bound; could be extended to anisotropic
class GeoParamBound(Bound):
    def __init__(self, min, max):
        super().__init__(min, max)

    @property
    def fixed(self):
        return abs(self.min - self.max) < 1e-6


class GeoParams(Params):
    def __init__(self, delta_slab):
        # Create base parameters for each non-bulk atom (x, y, z)
        self.params = [
            GeoBaseParam(base_scatterer)
            for base_scatterer in delta_slab.base_scatterers
        ]
        # apply symmetry constraints
        for siteel in delta_slab.site_elements:
            site_el_params = [
                param
                for param in self.base_params
                if param.site_element == siteel
            ]
            for linklist in delta_slab.slab.linklists:
                ref_atom = linklist[0]
                # put all linked atoms in the same symmetry group
                params_to_link = [
                    param
                    for param in site_el_params
                    if param.base_scatterer.atom in linklist
                ]
                if params_to_link:
                    self.params.append(
                        GeoSymmetryConstraint(children=params_to_link)
                    )
            unlinked_site_el_params = [
                param
                for param in site_el_params
                if param in self.terminal_params
            ]
            for param in unlinked_site_el_params:
                # NB: if an atom is not linked to any others, it will be placed
                #     into a symmetry constraint with a single child. No
                #     GeoBaseParams should remain.
                self.params.append(GeoSymmetryConstraint(children=[param]))

        super().__init__()

    @property
    def layers(self):
        return tuple(sorted(set(param.layer for param in self.base_params)))

    @property
    def symmetry_operations(self):
        return jnp.array([param.symmetry_linking for param in self.base_params])

    def constrain_layer(self, layer):
        layer_params = [
            param for param in self.terminal_params if param.layer == layer
        ]
        if not layer_params:
            raise ValueError(f"No free params for layer {layer}")
        new_constraint = GeoLayerConstraint(children=layer_params)
        self.params.append(new_constraint)

    def fix_layer(self, layer, z_offset=None):
        layer_params = [
            param for param in self.terminal_params if param.layer == layer
        ]
        if not layer_params:
            raise ValueError(f"No free params for layer {layer}")
        z_constraint = GeoLayerConstraint(children=layer_params)
        fix_constraint = GeoFixConstraint(children=[z_constraint])
        if z_offset is not None:
            fix_bound = GeoParamBound(z_offset, z_offset)
        fix_constraint.set_bound(fix_bound)
        self.params.append(z_constraint)
        self.params.append(fix_constraint)

    def set_geo_bounds(self, geo_bounds):
        self.geo_bounds = tuple(geo_bounds)
        for bound, param in zip(geo_bounds, self.terminal_params):
            param.set_bound(bound)

    @property
    def dynamic_propagators(self):
        return [param for param in self.free_params]

    @property
    def static_propagators(self):
        return [
            param
            for param in self.terminal_params
            if param not in self.free_params
        ]

    @property
    def static_propagator_inputs(self):
        return tuple(
            param.get_static_displacement() for param in self.static_propagators
        )

    @property
    def n_dynamic_propagators(self):
        return len(self.dynamic_propagators)

    @property
    def n_static_propagators(self):
        return len(self.static_propagators)

    @property
    def propagator_map(self):
        # map proagators to atom-site-elements
        return [
            (
                ("static", self.static_propagators.index(terminal))
                if terminal in self.static_propagators
                else ("dynamic", self.dynamic_propagators.index(terminal))
            )
            for base, terminal in self.base_to_terminal_map.items()
        ]

    def get_geo_transformer(self):
        """Return a JAX function that transforms the free parameters
        (self.n_free_params values normalized to [0, 1]) to the displacements
        for the dynamic propagators ((3, self.n_dynamic_propagators) values).
        """
        if not all(param.bound is not None for param in self.terminal_params):
            raise ValueError("Not all geometric parameters have bounds.")

        # offset is 0 (# TODO: could implement that)
        offsets = np.zeros((3 * self.n_dynamic_propagators))

        # weights are given by the bounds and the constraint method
        mins = [
            param.free_param_map * (param.bound.min)
            for param in self.dynamic_propagators
        ]
        offsets += jax.scipy.linalg.block_diag(*mins) @ jnp.ones(
            self.n_free_params
        )
        weights = [
            param.free_param_map * (param.bound.max - param.bound.min)
            for param in self.dynamic_propagators
        ]
        weights = jax.scipy.linalg.block_diag(*weights)
        assert weights.shape == (
            3 * self.n_dynamic_propagators,
            self.n_free_params,
        )

        transformer = LinearTransformer(weights, offsets, out_reshape=(-1, 3))
        return transformer


class GeoConstraint(ConstrainedDeltaParam):
    def __init__(self, children):
        # all children must be in the same layer
        if not all(child.layer == children[0].layer for child in children):
            raise ValueError("All children must be in the same layer")
        self.layer = children[0].layer
        self.bound = None
        super().__init__(children)


class GeoSymmetryConstraint(GeoConstraint):
    # constrains multiple atoms (x, y, z) to move together (x, y, z)
    # in a symmetry linked way
    # For n linked atom-elements, the number of free parameters is reduced
    # from 3*n to 3.
    def __init__(self, children):
        self.n_free_params = 3
        self.symmetry_operations = {child: child.symrefm for child in children}
        super().__init__(children)

    @property
    def free_param_map(self):
        # three free parameters are mapped to the x, y, z directions
        return np.array(([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))


class GeoLayerConstraint(GeoConstraint):
    # constrain multiple atoms in the same layer (x, y, z) to move together
    # (z only, since in-plane movement would break symmetries)
    def __init__(self, children):
        self.n_free_params = 1
        self.symmetry_operations = {
            child: np.array([[1.0, 0.0], [0.0, 1.0]]) for child in children
        }
        super().__init__(children)

    @property
    def free_param_map(self):
        # one free parameter is mapped to the z direction
        return np.array([0.0, 0.0, 1.0]).T


class GeoFixConstraint(GeoConstraint):
    # constrains childs to fixed values
    def __init__(self, children):
        self.n_free_params = 0
        # only one child is allowed
        if isinstance(children, GeoConstraint):
            self.only_child = children
        elif len(children) == 1:
            self.only_child = children[0]
        else:
            raise ValueError("FixConstraint can only take one child.")
        self.symmetry_operations = {
            self.only_child: np.array([[1.0, 0.0], [0.0, 1.0]])
        }
        super().__init__([self.only_child])

    def get_static_displacement(self):
        if self.bound is None:
            raise ValueError("Bound not set for fix constraint.")
        if not self.bound.fixed:
            raise ValueError("Bound must be fixed for GeoFixConstraint.")
        return jnp.dot(self.only_child.free_param_map, self.bound.min)


def geo_sym_linking(atom):
    linking = np.zeros(shape=(3, 3))
    linking[1:3, 1:3] = atom.symrefm  # TODO: round off the 1e-16 contributions
    linking[0, 0] = 1.0  # all symmetry linked atoms move together is z directon
    return linking
