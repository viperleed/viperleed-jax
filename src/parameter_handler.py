# Note: this would fit into a pytree very nicely

import jax
import numpy as np
import jax.numpy as jnp

# TODO: parameter normalization

class TensorParameterTransformer:

    def __init__(self, slab, energy_step):

        self.energy_step = energy_step
        # construct symmetry linking from slab
        self._get_sym_linking(slab)

        # set identity constraints for now
        self.geo_constraint_matrix = jnp.identity(self.geo_sym_linking_matrix.shape[1])
        self.vib_constraints = jnp.identity(self.vib_sym_linking_matrix.shape[1])

    def _get_sym_linking(self, slab):
        if not slab.foundplanegroup:
            raise ValueError("Symmetry detection needs to be run on slab.")
        non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
        linked_atoms_bulk = [[at.is_bulk for at in llist] for llist in slab.linklists]

        # bulk atoms should only be linked to bulk atoms!
        all_linked_atoms = [all(are_bulk) for are_bulk in linked_atoms_bulk]
        assert all_linked_atoms == [any(are_bulk) for are_bulk in linked_atoms_bulk]

        # select only the non-bulk atom linklists
        non_bulk_atom_linklists = [linklist for linklist, is_bulk in zip(slab.linklists, all_linked_atoms) if not is_bulk]

        geo_sym_linking_matrix = np.zeros(shape=(len(non_bulk_atoms)*3, len(non_bulk_atom_linklists)*3))
        vib_sym_linking_matrix = np.zeros(shape=(len(non_bulk_atoms), len(non_bulk_atom_linklists)))

        def geo_sym_linking(atom):
            linking = np.zeros(shape=(3,3))
            linking[1:3, 1:3] = atom.symrefm  #TODO: round off the 1e-16 contributions
            linking[0,0] = 1.0 # all symmetry linked atoms move together is z directon
            return linking

        for list_id, linklist in enumerate(non_bulk_atom_linklists):
            base_atom = linklist[0]
            for atom in linklist:
                atom_id = non_bulk_atoms.index(atom)

                geo_sym_linking_matrix[(atom_id)*3:(atom_id+1)*3, (list_id)*3:(list_id+1)*3] = geo_sym_linking(atom)
                vib_sym_linking_matrix[atom_id, list_id] = 1.0

        self.geo_sym_linking_matrix = jnp.asarray(geo_sym_linking_matrix)
        self.vib_sym_linking_matrix = jnp.asarray(vib_sym_linking_matrix)

    @property
    def n_vib_amps(self):
        return self.vib_sym_linking_matrix.shape[0]

    @property
    def n_displacements(self):
        return self.geo_sym_linking_matrix.shape[0]

    @property
    def n_parameters(self):
        # +1 for v0r
        return 1 + self.n_vib_amps + self.n_displacements

    @property
    def n_irreducible_vib_amps(self):
        return self.vib_sym_linking_matrix.shape[1]

    @property
    def n_irreducible_displacements(self):
        return self.geo_sym_linking_matrix.shape[1]

    @property
    def n_irreducible_parameters(self):
        # +1 for v0r
        return 1 + self.n_irreducible_vib_amps + self.n_irreducible_displacements

    @property
    def n_constrained_vib_amps(self):
        return self.vib_constraints.shape[1]

    @property
    def n_constrained_displacements(self):
        return self.geo_constraint_matrix.shape[1]

    @property
    def n_constrained_parameters(self):
        # +1 for v0r
        return 1 + self.n_constrained_vib_amps + self.n_constrained_displacements

    # TODO: make more sophisticated constraint checking & generation
    def apply_geo_constraints(self, geo_constraints):
        if geo_constraints.shape[0] != self.n_irreducible_displacements:
            raise ValueError("Geo constraints need to map to the number of irreducible displacements")
        self.geo_constraint_matrix = geo_constraints

    def apply_vib_constraints(self, vib_constraints):
        if vib_constraints.shape[0] != self.n_irreducible_vib_amps:
            raise ValueError("Vib constraints need to map to the number of irreducible vib amps")
        self.vib_constraints = vib_constraints

    # TODO: discuss when these should be applied
    # at the end, or to the irreducible parameters? – I think irreducible

    def set_v0r_bounds(self, v0r_min, v0r_max):
        self.v0r_bounds = (v0r_min, v0r_max)

    def set_vib_amp_bounds(self, vib_amp_min, vib_amp_max):
        self.vib_amp_bounds = (vib_amp_min, vib_amp_max)

    def set_displacement_bounds(self, displacement_min, displacement_max):
        self.displacement_bounds = (displacement_min, displacement_max)

    def unflatten_parameters(self, flattened_params):
        float_v0r = flattened_params[0]
        constrained_vib_amps = flattened_params[1:1+self.n_constrained_vib_amps]
        constrained_displacements = flattened_params[1+self.n_constrained_vib_amps:]

        v0r = self._expand_v0r(float_v0r)
        vib_amps = self._expand_vib_amps(constrained_vib_amps)
        displacements = self._expand_displacements(constrained_displacements)
        return v0r, vib_amps, displacements

    def _expand_v0r(self, v0r):
        """V0r is given to the optimizer as a float, but needs to be rounded to an integer"""
        v0r_min, v0r_max = self.v0r_bounds
        v0r = v0r_min + (v0r)*(v0r_max-v0r_min) # normalize 0..1
        return jnp.array(jnp.rint(v0r/self.energy_step), dtype=jnp.int32)

    def _expand_vib_amps(self, short_vib_amps):
        # undo normalization (0, 1)
        vib_amps_min, vib_amps_max = self.vib_amp_bounds
        vib_amps = vib_amps_min + short_vib_amps*(vib_amps_max-vib_amps_min)
        # undo constraints
        vib_amps =  vib_amps @ self.vib_constraints.T 
        # undo symmetry reduction
        return vib_amps @ self.vib_sym_linking_matrix.T

    def _expand_displacements(self, short_displacements):
        # normalize to range (0, 1)
        min_displacement, max_displacement = self.displacement_bounds
        displacements = min_displacement + short_displacements*(max_displacement-min_displacement)
        # undo constraints
        displacements = displacements @ self.geo_constraint_matrix.T
        # undo symmetry reduction
        displacements = displacements @ self.geo_sym_linking_matrix.T
        # undo flattening
        return displacements.reshape(-1, 3)

    @property
    def info(self):
        return (
            "Constrained parameters:\n"
            f"{self.n_constrained_parameters}\t"
            f"({self.n_constrained_displacements} geo, "
            f"{self.n_constrained_vib_amps} vib, 1 V0r)\n"
            "Symmetry irreducible parameters:\n"
            f"{self.n_irreducible_parameters}\t"
            f"({self.n_irreducible_displacements} geo, "
            f"{self.n_irreducible_vib_amps} vib, 1 V0r)\n"
            "Reducible parameters:\n"
            f"{self.n_parameters}\t"
            f"({self.n_displacements} geo, "
            f"{self.n_vib_amps} vib, 1 V0r)\n"
        )

    def tree_flatten(self):
        pass