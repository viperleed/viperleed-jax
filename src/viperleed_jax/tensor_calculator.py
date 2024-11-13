"""Module tensor_calculator."""

__authors__ = ("Alexander M. Imre (@amimre)",
               "Paul Haidegger (@Paulhai7)")
__created__ = "2024-05-03"

import time

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

from viperleed.calc import LOGGER as logger
from viperleed.calc.files.iorfactor import beamlist_to_array

from viperleed_jax import rfactor
from viperleed_jax.constants import BOHR, HARTREE
from viperleed_jax.interpolation import *
from viperleed_jax.lib_intensity import intensity_prefactor, sum_intensity
from viperleed_jax import lib_math
from viperleed_jax import atomic_units

from viperleed_jax.t_matrix import vib_dependent_tmatrix
from viperleed_jax.propagator import calc_propagator, symmetry_operations
from viperleed_jax.dense_quantum_numbers import DENSE_QUANTUM_NUMBERS
from viperleed_jax.dense_quantum_numbers import  map_l_array_to_compressed_quantum_index
from viperleed_jax.batching import Batching

from viperleed_jax.interpolation import interpolate_ragged_array

_R_FACTOR_SYNONYMS = {
    rfactor.pendry_R: ('pendry', 'r_p', 'rp', 'pendry r-factor'),
    rfactor.R_1: ('r1', 'r_1', 'r1 factor'),
    rfactor.R_2: ('r2', 'r_2', 'r2 factor'),
    rfactor.R_ms: ('ms', 'msr', 'rms', 'r_ms', 'r_ms factor'),
    rfactor.R_zj: ('zj', 'zj factor', 'zannazi', 'zannazi jona', 'zannazi-jona'),
}


@register_pytree_node_class
class TensorLEEDCalculator:
    """Main class for calculating tensor LEED intensities and R-factors.

    Parameters:
    -----------
    ref_data : ReferenceData
        The reference data for LEED calculations.
    phaseshifts : ndarray
        The phaseshifts for LEED calculations.
    slab : Slab
        The slab object representing the crystal structure.
    rparams : Rparams
        The parameters for LEED calculations.
    interpolation_step : float, optional
        The step size for interpolation, by default 0.5.
    interpolation_deg : int, optional
        The degree of interpolation, by default 3.
    bc_type : str, optional
        The boundary condition type for interpolation, by default 'not-a-knot'.
    batch_lmax : bool or int, optional
        Whether to use batched calculation. By default True.
    """

    def __init__(self, ref_data, phaseshifts, slab, rparams,
                 interpolation_step=0.5,
                 interpolation_deg=3,
                 bc_type='not-a-knot',
                 batch=True,
                 recalculate_ref_t_matrices=False):
        self.ref_data = ref_data
        self.phaseshifts = phaseshifts
        self.recalculate_ref_t_matrices = recalculate_ref_t_matrices

        self.interpolation_deg = interpolation_deg
        self.bc_type=bc_type

        # beam indices
        beam_indices = [beam.hk for beam in rparams.ivbeams]
        self.beam_indices = jnp.array([beam.hk for beam in rparams.ivbeams])
        self.n_beams = self.beam_indices.shape[0]

        self.comp_intensity = None
        self.comp_energies = None
        self.interpolation_step = interpolation_step
        self._parameter_space = None

        self.target_grid = jnp.arange(rparams.THEO_ENERGIES.start,
                                      rparams.THEO_ENERGIES.stop,
                                      self.interpolation_step)

        # unit cell in Bohr radii
        self.unit_cell = slab.ab_cell.copy() / BOHR

        # theta and phi (in radians)
        self.theta = jnp.deg2rad(rparams.THETA)
        self.phi = jnp.deg2rad(rparams.PHI)

        # energies, inner potential, kappa
        self.energies = jnp.asarray(self.ref_data.energies)
        self.v0i = ref_data.v0i
        self.kappa = atomic_units.kappa(self.energies, self.v0i)

        non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
        # TODO check this
        self.is_surface_atom = jnp.array([at.layer.num == 0 for at in non_bulk_atoms])

        self.ref_vibrational_amps = jnp.array(
            [at.site.vibamp[at.el] for at in non_bulk_atoms])
        self.origin_grid = ref_data.incident_energy_ev

        self.delta_amp_prefactors = self._calc_delta_amp_prefactors()

        self.exp_spline = None

        # default R-factor is Pendry
        self.rfactor_func = rfactor.pendry_R

        if self.interpolation_deg != 3:
            raise NotImplementedError

        # work out the energy batching
        if batch is False:
            # do not perform batching other than the requested lmax-batching
            self.batching = Batching(self.energies,
                                     ref_data.lmax,
                                     None)
        elif batch is True:
            self.batching = Batching(self.energies,
                                     ref_data.lmax,
                                     8)
        elif isinstance(batch, int):
            self.batching = Batching(self.energies,
                                     ref_data.lmax,
                                     batch)
        else:
            raise ValueError("batch_lmax must be bool or int.")
        logger.info(
            f'Batching initialized with {len(self.batching.batches)} batches '
            f'and a maximum batch size of {self.batching.max_batch_size}.'
        )

        self.tensor_amps_in, self.tensor_amps_out = self._batch_tensor_amps()

        # get experimental intensities and hk
        exp_energies, _, _, exp_intensities = beamlist_to_array(rparams.expbeams)
        exp_hk = [b.hk for b in rparams.expbeams]

        # determine the mapping
        exp_beam_mapping = []
        for b in beam_indices:
            if b in exp_hk:
                exp_beam_mapping.append(exp_hk.index(b))
            else:
                exp_beam_mapping.append(False)

        mask_out_expbeam = [b is False for b in exp_beam_mapping]
        exp_beam_mapping = np.array(exp_beam_mapping, dtype=np.int32)

        # apply the mapping
        mapped_exp_intensities = exp_intensities[:, exp_beam_mapping]

        # mask out the beams that are not in the expbeams
        mapped_exp_intensities = np.where(mask_out_expbeam, np.nan,
                                          mapped_exp_intensities)
        self.set_experiment_intensity(mapped_exp_intensities, exp_energies)

    @property
    def unit_cell_area(self):
        return jnp.linalg.norm(jnp.cross(self.unit_cell[:,0], self.unit_cell[:,1]))

    @property
    def reciprocal_unit_cell(self):
        return 2*jnp.pi*jnp.linalg.inv(self.unit_cell)

    @property
    def n_atoms(self):
        return len(self.ref_vibrational_amps)

    @property
    def max_l_max(self):
        return max(self.ref_data.needed_lmax)

    @property
    def parameter_space(self):
        if self._parameter_space is None:
            raise ValueError("Parameter space not set.")
        return self._parameter_space

    @property
    def n_free_parameters(self):
        return self.parameter_space.n_free_params

    def set_rfactor(self, rfactor_name):
        _rfactor_name = rfactor_name.lower().strip()
        for func, synonyms in _R_FACTOR_SYNONYMS.items():
            if _rfactor_name in synonyms:
                self.rfactor_func = func
                # TODO: log rfactor change
                return
        raise ValueError(f"Unknown R-factor name: {rfactor_name}")

    def set_experiment_intensity(self, comp_intensity, comp_energies):
        logger.debug(
            'Setting experimental intensities and initializing interpolators.')

        self.comp_intensity = comp_intensity
        self.comp_energies = comp_energies
        self.exp_spline = interpolate_ragged_array(
            self.comp_energies,
            self.comp_intensity,
            bc_type=self.bc_type,
        )

    def _batch_tensor_amps(self):
        tensor_amps_in = []
        tensor_amps_out = []
        for batch in self.batching.batches:
            tensor_amps_in.append(
                self.ref_data.tensor_amps_in[batch.l_max][batch.energy_indices])
            tensor_amps_out.append(
                self.ref_data.tensor_amps_out[batch.l_max][batch.energy_indices])
        return tensor_amps_in, tensor_amps_out

    def set_parameter_space(self, parameter_space):
        if self._parameter_space is not None:
            logger.debug("Overwriting parameter space.")
        # take delta_slab and set the parameter space
        self._parameter_space = parameter_space.freeze()
        logger.info("Parameter space set.\n"
                    f"{parameter_space.info}")
        logger.info(
            "This parameter space requires dynamic calculation of "
            f"{self._parameter_space.n_dynamic_t_matrices} t-matrice(s) and "
            f"{self._parameter_space.n_dynamic_propagators} propagator(s)."
        )

        if self.recalculate_ref_t_matrices:
            # calculate reference t-matrices for full LMAX
            n_ref_vib_amps = len(parameter_space.vib_subtree.leaves)
            logger.debug(
                f"Calculating {n_ref_vib_amps} reference t-matrices for "
                f"LMAX={self.max_l_max}.")
            ref_vib_amps = [leaf.ref_vib_amp
                            for leaf in parameter_space.vib_subtree.leaves]
            site_elements = [leaf.site_element
                             for leaf in parameter_space.vib_subtree.leaves]
            self.ref_t_matrices = self._calculate_reference_t_matrices(
                ref_vib_amps, site_elements)
        else:
            self.ref_t_matrices = self.ref_data.ref_t_matrix[self.max_l_max]

        # pre-calculate the static t-matrices
        logger.debug(
            f"Pre-calculating {self._parameter_space.n_static_t_matrices} "
            "static t-matrice(s).")
        self._calculate_static_t_matrices()

        # rotation angles
        propagator_symmetry_operations, propagator_transpose = self._propagator_rotation_factors()
        self.propagator_symmetry_operations = jnp.asarray(propagator_symmetry_operations)
        # NB: Using an integer array here because there seems so be some kind of
        # bug where jax.jit would flip on of the boolean values for some
        # cases.
        self.propagator_transpose_int = propagator_transpose.astype(jnp.int32)

        # pre-calculate the static propagators
        logger.debug(
            f"Pre-calculating {self._parameter_space.n_static_propagators} "
            "static propagator(s).")
        self._calculate_static_propagators()

    def _calculate_static_t_matrices(self):
        # this is only done once – perform for maximum lmax and crop later
        t_matrix_vmap_en = jax.vmap(vib_dependent_tmatrix,
                                   in_axes=(None, 0, 0, None),
                                   out_axes=0)
        static_t_matrices = jnp.array([
            t_matrix_vmap_en(
                self.max_l_max,
                self.phaseshifts[site_el][:, :self.max_l_max+1],
                self.energies,
                vib_amp,
            )
            for site_el, vib_amp
            in self._parameter_space.static_t_matrix_inputs.items()])
        self._static_t_matrices = jnp.einsum('ael->eal', static_t_matrices)

    def _calculate_static_propagators(self):
        # this is only done once – perform for maximum lmax and crop later
        propagator_vmap_en = jax.vmap(calc_propagator,
                                      in_axes=(None, None, 0))
        displacements_ang = jnp.asarray(self._parameter_space.static_propagator_inputs)
        displacements_au = atomic_units.to_internal_displacement_vector(displacements_ang)
        self._static_propagators = jnp.array([
            propagator_vmap_en(
                self.max_l_max,
                displacement,
                self.kappa
            )
            for displacement in displacements_au])

    def _calculate_dynamic_t_matrices(self, vib_amps, energy_indices):
        t_matrix_vmap_en = jax.vmap(vib_dependent_tmatrix,
                                   in_axes=(None, 0, 0, None),
                                   out_axes=0)
        dynamic_t_matrices = [
            t_matrix_vmap_en(
                self.max_l_max,
                self.phaseshifts[site_el][energy_indices, : self.max_l_max + 1],
                self.energies[energy_indices],
                vib_amp.reshape(),  # cast shape from (1,) to (); required to play nicely with grad
            )
            for vib_amp, site_el in zip(
                vib_amps, self.parameter_space.dynamic_t_matrix_site_elements
            )
        ]
        dynamic_t_matrices = jnp.asarray(dynamic_t_matrices)
        return jnp.einsum('ael->eal', dynamic_t_matrices)

    def _calculate_reference_t_matrices(self, ref_vib_amps, site_elements):
        t_matrix_vmap_en = jax.vmap(vib_dependent_tmatrix,
                                   in_axes=(None, 0, 0, None),
                                   out_axes=0)
        ref_t_matrices =  jnp.array([
            t_matrix_vmap_en(
                self.max_l_max,
                self.phaseshifts[site_el][:, :self.max_l_max+1],
                self.energies,
                vib_amp
            )
            for vib_amp, site_el
            in zip(ref_vib_amps, site_elements)])
        return jnp.einsum('ael->eal', ref_t_matrices)

    def _calculate_t_matrices(self, vib_amps, energy_indices):
        # return t-matrices indexed as (energies, atom-site-elements, lm)

        dynamic_t_matrices = self._calculate_dynamic_t_matrices(vib_amps, energy_indices)
        # map t-matrices to atom-site-element basis
        mapped_dynamic_t_matrices = dynamic_t_matrices[:, self.parameter_space.t_matrix_id] #TODO: clamp?

        # if there are 0 static t-matrices, indexing would raise Error
        if len(self._static_t_matrices) == 0:
            static_t_matrices = jnp.array([jnp.zeros_like(dynamic_t_matrices[0])])
        else:
            static_t_matrices = self._static_t_matrices[energy_indices, :, :]
        mapped_static_t_matrices = static_t_matrices[:, self.parameter_space.t_matrix_id, :]

        t_matrices = jnp.where(
            self.parameter_space.is_dynamic_t_matrix[jnp.newaxis, :, jnp.newaxis],
            mapped_dynamic_t_matrices,
            mapped_static_t_matrices)
        return t_matrices

    @partial(jax.profiler.annotate_function, name="tc.calculate_dynamic_propagator")
    def _calculate_dynamic_propagators(self, displacements, energy_indices):
        propagator_vmap_en = jax.vmap(calc_propagator,
                                      in_axes=(None, None, 0))

        def body_fn(carry, displacement):
            # Compute the result for the current displacement
            result = propagator_vmap_en(
                self.max_l_max,
                displacement,
                self.kappa[energy_indices],
            )
            # No carry state needed, just passing through
            return carry, result

        # Initial carry state can be None if not needed
        _, results = jax.lax.scan(body_fn, None, displacements)
        return results

    def _calculate_propagators(self, displacements, energy_indices):
        # return propagators indexed as (base_scatterers, energies, lm, l'm')
        dynamic_propagators = self._calculate_dynamic_propagators(displacements, energy_indices)

        # if there are 0 static propagators, indexing would raise Error
        if len(self._static_propagators) == 0:
            static_propagators = jnp.array([jnp.zeros_like(dynamic_propagators[0])])
        else:
            static_propagators = self._static_propagators[:, energy_indices, :, :]

        mapped_dynamic_propagators = dynamic_propagators[self.parameter_space.propagator_id]
        mapped_static_propagators = static_propagators[self.parameter_space.propagator_id]

        propagators = jnp.where(
            self.parameter_space.is_dynamic_propagator[:, jnp.newaxis, jnp.newaxis, jnp.newaxis],
            mapped_dynamic_propagators,
            mapped_static_propagators)
        # selective transpositions
        propagators = (
            (1-self.propagator_transpose_int)[:, np.newaxis, np.newaxis, np.newaxis ]*propagators +
            self.propagator_transpose_int[:, np.newaxis, np.newaxis, np.newaxis ]*jnp.transpose(propagators, (0, 1, 3, 2))
        )
        # apply rotations and rearrange to make energy the first axis
        propagators = jnp.einsum('aelm,alm->ealm',
                                propagators,
                                self.propagator_symmetry_operations,
                                optimize='optimal')

        return propagators

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_propagators_new(self, displacements):
        # return propagators indexed as (energies, base_scatterers, lm, l'm')

        dynamic_propagators = self._calculate_dynamic_propagators(displacements)

        # if there are 0 static propagators, indexing would raise Error
        if len(self._static_propagators) == 0:
            static_propagators = jnp.array([jnp.zeros_like(dynamic_propagators[0])])
        else:
            static_propagators = self._static_propagators

        mapping_size = max(self.parameter_space.n_static_propagators,
                           self.parameter_space.n_dynamic_propagators)
        mapping = jnp.zeros(shape=(self.parameter_space.n_base_scatterers,
                                   mapping_size))
        for base_id, prop_id in enumerate(self.parameter_space.propagator_id):
            mapping = mapping.at[base_id, prop_id].set(1.0)

        mapping_static = mapping[:, :self.parameter_space.n_static_propagators]
        mapping_dynamic = mapping[:, :self.parameter_space.n_dynamic_propagators]

        static_propagators = jnp.einsum(
            'a,as,selm->ealm',
            (1. - self.parameter_space.is_dynamic_propagator),   # d
            mapping_static,                                      # ad
            static_propagators,                                  # delm
            optimize='optimal',
        )

        dynamic_propagators = jnp.einsum(
            'a,as,selm->ealm',
            self.parameter_space.is_dynamic_propagator,          # s
            mapping_dynamic,                                     # as
            dynamic_propagators,                                 # selm
            optimize='optimal',
        )

        # TODO: transpositoin

        return jnp.einsum(
            'aelm,aelm,alm->ealm',
            static_propagators,                                  # aelm
            dynamic_propagators,                                 # aelm
            self.propagator_symmetry_operations,                 # alm
            optimize='optimal',
        )

    # TODO: for testing purposes: contrib should be exactly 0 for not pertubations and if recalculate_ref_t_matrices=True
    def _calculate_static_ase_contributions(self):

        static_ase_contributions = np.zeros(
            (len(self.energies), self.n_beams, self.parameter_space.n_static_ase),
            dtype=np.complex128)

        for batch in self.batching.batches:
            l_max = batch.l_max
            batch_id = batch.batch_id
            energy_ids = jnp.asarray(batch.energy_indices)

            # get and reindex static t-matrices
            if len(self._static_t_matrices) == 0:
                raise ValueError("No static t-matrices found.")
            static_t_matrices = self._static_t_matrices[energy_ids, :, :]
            # broadcast to complete ase basis
            static_t_matrices = static_t_matrices[:, self.parameter_space.t_matrix_id, :]
            # broadcast down to static ases
            static_t_matrices = static_t_matrices[:, self.parameter_space.static_ase_id, :]

            if len(self._static_propagators) == 0:
                raise ValueError("No static propagators found.")
            static_propagators = self._static_propagators[:, energy_ids, :, :]
            # broadcast to complete ase basis
            static_propagators = static_propagators[self.parameter_space.propagator_id, ...]
            # apply rotations
            static_propagators = jnp.einsum(
                'aelm,alm->aelm',
                static_propagators,
                self.propagator_symmetry_operations,
                optimize='optimal',
            )
            # broadcast down to static axes
            static_propagators = static_propagators[self.parameter_space.static_ase_id, ...]

            # crop t-matrices and propagators
            t_matrices = static_t_matrices[:, :, :l_max+1]
            propagators = static_propagators[:, :, :(l_max+1)**2, :(l_max+1)**2]

            # reference t-matrix
            ref_t_matrices = jnp.asarray(self.ref_t_matrices)
            ref_t_matrices = ref_t_matrices[energy_ids, :, :l_max+1]
            ref_t_matrices = ref_t_matrices[:, self.parameter_space.static_ase_id, :]

            tensor_amps_in = self.tensor_amps_in[batch_id]
            tensor_amps_out = self.tensor_amps_out[batch_id]

            # get only static elements
            tensor_amps_in = tensor_amps_in[:, self.parameter_space.static_ase_id]
            tensor_amps_out = tensor_amps_out[:, self.parameter_space.static_ase_id]

            for seq_e_id, e_id in enumerate(energy_ids):
                for seq_ase_id in range(self.parameter_space.n_static_ase):

                    mapped_t_matrix = map_l_array_to_compressed_quantum_index(t_matrices[seq_e_id, seq_ase_id], l_max)
                    mapped_t_matrix_ref = map_l_array_to_compressed_quantum_index(ref_t_matrices[seq_e_id, seq_ase_id], l_max)
                    delta_t_matrix = calculate_delta_t_matrix(
                        propagators[seq_ase_id, seq_e_id],
                        mapped_t_matrix,
                        mapped_t_matrix_ref,
                    )
                    contribution = jnp.einsum(
                        'bl,lk,k->b',
                        tensor_amps_out[seq_e_id, seq_ase_id],
                        delta_t_matrix,
                        tensor_amps_in[seq_e_id, seq_ase_id],
                        optimize='optimal',
                    )
                    static_ase_contributions[e_id, :, seq_ase_id] = contribution

        return static_ase_contributions

    def _calc_delta_amp_prefactors(self):
        energies = self.energies
        v_imag = self.v0i

        # energy dependent quantities
        out_k_par2 = self.ref_data.kx_in
        out_k_par3 = self.ref_data.ky_in

        k_inside = jnp.sqrt(2*energies-2j*v_imag+1j*lib_math.EPS)

        # Propagator evaluated relative to the muffin tin zero i.e.
        # it uses energy = incident electron energy + inner potential
        out_k_par = out_k_par2**2 + out_k_par3**2
        out_k_perp_inside = jnp.sqrt(
            ((2*energies-2j*v_imag)[:, jnp.newaxis] - out_k_par)
            + 1j*lib_math.EPS
        )

        # Prefactors from Equation (41) from Rous, Pendry 1989
        prefactors = jnp.einsum('e,eb,->eb',
            1/k_inside,
            1/out_k_perp_inside,
            1/(2*(self.unit_cell_area))
        )
        return prefactors

    def _intensity_prefactors(self, onset_height_change):
        # onset height change was called CXDisp in the original code

        # from lib_intensity
        (in_k_vacuum, in_k_perp_vacuum,
        out_k_perp, out_k_perp_vacuum) = self._wave_vectors()

        a = out_k_perp_vacuum
        c = in_k_vacuum * jnp.cos(self.theta)

        # TODO: re-check if it should be a.real or abs(a)
        prefactor = abs(jnp.exp(-1j * onset_height_change/BOHR * (jnp.outer(in_k_perp_vacuum, jnp.ones(shape=(self.n_beams,))) + out_k_perp
                                                    ))) ** 2 * a.real / jnp.outer(c, jnp.ones(shape=(self.n_beams,))).real
        return prefactor

    def _wave_vectors(self):
        e_kin = self.energies
        v_real = self.ref_data.v0r
        v_imag = self.v0i
        n_energies = e_kin.shape[0]
        n_beams = self.beam_indices.shape[0]
        # incident wave vector
        in_k_vacuum = jnp.sqrt(jnp.maximum(0, 2 * (e_kin - v_real)))
        in_k_par = in_k_vacuum * jnp.sin(self.theta)  # parallel component
        in_k_par_2 = in_k_par * jnp.cos(self.phi)  # shape =( n_energy )
        in_k_par_3 = in_k_par * jnp.sin(self.phi)  # shape =( n_energy )
        in_k_perp_vacuum = 2 * e_kin - in_k_par_2 ** 2 - in_k_par_3 ** 2 - 2 * 1j * v_imag
        in_k_perp_vacuum = jnp.sqrt(in_k_perp_vacuum)

        # outgoing wave vector components
        in_k_par_components = jnp.stack((in_k_par_2, in_k_par_3))  # shape =(n_en, 2)
        in_k_par_components = jnp.outer(in_k_par_components, jnp.ones(shape=(n_beams,))).reshape(
        (n_energies, 2, n_beams))  # shape =(n_en ,2 ,n_beams)
        out_wave_vec = jnp.dot(self.beam_indices, self.reciprocal_unit_cell)  # shape =(n_beams, 2)
        out_wave_vec = jnp.outer(jnp.ones_like(e_kin), out_wave_vec.transpose()).reshape((n_energies, 2, n_beams))  # shape =(n_en , n_beams)
        out_k_par_components = in_k_par_components + out_wave_vec

        # out k vector
        out_k_perp_vacuum = (2*jnp.outer(e_kin-v_real,jnp.ones(shape=(n_beams,)))
                    - out_k_par_components[:, 0, :] ** 2
                    - out_k_par_components[:, 1, :] ** 2).astype(dtype="complex64")
        out_k_perp = jnp.sqrt(out_k_perp_vacuum + 2*jnp.outer(v_real-1j*v_imag, jnp.ones(shape=(n_beams,))))
        out_k_perp_vacuum = jnp.sqrt(out_k_perp_vacuum)

        return in_k_vacuum, in_k_perp_vacuum, out_k_perp, out_k_perp_vacuum

    def _propagator_rotation_factors(self):

        ops =[symmetry_operations(self.max_l_max, plane_sym_op) for plane_sym_op
            in self.parameter_space.propagator_plane_symmetry_operations
        ]
        symmetry_tensors = np.array([op[0] for op in ops])
        mirror_propagators = np.array([op[1] for op in ops])

        return symmetry_tensors, mirror_propagators

    def delta_amplitude(self, free_params):
        _free_params = jnp.asarray(free_params)
        # split free parameters
        (_,
         vib_params,
         geo_parms,
         occ_params) = self.parameter_space.split_free_params(jnp.asarray(_free_params))

        # displacements, converted to atomic units
        displacements_ang = self.parameter_space.reference_displacements(geo_parms)
        displacements_ang = jnp.asarray(displacements_ang)
        displacements_au = atomic_units.to_internal_displacement_vector(displacements_ang)

        # vibrational amplitudes, converted to atomic units
        vib_amps_au = self.parameter_space.reference_vib_amps(vib_params)
        # vib_amps_au = jax.vmap(atomic_units.to_internal_vib_amps,
        #                         in_axes=0)(vib_amps_ang)

        # chemical weights
        chem_weights = jnp.asarray(
            self.parameter_space.occ_weight_transformer(occ_params))

        # atom ids that will be batched over
        atom_ids = jnp.arange(self.parameter_space.n_base_scatterers)

        # Loop over batches
        # -----------------

        batched_delta_amps = []
        for batch in self.batching.batches:
            l_max = batch.l_max
            batch_id = batch.batch_id
            energy_ids = jnp.asarray(batch.energy_indices)

            # propagators - already rotated
            propagators = self._calculate_propagators(displacements_au, energy_ids)

            # crop propagators
            propagators = propagators[:, :, :(l_max+1)**2, :(l_max+1)**2]

            # dynamic t-matrices
            t_matrices = self._calculate_t_matrices(vib_amps_au, energy_ids)

            # crop t-matrices
            ref_t_matrices = jnp.asarray(self.ref_t_matrices)[energy_ids, :, :l_max+1]
            t_matrices = t_matrices[:, :, :l_max+1]

            # tensor amplitudes
            tensor_amps_in = jnp.asarray(self.tensor_amps_in[batch_id])
            tensor_amps_out = jnp.asarray(self.tensor_amps_out[batch_id])

            # map t-matrices to compressed quantum index
            mapped_t_matrix_vib = jax.vmap(jax.vmap(
                map_l_array_to_compressed_quantum_index,
                in_axes=(0, None)), in_axes=(0, None))(t_matrices, l_max)
            mapped_t_matrix_ref = jax.vmap(jax.vmap(
                map_l_array_to_compressed_quantum_index,
                in_axes=(0, None)), in_axes=(0, None))(ref_t_matrices, l_max)

            # for every energy
            def calc_energy(e_id):
                en_propagators = propagators[e_id, :, ...]

                en_t_matrix_vib = mapped_t_matrix_vib[e_id]
                en_t_matrix_ref = mapped_t_matrix_ref[e_id]

                def f_calc(a):
                    # calculate delta t matrix which contains all perturbations
                    delta_t_matrix = calculate_delta_t_matrix(
                        en_propagators[a, :, :].conj(),
                        en_t_matrix_vib[a],
                        en_t_matrix_ref[a],
                        chem_weights[a]
                    )

                    # Sum from equation (41) in Rous, Pendry 1989
                    return jnp.einsum(
                        'bl,lk,k->b',
                        tensor_amps_out[e_id, a],
                        delta_t_matrix,
                        tensor_amps_in[e_id, a],
                        optimize='optimal')

                batch_amps = jax.vmap(f_calc, in_axes=(0,), out_axes=0)(atom_ids)
                amps = jnp.sum(batch_amps, axis=0)

                del en_propagators
                del en_t_matrix_vib
                del en_t_matrix_ref
                return amps

            # map over energies
            l_delta_amps = jax.lax.map(calc_energy, jnp.arange(len(batch)))

            batched_delta_amps.append(l_delta_amps)

        # now re-sort the delta_amps to the original order
        delta_amps = jnp.concatenate(batched_delta_amps, axis=0)
        delta_amps = delta_amps[self.batching.restore_sorting]

        # Finally apply the prefactors calculated earlier to the result
        delta_amps = delta_amps * self.delta_amp_prefactors

        return delta_amps

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def jit_delta_amplitude(self, free_params):
        return self.delta_amplitude(free_params)

    def intensity(self, free_params):
        delta_amplitude = self.delta_amplitude(free_params)
        _, _, geo_params, _ = self.parameter_space.split_free_params(jnp.asarray(free_params))
        intensity_prefactors = self._intensity_prefactors(
            self.parameter_space.potential_onset_height_change(geo_params)
        )
        intensities = sum_intensity(intensity_prefactors,
                                    self.ref_data.ref_amps,
                                    delta_amplitude)
        return intensities

    @property
    def reference_intensity(self):
        dummy_delta_amps = jnp.zeros((len(self.energies), self.n_beams,), dtype=jnp.complex128)
        intensity_prefactors = self._intensity_prefactors(jnp.array(0.))
        intensities = sum_intensity(intensity_prefactors,
                                    self.ref_data.ref_amps,
                                    dummy_delta_amps)
        return intensities

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def jit_intensity(self, free_params):
        return self.intensity(free_params)

    @property
    def unperturbed_intensity(self):
        """Return intensity from reference data without pertubation."""
        raise NotImplementedError

    def interpolated(self, free_params, deriv_deg=0):
        spline = interpax.CubicSpline(self.origin_grid,
                                      self.intensity(free_params),
                                      bc_type=self.bc_type,
                                      extrapolate=False,
                                      check=False,                              # TODO: do check once in the object creation
        )
        for i in range(deriv_deg):
            spline = spline.derivative()
        return spline(self.target_grid)

    @partial(jax.jit, static_argnames=('self', 'deriv_deg'))
    def jit_interpolated(self, free_params, deriv_deg=0):
        return self.interpolated(free_params, deriv_deg)

    def R(self, free_params):
        _free_params = jnp.asarray(free_params)
        if self.comp_intensity is None:
            raise ValueError("Comparison intensity not set.")
        v0i_electron_volt = -self.v0i*HARTREE
        non_interpolated_intensity = self.intensity(_free_params)

        v0r_param, *_ = self.parameter_space.split_free_params(jnp.asarray(_free_params))
        v0r_shift = self.parameter_space.v0r_transformer(v0r_param)

        # apply v0r shift
        theo_spline = interpax.CubicSpline(self.origin_grid + v0r_shift,
                                           non_interpolated_intensity,
                                           check=False,
                                           extrapolate=False)
        return self.rfactor_func(
            theo_spline,
            v0i_electron_volt,
            self.interpolation_step,
            self.target_grid,
            self.exp_spline
        )

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def jit_R(self, free_params):
        return self.R(free_params)

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def jit_grad_R(self, free_params):
        return jax.grad(self.R)(free_params)

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def jit_R_val_and_grad(self, free_params):
        return jax.value_and_grad(self.R)(free_params)

    # JAX PyTree methods

    def tree_flatten(self):
        dynamic_elements = {
            'rfactor_name': _R_FACTOR_SYNONYMS[self.rfactor_func][0]
        }
        simple_elements = {
            'ref_data': self.ref_data,
            'ref_t_matrices': self.ref_t_matrices,
            'phaseshifts': self.phaseshifts,
            'batching': self.batching,
            'interpolation_deg': self.interpolation_deg,
            'bc_type': self.bc_type,
            'interpolation_step': self.interpolation_step,
            'beam_indices': self.beam_indices,
            'ref_vibrational_amps': self.ref_vibrational_amps,
            'unit_cell': self.unit_cell,
            'target_grid': self.target_grid,
            'phi': self.phi,
            'theta': self.theta,
            'is_surface_atom': self.is_surface_atom,
            '_parameter_space': self.parameter_space,
            'exp_spline': self.exp_spline,
            'comp_intensity': self.comp_intensity,
            'comp_energies': self.comp_energies,
            'origin_grid': self.origin_grid,
            'delta_amp_prefactors': self.delta_amp_prefactors,
            'tensor_amps_in': self.tensor_amps_in,
            'tensor_amps_out': self.tensor_amps_out,
            '_static_propagators': self._static_propagators,
            'propagtor_symmetry_operations': self.propagator_symmetry_operations,
            'propagator_transpose_int': self.propagator_transpose_int,
        }
        aux_data = (dynamic_elements, simple_elements)
        children = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dynamic_elements, simple_elements = aux_data

        calculator = cls.__new__(cls)
        # set static elements
        for kw, value in simple_elements.items():
            setattr(calculator, kw, value)

        # set dynamic elements
        calculator.set_rfactor(dynamic_elements['rfactor_name'])

        return calculator


def benchmark_calculator(calculator, free_params, n_repeats=10):
    if n_repeats < 1:
        raise ValueError("Number of repeats must be greater than 0.")

    # R factor
    start_time = time.time()
    calculator.jit_R(free_params).block_until_ready()
    r_fac_compile_time = time.time() - start_time

    start_time = time.time()
    for _ in range(n_repeats):
        start_time = time.time()
        calculator.jit_R(free_params).block_until_ready()
    r_fac_time = (time.time() - start_time)/n_repeats

    if r_fac_compile_time < 3*r_fac_time:
        # function was most likely already jit compiled
        r_fac_compile_time = None

    # gradients
    start_time = time.time()
    calculator.jit_grad_R(free_params).block_until_ready()
    grad_compile_time = time.time() - start_time

    start_time = time.time()
    for _ in range(n_repeats):
        calculator.jit_grad_R(free_params).block_until_ready()
    grad_time = (time.time() - start_time)/n_repeats

    if grad_compile_time < 3*grad_time:
        # function was most likely already jit compiled
        grad_compile_time = None

    return r_fac_compile_time, r_fac_time, grad_compile_time, grad_time


def calculate_delta_t_matrix(propagator, t_matrix_vib, t_matrix_ref, chem_weight):
    # delta_t_matrix is the change of the atomic t-matrix with new
    # vibrational amplitudes and after applying the displacement
    # Equation (33) in Rous, Pendry 1989
    delta_t_matrix = jnp.einsum(
        'mi, mn, ln-> il',
        propagator,
        jnp.diag(1j*t_matrix_vib),
        propagator,
        optimize='optimal'
    )
    delta_t_matrix = delta_t_matrix - jnp.diag(1j*t_matrix_ref)
    delta_t_matrix = delta_t_matrix*chem_weight
    return delta_t_matrix
