"""Module tensor_calculator."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@Paulhai7)')
__created__ = '2024-05-03'

import copy

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from viperleed.calc import LOGGER as logger
from viperleed.calc.files import poscar
from viperleed.calc.files.iorfactor import beamlist_to_array
from viperleed.calc.files.vibrocc import writeVIBROCC

from viperleed_jax import atomic_units, lib_math, rfactor
from viperleed_jax.batching import Batching
from viperleed_jax.constants import BOHR, HARTREE
from viperleed_jax.dense_quantum_numbers import (
    map_l_array_to_compressed_quantum_index,
)
from viperleed_jax.interpolation import *
from viperleed_jax.interpolation import interpolate_ragged_array
from viperleed_jax.lib_intensity import sum_intensity
from viperleed_jax.propagator import calc_propagator, symmetry_operations
from viperleed_jax.t_matrix import vib_dependent_tmatrix
from viperleed_jax.rfactor import R_FACTOR_SYNONYMS
from viperleed_jax import utils


@register_pytree_node_class
class TensorLEEDCalculator:
    """Main class for calculating tensor LEED intensities and R-factors.

    Parameters:
    ----------
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
    """

    def __init__(
        self,
        ref_calc_params,
        ref_calc_result,
        phaseshifts,
        slab,
        rparams,
        interpolation_step=0.5,
        interpolation_deg=3,
        bc_type='not-a-knot',
        batch_energies=None,
        batch_atoms=None,
        recalculate_ref_t_matrices=False,
    ):
        self.ref_calc_params = ref_calc_params
        self.ref_calc_result = ref_calc_result
        self.phaseshifts = phaseshifts
        self.recalculate_ref_t_matrices = recalculate_ref_t_matrices

        self.interpolation_deg = interpolation_deg
        self.bc_type = bc_type

        # beam indices
        beam_indices = [beam.hk for beam in rparams.ivbeams]
        self.beam_indices = jnp.array([beam.hk for beam in rparams.ivbeams])
        self.n_beams = self.beam_indices.shape[0]

        self.comp_intensity = None
        self.comp_energies = None
        self.interpolation_step = interpolation_step
        self._parameter_space = None

        self.target_grid = jnp.arange(
            rparams.THEO_ENERGIES.start,
            rparams.THEO_ENERGIES.stop,
            self.interpolation_step,
        )

        # unit cell in Bohr radii
        self.unit_cell = slab.ab_cell.copy() / BOHR

        # theta and phi (in radians)
        self.theta = jnp.deg2rad(rparams.THETA)
        self.phi = jnp.deg2rad(rparams.PHI)

        # TODO: refactor into a dataclass
        self.max_l_max = self.ref_calc_params.max_lmax
        self.energies = jnp.asarray(self.ref_calc_params.energies)

        non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
        # TODO check this
        self.is_surface_atom = jnp.array(
            [at.layer.num == 0 for at in non_bulk_atoms]
        )

        self.ref_vibrational_amps = jnp.array(
            [at.site.vibamp[at.el] for at in non_bulk_atoms]
        )
        self.origin_grid = ref_calc_params.incident_energy_ev

        self.delta_amp_prefactors = self._calc_delta_amp_prefactors()

        self.exp_spline = None

        # set batch sizes
        self.batch_energies = batch_energies
        if batch_atoms is None:
            self.batch_atoms = self.n_atoms
        else:
            self.batch_atoms = batch_atoms

        # default R-factor is Pendry
        self.rfactor_func = rfactor.pendry_R

        if self.interpolation_deg != 3:
            raise NotImplementedError

        # calculate batching
        self.batching = Batching(self.energies, ref_calc_params.lmax)

        # get experimental intensities and hk
        exp_energies, _, _, exp_intensities = beamlist_to_array(
            rparams.expbeams
        )
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
        mapped_exp_intensities = np.where(
            mask_out_expbeam, np.nan, mapped_exp_intensities
        )
        self.set_experiment_intensity(mapped_exp_intensities, exp_energies)

        self.kappa = jnp.array(self.ref_calc_params.kappa)

    @property
    def unit_cell_area(self):
        return jnp.linalg.norm(
            jnp.cross(self.unit_cell[:, 0], self.unit_cell[:, 1])
        )

    @property
    def reciprocal_unit_cell(self):
        return 2 * jnp.pi * jnp.linalg.inv(self.unit_cell)

    @property
    def n_atoms(self):
        return len(self.ref_vibrational_amps)

    @property
    def parameter_space(self):
        if self._parameter_space is None:
            raise ValueError('Parameter space not set.')
        return self._parameter_space

    @property
    def n_free_parameters(self):
        return self.parameter_space.n_free_params

    @property
    def atom_ids(self):
        # atom ids that will be batched over
        return jnp.arange(self.parameter_space.n_atom_basis)

    def set_rfactor(self, rfactor_name):
        _rfactor_name = rfactor_name.lower().strip()
        for func, synonyms in R_FACTOR_SYNONYMS.items():
            if _rfactor_name in synonyms:
                self.rfactor_func = func
                logger.info(f'R-factor set to {func.__name__}.')
                return
        err_msg = f'Unknown R-factor name: {rfactor_name}'
        raise ValueError(err_msg)

    def set_experiment_intensity(self, comp_intensity, comp_energies):
        logger.debug(
            'Setting experimental intensities and initializing interpolators.'
        )

        self.comp_intensity = comp_intensity
        self.comp_energies = comp_energies
        self.exp_spline = interpolate_ragged_array(
            self.comp_energies,
            self.comp_intensity,
            bc_type=self.bc_type,
        )


    def set_parameter_space(self, parameter_space):
        if self._parameter_space is not None:
            logger.debug('Overwriting parameter space.')
        # take delta_slab and set the parameter space
        self._parameter_space = parameter_space.freeze()
        logger.info(f'Parameter space set.\n{parameter_space.info}')
        logger.info(
            'This parameter space requires dynamic calculation of '
            f'{self._parameter_space.n_dynamic_t_matrices} t-matrice(s) and '
            f'{self._parameter_space.n_dynamic_propagators} propagator(s).'
        )

        if self.recalculate_ref_t_matrices:
            # calculate reference t-matrices for full LMAX
            n_ref_vib_amps = len(parameter_space.vib_tree.leaves)
            logger.debug(
                f'Calculating {n_ref_vib_amps} reference t-matrices for '
                f'LMAX={self.max_l_max}.'
            )
            ref_vib_amps = [
                leaf.ref_vib_amp for leaf in parameter_space.vib_tree.leaves
            ]
            site_elements = [
                leaf.site_element for leaf in parameter_space.vib_tree.leaves
            ]
            self.ref_t_matrices = self._calculate_reference_t_matrices(
                ref_vib_amps, site_elements
            )
        else:
            # use the stored reference t-matrices from reference calculation
            self.ref_t_matrices = self.ref_calc_result.t_matrices

        # pre-calculate the static t-matrices
        logger.debug(
            f'Pre-calculating {self._parameter_space.n_static_t_matrices} '
            'static t-matrice(s).'
        )
        self._calculate_static_t_matrices()

        # rotation angles
        propagator_symmetry_operations, propagator_transpose = (
            self._propagator_rotation_factors()
        )
        self.propagator_symmetry_operations = jnp.asarray(
            propagator_symmetry_operations
        )
        # NB: Using an integer array here because there seems so be some kind of
        # bug where jax.jit would flip on of the boolean values for some
        # cases.
        self.propagator_transpose_int = propagator_transpose.astype(jnp.int32)

        # pre-calculate the static propagators
        logger.debug(
            f'Pre-calculating {self._parameter_space.n_static_propagators} '
            'static propagator(s).'
        )
        self._calculate_static_propagators()

    def _calculate_static_t_matrices(self):
        # This is only done once – perform for maximum lmax and crop later
        energy_indices = jnp.arange(len(self.energies))

        # Outer loop: iterate over energy indices with batching
        def energy_fn(e_idx):
            # For each energy, compute t-matrices for all static input pairs.
            # self._parameter_space.static_t_matrix_inputs is assumed to be a list
            # of (site_el, vib_amp) pairs.
            def compute_t(pair):
                site_el, vib_amp = pair
                return vib_dependent_tmatrix(
                    self.max_l_max,
                    self.phaseshifts[site_el][e_idx, : self.max_l_max + 1],
                    self.energies[e_idx],
                    vib_amp,
                )

            # Use a Python loop to compute for each pair and stack the results.
            # This loop is over a typically small list so it shouldn't be a bottleneck.
            return jnp.stack(
                [
                    compute_t(pair)
                    for pair in self._parameter_space.static_t_matrix_inputs
                ]
            )

        # Map over energies with the given batch size.
        static_t_matrices = jax.lax.map(
            energy_fn, energy_indices, batch_size=self.batch_energies
        )
        # static_t_matrices has shape (num_energies, num_static_inputs, lm, ...),
        # which is equivalent to the original einsum('ael->eal') result.
        self._static_t_matrices = static_t_matrices

    def _calculate_static_propagators(self):
        # Convert static propagator inputs to an array.
        static_inputs = self._parameter_space.static_propagator_inputs
        if len(static_inputs) == 0:
            # If there are no static inputs, store an empty array.
            self._static_propagators = jnp.array([])
            return

        displacements_ang = jnp.asarray(static_inputs)
        displacements_au = atomic_units.to_internal_displacement_vector(
            displacements_ang
        )
        spherical_harmonics_components = jnp.array(
            [
                lib_math.spherical_harmonics_components(self.max_l_max, disp)
                for disp in displacements_au
            ]
        )

        # Outer loop: iterate over energy indices.
        def energy_fn(e_idx):
            # For each energy, iterate over all displacements.
            def displacement_fn(i):
                disp = displacements_au[i]
                comps = spherical_harmonics_components[i]
                return calc_propagator(
                    self.max_l_max,
                    disp,
                    comps,
                    self.kappa[e_idx],
                )

            return jax.lax.map(
                displacement_fn,
                jnp.arange(displacements_au.shape[0]),
                batch_size=self.batch_atoms,
            )

        # Map over energies with the specified batch size.
        static_propagators = jax.lax.map(
            energy_fn,
            jnp.arange(len(self.energies)),
            batch_size=self.batch_energies,
        )
        # The result has shape (num_energies, num_displacements, ...).
        # Use einsum to swap axes so that the final shape is
        # (num_displacements, num_energies, ...), matching the original ordering.
        self._static_propagators = jnp.einsum('ed...->de...', static_propagators)

    def _calculate_dynamic_t_matrices(self, vib_amps, energy_indices):
        # Convert energy_indices to a JAX array for the outer mapping.
        energy_indices = jnp.array(energy_indices)
        # Pre-build the static list of (vib_amp, site_element) pairs.
        pairs = list(zip(vib_amps, self.parameter_space.dynamic_t_matrix_site_elements))

        def energy_map_fn(e_idx):
            # For each energy index, loop over the static pairs.
            results = []
            for vib_amp, site_el in pairs:
                result = vib_dependent_tmatrix(
                    self.max_l_max,
                    self.phaseshifts[site_el][e_idx, : self.max_l_max + 1],
                    self.energies[e_idx],
                    vib_amp.reshape(),  # reshape from (1,) to scalar for grad compatibility
                )
                results.append(result)
            return jnp.stack(results)

        dynamic_t_matrices = jax.lax.map(energy_map_fn, energy_indices,
                                         batch_size=self.batch_energies)
        return jnp.asarray(dynamic_t_matrices)

    def _calculate_reference_t_matrices(self, ref_vib_amps, site_elements):
        def map_fn(pair):
            vib_amp, site_el = pair
            return vib_dependent_tmatrix(
                self.max_l_max,
                self.phaseshifts[site_el][:, : self.max_l_max + 1],
                self.energies,
                vib_amp,
            )

        ref_t_matrices = jax.lax.map(map_fn, (ref_vib_amps, site_elements),
                                     batch_size=self.batch_atoms)
        return jnp.einsum('ael->eal', ref_t_matrices)

    def _calculate_t_matrices(self, vib_amps, energy_indices):
        # Process one energy at a time to reduce memory usage.
        energy_indices = jnp.array(energy_indices)

        def energy_fn(e_idx):
            # Compute the dynamic t-matrix for a single energy.
            # _calculate_dynamic_t_matrices expects a sequence of energies; here we pass a list of one index.
            dyn_t = self._calculate_dynamic_t_matrices(vib_amps, [e_idx])[0]
            # Map the dynamic t-matrix to the atom-site-element basis.
            dyn_mapped = dyn_t[self.parameter_space.t_matrix_id]
            # Get the corresponding static t-matrix, or zeros if none exist.
            if len(self._static_t_matrices) == 0:
                stat_t = jnp.zeros_like(dyn_t)
            else:
                stat_t = self._static_t_matrices[e_idx, :, :]
            stat_mapped = stat_t[self.parameter_space.t_matrix_id, :]
            # Select between dynamic and static for this energy.
            # The condition is broadcasted to shape (num_selected, lm)
            return jnp.where(
                self.parameter_space.is_dynamic_t_matrix[:, jnp.newaxis],
                dyn_mapped,
                stat_mapped,
            )

        # Process each energy one by one.
        t_matrices = jax.lax.map(energy_fn, energy_indices,
                                 batch_size=self.batch_energies)
        return t_matrices

    def _calculate_dynamic_propagator(self, displacements, components, kappa):
        """
        Compute dynamic propagators for a single energy index.

        Returns an array of shape (num_displacements, ...).
        """
        return jax.lax.map(
            lambda atom_idx: calc_propagator(
                self.max_l_max,
                displacements[atom_idx],
                components[atom_idx],
                kappa,
            ),
            jnp.arange(len(displacements)),
            batch_size=self.batch_atoms,
        )

    def _calculate_propagators(
        self, displacements, displacements_components, energy_indices
    ):
        # We want the final result indexed as (energies, atom_basis, lm, l'm')
        energy_indices = jnp.array(energy_indices)

        def process_energy(e_idx):
            # --- Dynamic propagators ---
            if len(displacements) > 0:
                # Now call the per-energy dynamic propagator.
                dyn = self._calculate_dynamic_propagator(
                    displacements,
                    displacements_components,
                    self.kappa[e_idx],
                )
            else:
                dyn = jnp.zeros_like(self._static_propagators[0])

            # --- Static propagators ---
            if len(self._static_propagators) == 0:
                stat = jnp.zeros_like(dyn)
            else:
                # Assuming self._static_propagators is indexed as (atom_basis, num_energies, lm, m)
                stat = self._static_propagators[:, e_idx, :, :]

            # --- Map to atom basis using propagator_id ---
            mapped_dyn = dyn[self.parameter_space.propagator_id]
            mapped_stat = stat[self.parameter_space.propagator_id]

            # --- Combine dynamic and static parts ---
            # Condition is broadcast along the last two axes.
            cond = self.parameter_space.is_dynamic_propagator[:, None, None]
            combined = jnp.where(cond, mapped_dyn, mapped_stat)
            # combined now has shape (atom_basis, lm, m)

            # --- Apply selective transposition ---
            trans_int = self.propagator_transpose_int[:, None, None]
            combined = (1 - trans_int) * combined + trans_int * jnp.transpose(
                combined, (0, 2, 1)
            )
            # combined remains (atom_basis, lm, m)

            return combined

        # Process each energy individually.
        # Each process_energy returns (atom_basis, lm, m); mapping over energies yields shape:
        # (num_energies, atom_basis, lm, m)
        per_energy = jax.lax.map(
            process_energy, energy_indices, batch_size=self.batch_energies
        )
        # Transpose to (atom_basis, num_energies, lm, m) to match what the symmetry einsum expects.
        per_energy = jnp.transpose(per_energy, (1, 0, 2, 3))

        # --- Apply rotations (symmetry operations) and rearrange ---
        propagators = jnp.einsum(
            'aelm,alm->ealm',
            per_energy,
            self.propagator_symmetry_operations,
            optimize='optimal',
        )
        # Final shape is (energies, atom_basis, lm, m)
        return propagators


    # TODO: for testing purposes: contrib should be exactly 0 for not pertubations and if recalculate_ref_t_matrices=True
    def _calculate_static_ase_contributions(self):
        static_ase_contributions = np.zeros(
            (
                len(self.energies),
                self.n_beams,
                self.parameter_space.n_static_ase,
            ),
            dtype=np.complex128,
        )

        for batch in self.batching.batches:
            l_max = batch.l_max
            batch_id = batch.batch_id
            energy_ids = jnp.asarray(batch.energy_indices)

            # get and reindex static t-matrices
            if len(self._static_t_matrices) == 0:
                raise ValueError('No static t-matrices found.')
            static_t_matrices = self._static_t_matrices[energy_ids, :, :]
            # broadcast to complete ase basis
            static_t_matrices = static_t_matrices[
                :, self.parameter_space.t_matrix_id, :
            ]
            # broadcast down to static ases
            static_t_matrices = static_t_matrices[
                :, self.parameter_space.static_ase_id, :
            ]

            if len(self._static_propagators) == 0:
                raise ValueError('No static propagators found.')
            static_propagators = self._static_propagators[:, energy_ids, :, :]
            # broadcast to complete ase basis
            static_propagators = static_propagators[
                self.parameter_space.propagator_id, ...
            ]
            # apply rotations
            static_propagators = jnp.einsum(
                'aelm,alm->aelm',
                static_propagators,
                self.propagator_symmetry_operations,
                optimize='optimal',
            )
            # broadcast down to static axes
            static_propagators = static_propagators[
                self.parameter_space.static_ase_id, ...
            ]

            # crop t-matrices and propagators
            t_matrices = static_t_matrices[:, :, : l_max + 1]
            propagators = static_propagators[
                :, :, : (l_max + 1) ** 2, : (l_max + 1) ** 2
            ]

            # reference t-matrix
            ref_t_matrices = jnp.asarray(self.ref_t_matrices)
            ref_t_matrices = ref_t_matrices[energy_ids, :, : l_max + 1]
            ref_t_matrices = ref_t_matrices[
                :, self.parameter_space.static_ase_id, :
            ]

            tensor_amps_in = self.tensor_amps_in[batch_id]
            tensor_amps_out = self.tensor_amps_out[batch_id]

            # get only static elements
            tensor_amps_in = tensor_amps_in[
                :, self.parameter_space.static_ase_id
            ]
            tensor_amps_out = tensor_amps_out[
                :, self.parameter_space.static_ase_id
            ]

            for seq_e_id, e_id in enumerate(energy_ids):
                for seq_ase_id in range(self.parameter_space.n_static_ase):
                    mapped_t_matrix = map_l_array_to_compressed_quantum_index(
                        t_matrices[seq_e_id, seq_ase_id], l_max
                    )
                    mapped_t_matrix_ref = (
                        map_l_array_to_compressed_quantum_index(
                            ref_t_matrices[seq_e_id, seq_ase_id], l_max
                        )
                    )
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
        v_imag = self.ref_calc_params.v0i

        # energy dependent quantities
        out_k_par2 = self.ref_calc_params.kx_in
        out_k_par3 = self.ref_calc_params.ky_in

        k_inside = jnp.sqrt(2 * energies - 2j * v_imag + 1j * lib_math.EPS)

        # Propagator evaluated relative to the muffin tin zero i.e.
        # it uses energy = incident electron energy + inner potential
        out_k_par = out_k_par2**2 + out_k_par3**2
        out_k_perp_inside = jnp.sqrt(
            ((2 * energies - 2j * v_imag)[:, jnp.newaxis] - out_k_par)
            + 1j * lib_math.EPS
        )

        # Prefactors from Equation (41) from Rous, Pendry 1989
        prefactors = jnp.einsum(
            'e,eb,->eb',
            1 / k_inside,
            1 / out_k_perp_inside,
            1 / (2 * (self.unit_cell_area)),
        )
        return prefactors

    def _intensity_prefactors(self, onset_height_change):
        # onset height change was called CXDisp in the original code

        # from lib_intensity
        (in_k_vacuum, in_k_perp_vacuum, out_k_perp, out_k_perp_vacuum) = (
            self._wave_vectors()
        )

        a = out_k_perp_vacuum
        c = in_k_vacuum * jnp.cos(self.theta)

        # TODO: re-check if it should be a.real or abs(a)
        prefactor = (
            abs(
                jnp.exp(
                    -1j
                    * onset_height_change
                    / BOHR
                    * (
                        jnp.outer(
                            in_k_perp_vacuum, jnp.ones(shape=(self.n_beams,))
                        )
                        + out_k_perp
                    )
                )
            )
            ** 2
            * a.real
            / jnp.outer(c, jnp.ones(shape=(self.n_beams,))).real
        )
        return prefactor

    def _wave_vectors(self):
        e_kin = self.energies
        v_real = self.ref_calc_params.v0r
        v_imag = self.ref_calc_params.v0i
        n_energies = e_kin.shape[0]
        n_beams = self.beam_indices.shape[0]
        # incident wave vector
        in_k_vacuum = jnp.sqrt(jnp.maximum(0, 2 * (e_kin - v_real)))
        in_k_par = in_k_vacuum * jnp.sin(self.theta)  # parallel component
        in_k_par_2 = in_k_par * jnp.cos(self.phi)  # shape =( n_energy )
        in_k_par_3 = in_k_par * jnp.sin(self.phi)  # shape =( n_energy )
        in_k_perp_vacuum = (
            2 * e_kin - in_k_par_2**2 - in_k_par_3**2 - 2 * 1j * v_imag
        )
        in_k_perp_vacuum = jnp.sqrt(in_k_perp_vacuum)

        # outgoing wave vector components
        in_k_par_components = jnp.stack(
            (in_k_par_2, in_k_par_3)
        )  # shape =(n_en, 2)
        in_k_par_components = jnp.outer(
            in_k_par_components, jnp.ones(shape=(n_beams,))
        ).reshape((n_energies, 2, n_beams))  # shape =(n_en ,2 ,n_beams)
        out_wave_vec = jnp.dot(
            self.beam_indices, self.reciprocal_unit_cell
        )  # shape =(n_beams, 2)
        out_wave_vec = jnp.outer(
            jnp.ones_like(e_kin), out_wave_vec.transpose()
        ).reshape((n_energies, 2, n_beams))  # shape =(n_en , n_beams)
        out_k_par_components = in_k_par_components + out_wave_vec

        # out k vector
        out_k_perp_vacuum = (
            2 * jnp.outer(e_kin - v_real, jnp.ones(shape=(n_beams,)))
            - out_k_par_components[:, 0, :] ** 2
            - out_k_par_components[:, 1, :] ** 2
        ).astype(dtype='complex64')
        out_k_perp = jnp.sqrt(
            out_k_perp_vacuum
            + 2 * jnp.outer(v_real - 1j * v_imag, jnp.ones(shape=(n_beams,)))
        )
        out_k_perp_vacuum = jnp.sqrt(out_k_perp_vacuum)

        return in_k_vacuum, in_k_perp_vacuum, out_k_perp, out_k_perp_vacuum

    def _propagator_rotation_factors(self):
        ops = [
            symmetry_operations(self.max_l_max, plane_sym_op)
            for plane_sym_op in self.parameter_space.propagator_plane_symmetry_operations
        ]
        symmetry_tensors = np.array([op[0] for op in ops])
        mirror_propagators = np.array([op[1] for op in ops])

        return symmetry_tensors, mirror_propagators

    def delta_amplitude(self, free_params):
        """Calculate the delta amplitude for a given set of free parameters."""
        _free_params = jnp.asarray(free_params)
        # split free parameters
        (_, vib_params, geo_parms, occ_params) = (
            self.parameter_space.split_free_params(jnp.asarray(_free_params))
        )

        # displacements, converted to atomic units
        displacements_ang = self.parameter_space.reference_displacements(
            geo_parms
        )
        displacements_ang = jnp.asarray(displacements_ang)
        displacements_au = atomic_units.to_internal_displacement_vector(
            displacements_ang
        )
        displacement_components = jnp.array(
            [
                lib_math.spherical_harmonics_components(
                    self.max_l_max, displacement
                )
                for displacement in displacements_au
            ]
        )

        # vibrational amplitudes, converted to atomic units
        vib_amps_au = self.parameter_space.reference_vib_amps(vib_params)
        # vib_amps_au = jax.vmap(atomic_units.to_internal_vib_amps,
        #                         in_axes=0)(vib_amps_ang)

        # chemical weights
        chem_weights = jnp.asarray(
            self.parameter_space.occ_weight_transformer(occ_params)
        )

        # Loop over batches
        # -----------------

        # Use python for loop here, as batches can have different array sizes

        batched_delta_amps = []
        for batch in self.batching.batches:
            l_max = batch.l_max
            energy_ids = jnp.asarray(batch.energy_indices)

            # propagators - already rotated
            propagators = self._calculate_propagators(
                displacements_au, displacement_components, energy_ids
            )

            # crop propagators
            propagators = propagators[
                :, :, : (l_max + 1) ** 2, : (l_max + 1) ** 2
            ]

            # dynamic t-matrices
            t_matrices = self._calculate_t_matrices(vib_amps_au, energy_ids)

            # crop t-matrices
            ref_t_matrices = jnp.asarray(self.ref_t_matrices)[
                energy_ids, :, : l_max + 1
            ]
            t_matrices = t_matrices[:, :, : l_max + 1]

            # tensor amplitudes
            # tensor_amps_in = self.ref_calc_result.in_amps
            # tensor_amps_out = self.ref_calc_result.out_amps

            # map t-matrices to compressed quantum index
            mapped_t_matrix_vib = jax.vmap(
                jax.vmap(
                    map_l_array_to_compressed_quantum_index, in_axes=(0, None)
                ),
                in_axes=(0, None),
            )(t_matrices, l_max)
            mapped_t_matrix_ref = jax.vmap(
                jax.vmap(
                    map_l_array_to_compressed_quantum_index, in_axes=(0, None)
                ),
                in_axes=(0, None),
            )(ref_t_matrices, l_max)

            # for every energy
            @jax.checkpoint # seems to be faster # TODO: test other checkpointing
            def calc_energy(e_id):
                en_propagators = propagators[e_id, :, ...]
                en_t_matrix_vib = mapped_t_matrix_vib[e_id]
                en_t_matrix_ref = mapped_t_matrix_ref[e_id]

                def compute_atom_contrib(a):
                    delta_t_matrix = calculate_delta_t_matrix(
                        en_propagators[a, :, :].conj(),
                        en_t_matrix_vib[a],
                        en_t_matrix_ref[a],
                        chem_weights[a],
                    )
                    # Sum from equation (41) in Rous, Pendry 1989
                    return jnp.einsum(
                        'bl,lk,k->b',
                        self.ref_calc_result.out_amps[e_id, a],
                        delta_t_matrix,
                        self.ref_calc_result.in_amps[e_id, a],
                        optimize='optimal',
                    )

                # Use lax.map with a batch_size of n_atom
                contributions = jax.lax.map(
                    compute_atom_contrib, self.atom_ids, batch_size=self.batch_atoms
                )
                return jnp.sum(contributions, axis=0)

            # map over energies
            l_delta_amps = jax.lax.map(calc_energy, jnp.arange(len(batch)))

            batched_delta_amps.append(l_delta_amps)

        # now re-sort the delta_amps to the original order
        delta_amps = jnp.concatenate(batched_delta_amps, axis=0)
        delta_amps = delta_amps[self.batching.restore_sorting]

        # Finally apply the prefactors calculated earlier to the result
        delta_amps = delta_amps * self.delta_amp_prefactors

        return delta_amps

    @partial(
        jax.jit, static_argnames=('self')
    )  # TODO: not good, redo as pytree
    def jit_delta_amplitude(self, free_params):
        return self.delta_amplitude(free_params)

    def intensity(self, free_params):
        delta_amplitude = self.delta_amplitude(free_params)
        _, _, geo_params, _ = self.parameter_space.split_free_params(
            jnp.asarray(free_params)
        )
        intensity_prefactors = self._intensity_prefactors(
            self.parameter_space.potential_onset_height_change(geo_params)
        )
        intensities = sum_intensity(
            intensity_prefactors, self.ref_calc_result.ref_amps, delta_amplitude
        )
        return intensities

    @property
    def reference_intensity(self):
        dummy_delta_amps = jnp.zeros(
            (
                len(self.energies),
                self.n_beams,
            ),
            dtype=jnp.complex128,
        )
        intensity_prefactors = self._intensity_prefactors(jnp.array(0.0))
        intensities = sum_intensity(
            intensity_prefactors, self.ref_calc_result.ref_amps, dummy_delta_amps
        )
        return intensities

    @partial(
        jax.jit, static_argnames=('self')
    )  # TODO: not good, redo as pytree
    def jit_intensity(self, free_params):
        return self.intensity(free_params)

    @property
    def unperturbed_intensity(self):
        """Return intensity from reference data without pertubation."""
        raise NotImplementedError

    def interpolated(self, free_params, deriv_deg=0):
        spline = interpax.CubicSpline(
            self.origin_grid,
            self.intensity(free_params),
            bc_type=self.bc_type,
            extrapolate=False,
            check=False,  # TODO: do check once in the object creation
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
            raise ValueError('Comparison intensity not set.')
        v0i_electron_volt = -self.ref_calc_params.v0i * HARTREE
        non_interpolated_intensity = self.intensity(_free_params)

        v0r_param, *_ = self.parameter_space.split_free_params(
            jnp.asarray(_free_params)
        )
        v0r_shift = self.parameter_space.v0r_transformer(v0r_param)

        # apply v0r shift
        theo_spline = interpax.CubicSpline(
            self.origin_grid + v0r_shift,
            non_interpolated_intensity,
            check=False,
            extrapolate=False,
        )
        return self.rfactor_func(
            theo_spline,
            v0i_electron_volt,
            self.interpolation_step,
            self.target_grid,
            self.exp_spline,
        )

    @partial(
        jax.jit, static_argnames=('self'))
    def jit_R(self, free_params):
        """JIT compiled R-factor calculation."""
        return self.R(free_params)

    @partial(
        jax.jit, static_argnames=('self')
    )  # TODO: not good, redo as pytree
    def jit_R_val_and_grad(self, free_params):
        """JIT compiled R-factor calculation with gradient."""
        val, grad = jax.value_and_grad(self.R)(free_params)
        grad = jnp.asarray(grad)
        return val, grad

    def jit_grad_R(self, free_params):
        """JIT compiled R-factor gradient calculation."""
        _, grad = self.jit_R_val_and_grad(free_params)
        return grad
        return jnp.asarray(jax.grad(self.R)(free_params))

    # JAX PyTree methods

    def tree_flatten(self):
        dynamic_elements = {
            'rfactor_name': R_FACTOR_SYNONYMS[self.rfactor_func][0]
        }
        simple_elements = {
            '_parameter_space': self.parameter_space,
            '_static_propagators': self._static_propagators,
            '_static_t_matrices': self._static_t_matrices,
            'batching': self.batching,
            'bc_type': self.bc_type,
            'beam_indices': self.beam_indices,
            'comp_energies': self.comp_energies,
            'comp_intensity': self.comp_intensity,
            'delta_amp_prefactors': self.delta_amp_prefactors,
            'energies': self.energies,
            'exp_spline': self.exp_spline,
            'interpolation_deg': self.interpolation_deg,
            'interpolation_step': self.interpolation_step,
            'is_surface_atom': self.is_surface_atom,
            'n_beams': self.n_beams,
            'origin_grid': self.origin_grid,
            'phaseshifts': self.phaseshifts,
            'phi': self.phi,
            'propagator_symmetry_operations': self.propagator_symmetry_operations,
            'propagator_transpose_int': self.propagator_transpose_int,
            'ref_t_matrices': self.ref_t_matrices,
            'ref_vibrational_amps': self.ref_vibrational_amps,
            'target_grid': self.target_grid,
            'tensor_amps_in': self.tensor_amps_in,
            #'tensor_amps_out': self.tensor_amps_out,
            'theta': self.theta,
            'unit_cell': self.unit_cell,
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

    def benchmark(self, free_params=None, n_repeats=10, csv_file_path=None):
        """Run benchmarks and add log results."""
        logger.info('Runnning timing benchmarks for tensor-LEED calculation...')
        bench_results = utils.benchmark_calculator(
            self, free_params, n_repeats, csv_file_path
        )
        logger.info(utils.format_benchmark_results(bench_results) + '\n')

    # TODO: needs tests
    def write_to_slab(
        self, rpars, slab, base_scatterers, params, write_to_file=True
    ):
        for atom in slab:
            atom.storeOriState()
        for site in slab.sitelist:
            if site.oriState is None:
                tmp = copy.deepcopy(site)
                site.oriState = tmp

        # update geometries just in case
        slab.collapse_fractional_coordinates()
        slab.update_cartesian_from_fractional()
        slab.update_layer_coordinates()

        # expand the reduced paramter vector
        v0r, vibrations, displacements, occupations = (
            self.parameter_space.expand_params(params)
        )

        # update V0r in rpars
        rpars.best_v0r = v0r

        for atom in slab:
            atom.storeOriState()
        for site in slab.sitelist:
            if site.oriState is None:
                tmp = copy.deepcopy(site)
                site.oriState = tmp

        # update geometries just in case
        slab.collapse_fractional_coordinates()
        slab.update_cartesian_from_fractional()
        slab.update_layer_coordinates()

        # expand the reduced paramter vector
        v0r, vibrations, displacements, occupations = (
            self.parameter_space.expand_params(params)
        )
        # convert to numpy arrays
        v0r = np.array(v0r)
        displacements = np.array(displacements)
        # convert displacements from zxy to xyz order
        displacements = displacements[:, [1, 2, 0]]
        vibrations = np.array(vibrations)
        occupations = np.array(occupations)

        for at in slab.atlist:
            if at.is_bulk:
                continue
            at_scatterers = [
                s for s in base_scatterers.scatterers if s.num == at.num
            ]
            scatterer_indices = [
                base_scatterers.scatterers.index(s) for s in at_scatterers
            ]
            scatterer_indices = np.array(scatterer_indices)

            at_displacements = displacements[scatterer_indices]
            at_occupations = occupations[scatterer_indices]

            averaged_displacement = at_displacements * at_occupations
            averaged_displacement = (
                averaged_displacement.sum(axis=0) / at_occupations.sum()
            )

            rel_at_displacements = at_displacements - averaged_displacement

            at.cartpos += averaged_displacement

            for scatterer, rel_disp in zip(at_scatterers, rel_at_displacements):
                atom.disp_geo[scatterer.element] = (
                    rel_disp  # TODO: fix to cartpos
                )

        slab.update_fractional_from_cartesian()

        for site in slab.sitelist:
            for element in site.vibamp.keys():
                siteel_scatterers = [
                    s
                    for s in base_scatterers.scatterers
                    if s.site == site.label and s.element == element
                ]
                if len(siteel_scatterers) == 0:
                    continue

                scatterer_indices = [
                    base_scatterers.scatterers.index(s)
                    for s in siteel_scatterers
                ]
                scatterer_indices = np.array(scatterer_indices)

                scatterer_vibs = vibrations[scatterer_indices]
                scatterer_occupations = occupations[scatterer_indices]

                averaged_vib = scatterer_vibs * scatterer_occupations
                averaged_vib = (
                    averaged_vib.sum(axis=0) / scatterer_occupations.sum()
                )
                averaged_occ = scatterer_occupations.sum() / len(
                    scatterer_occupations
                )

                rel_scatterer_vibs = scatterer_vibs - averaged_vib
                rel_scatterer_occ = scatterer_occupations - averaged_occ

                site.vibamp[element] = averaged_vib
                site.occ[element] = averaged_occ

                for scatterer, rel_vib in zip(
                    siteel_scatterers, rel_scatterer_vibs
                ):
                    scatterer.atom.disp_vib[element] = rel_vib
                    scatterer.atom.disp_occ[element] = rel_scatterer_occ

        # Optionally write to file
        if write_to_file:
            # write POSCAR
            poscar.write(slab, 'POSCAR_TL_optimized', comments='all')
            # write VIBROCC
            writeVIBROCC(slab, rpars, 'VIBROCC_TL_optimized')


def calculate_delta_t_matrix(
    propagator, t_matrix_vib, t_matrix_ref, chem_weight
):
    # delta_t_matrix is the change of the atomic t-matrix with new
    # vibrational amplitudes and after applying the displacement
    # Equation (33) in Rous, Pendry 1989
    delta_t_matrix = jnp.dot(propagator.T * (1j * t_matrix_vib), propagator.T)
    delta_t_matrix = delta_t_matrix - jnp.diag(1j * t_matrix_ref)
    delta_t_matrix = delta_t_matrix * chem_weight
    return delta_t_matrix
