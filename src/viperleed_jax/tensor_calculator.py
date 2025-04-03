"""Module tensor_calculator."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@Paulhai7)')
__created__ = '2024-05-03'

import copy
from functools import partial
from dataclasses import dataclass, field

import numpy as np

import jax
import jax.numpy as jnp
from viperleed.calc import LOGGER as logger
from viperleed.calc.files import poscar
from viperleed.calc.files.iorfactor import beamlist_to_array
from viperleed.calc.files.vibrocc import writeVIBROCC

from viperleed_jax import atomic_units, lib_math, rfactor
from viperleed_jax.batching import Batching
from viperleed_jax.constants import BOHR, HARTREE
from viperleed_jax.dense_quantum_numbers import (
    vmapped_l_array_to_compressed_quantum_index,
)
from viperleed_jax.interpolation import *
from viperleed_jax.interpolation import interpolate_ragged_array
from viperleed_jax.lib_intensity import sum_intensity, intensity_prefactors
from viperleed_jax.propagator import (
    calc_propagator,
    symmetry_operations,
    calculate_propagators,
)
from viperleed_jax.t_matrix import calculate_t_matrices, vib_dependent_tmatrix
from viperleed_jax.rfactor import R_FACTOR_SYNONYMS
from viperleed_jax import utils


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        'kappa',
        'static_propagators',
        'propagator_transpose_int',
        'symmetry_operations',
        'propagator_id',
        'is_dynamic_propagator',
    ],
    meta_fields=[],
)
@dataclass
class PropagatorContext:
    kappa: jnp.ndarray  # shape (n_energies,)
    static_propagators: jnp.ndarray  # shape (atom_basis, n_energies, lm, m)
    propagator_transpose_int: jnp.ndarray  # shape (atom_basis,)
    symmetry_operations: jnp.ndarray  # shape (atom_basis, lm, m)
    propagator_id: jnp.ndarray
    is_dynamic_propagator: jnp.ndarray


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        'energies',
        'static_t_matrices',
        't_matrix_id',
        'is_dynamic_mask',
    ],
    meta_fields=[
        'dynamic_site_elements',
    ],
)
@dataclass
class TMatrixContext:
    energies: jnp.ndarray
    static_t_matrices: jnp.ndarray
    dynamic_site_elements: jnp.ndarray = field(metadata=dict(static=True))
    t_matrix_id: jnp.ndarray
    is_dynamic_mask: jnp.ndarray


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

        # evaluate the wave vectors
        self.wave_vectors = self._eval_wave_vectors()

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
            ref_t_matrices = self._calculate_reference_t_matrices(
                ref_vib_amps, site_elements
            )
        else:
            # use the stored reference t-matrices from reference calculation
            ref_t_matrices = self.ref_calc_result.t_matrices
        # convert to jnp array
        self.ref_t_matrices = jnp.asarray(ref_t_matrices)

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

        # set the propagator context
        self.propagator_context = PropagatorContext(
            is_dynamic_propagator=self._parameter_space.is_dynamic_propagator,
            propagator_id=self._parameter_space.propagator_id,
            kappa=self.kappa,
            static_propagators=self._static_propagators,
            propagator_transpose_int=self.propagator_transpose_int,
            symmetry_operations=self.propagator_symmetry_operations,
        )

        # set the t-matrix context
        self.t_matrix_context = TMatrixContext(
            energies=self.energies,
            static_t_matrices=self._static_t_matrices,
            dynamic_site_elements=self.parameter_space.dynamic_t_matrix_site_elements,
            t_matrix_id=self.parameter_space.t_matrix_id,
            is_dynamic_mask=self.parameter_space.is_dynamic_t_matrix,
        )

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
        self._static_propagators = jnp.einsum(
            'ed...->de...', static_propagators
        )

    def _calculate_reference_t_matrices(self, ref_vib_amps, site_elements):
        def map_fn(pair):
            vib_amp, site_el = pair
            return vib_dependent_tmatrix(
                self.max_l_max,
                self.phaseshifts[site_el][:, : self.max_l_max + 1],
                self.energies,
                vib_amp,
            )

        ref_t_matrices = jax.lax.map(
            map_fn, (ref_vib_amps, site_elements), batch_size=self.batch_atoms
        )
        return jnp.einsum('ael->eal', ref_t_matrices)

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
            self.wave_vectors()
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

    def _eval_wave_vectors(self):
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
        # split free parameters
        (_, vib_params, geo_params, occ_params) = (
            self.parameter_space.split_free_params(free_params)
        )

        # displacements, converted to atomic units
        displacements_ang = self.parameter_space.reference_displacements(
            geo_params
        )
        displacements_au = atomic_units.to_internal_displacement_vector(
            displacements_ang
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
            propagators = calculate_propagators(
                self.propagator_context,
                displacements_au,
                energy_ids,
                self.batch_energies,
                self.batch_atoms,
                self.max_l_max,
            )

            # crop propagators
            propagators = propagators[
                :, :, : (l_max + 1) ** 2, : (l_max + 1) ** 2
            ]

            # dynamic t-matrices
            t_matrices = calculate_t_matrices(
                self.t_matrix_context,
                l_max,
                self.batch_energies,
                self.phaseshifts,
                vib_amps_au,
                energy_ids)

            # crop t-matrices
            ref_t_matrices = self.ref_t_matrices[
                energy_ids, :, : l_max + 1
            ]
            t_matrices = t_matrices[:, :, : l_max + 1]

            # map t-matrices to compressed quantum index
            mapped_t_matrix_vib = vmapped_l_array_to_compressed_quantum_index(
                t_matrices, l_max)
            mapped_t_matrix_ref = vmapped_l_array_to_compressed_quantum_index(
                ref_t_matrices, l_max
            )

            energy_ids = jnp.arange(len(batch))  # smaller batch of energies
            l_delta_amps = batch_delta_amps(
                energy_ids,
                propagators,
                mapped_t_matrix_vib,
                mapped_t_matrix_ref,
                self.ref_calc_result.in_amps,
                self.ref_calc_result.out_amps,
                chem_weights,
                self.parameter_space.n_atom_basis,
                self.batch_atoms,
            )
            batched_delta_amps.append(l_delta_amps)

        # perform final processing and return
        return _recombine_delta_amps(
            batched_delta_amps,
            self.batching.restore_sorting,
            self.delta_amp_prefactors
        )

    def intensity(self, free_params):
        delta_amplitude = self.delta_amplitude(free_params)
        _, _, geo_params, _ = self.parameter_space.split_free_params(free_params)
        prefactors = intensity_prefactors(
            self.parameter_space.potential_onset_height_change(geo_params),
            self.n_beams,
            self.theta,
            self.wave_vectors,
        )

        return sum_intensity(
            prefactors, self.ref_calc_result.ref_amps, delta_amplitude
        )

    @property
    def reference_intensity(self):
        dummy_delta_amps = jnp.zeros(
            (
                len(self.energies),
                self.n_beams,
            ),
            dtype=jnp.complex128,
        )

        prefactors = intensity_prefactors(
            jnp.array([0.0]),
            self.n_beams,
            self.theta,
            self.wave_vectors,
        )
        return sum_intensity(
            prefactors,
            self.ref_calc_result.ref_amps,
            dummy_delta_amps,
        )

    @property
    def unperturbed_intensity(self):
        """Return intensity from reference data without pertubation."""
        raise NotImplementedError

    def interpolated(self, free_params, deriv_deg=0):
        """Return interpolated intensity and energy derivatives."""
        return _interpolate_intensity(
            self.intensity(free_params),
            self.origin_grid,
            self.target_grid,
            deriv_deg,
            self.bc_type,
        )

    def R(self, free_params):
        """Evaluate R-factor."""
        if self.comp_intensity is None:
            raise ValueError('Comparison intensity not set.')
        non_interpolated_intensity = self.intensity(free_params)

        v0r_param, *_ = self.parameter_space.split_free_params(free_params)
        v0r_shift = self.parameter_space.v0r_transformer(v0r_param)

        return calc_r_factor(
            non_interpolated_intensity,
            v0r_shift,
            self.ref_calc_params,
            self.rfactor_func,
            self.origin_grid,
            self.interpolation_step,
            self.target_grid,
            self.exp_spline,
        )

    def R_val_and_grad(self, free_params):
        """Evaluate R-factor and its gradients."""
        val, grad = jax.value_and_grad(self.R)(free_params)
        grad = jnp.asarray(grad)
        return val, grad

    def grad_R(self, free_params):
        """Evaluate R-factor gradients."""
        _, grad = self.R_val_and_grad(free_params)
        return grad


    def benchmark(self, free_params=None, n_repeats=10, csv_file_path=None,
                  use_grad=True):
        """Run benchmarks and add log results."""
        logger.info('Runnning timing benchmarks for tensor-LEED calculation...')
        bench_results = utils.benchmark_calculator(
            self, free_params, n_repeats, csv_file_path, use_grad
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


# @partial(jax.jit, static_argnames=['n_atom_basis', 'batch_atoms'])
def calc_energy(
    e_id,
    propagators,
    t_matrix_vib,
    t_matrix_ref,
    amps_in,
    amps_out,
    chem_weights,
    n_atom_basis,
    batch_atoms,
):
    def compute_atom_contrib(a):
        delta_t_matrix = calculate_delta_t_matrix(
            propagators[e_id, a].conj(),
            t_matrix_vib[e_id, a],
            t_matrix_ref[e_id, a],
            chem_weights[a],
        )
        return jnp.einsum(
            'bl,lk,k->b',
            amps_out[e_id, a],
            delta_t_matrix,
            amps_in[e_id, a],
            optimize='optimal',
        )

    contribs = jax.lax.map(
        compute_atom_contrib, jnp.arange(n_atom_basis), batch_size=batch_atoms
    )
    return jnp.sum(contribs, axis=0)


@partial(jax.jit, static_argnames=['batch_atoms', 'n_atom_basis'])
def batch_delta_amps(
    energy_ids,
    propagators,
    t_matrix_vib,
    t_matrix_ref,
    amps_in,
    amps_out,
    chem_weights,
    n_atom_basis,
    batch_atoms,
):
    @jax.checkpoint
    def calc_energy(e_id):
        def compute_atom_contrib(a):
            delta_t_matrix = calculate_delta_t_matrix(
                propagators[e_id, a].conj(),
                t_matrix_vib[e_id, a],
                t_matrix_ref[e_id, a],
                chem_weights[a],
            )
            return jnp.einsum(
                'bl,lk,k->b',
                amps_out[e_id, a],
                delta_t_matrix,
                amps_in[e_id, a],
                optimize='optimal',
            )

        contribs = jax.lax.map(
            compute_atom_contrib,
            jnp.arange(n_atom_basis),
            batch_size=batch_atoms,
        )
        return jnp.sum(contribs, axis=0)

    return jax.lax.map(calc_energy, energy_ids)

@partial(
    jax.jit,
    static_argnames=(
        'interpolation_step',
        'rfactor_func',
        #'ref_calc_params',
    ),
)
def calc_r_factor(
    non_interpolated_intensity,
    v0r_shift,
    ref_calc_params,
    rfactor_func,
    origin_grid,
    interpolation_step,
    target_grid,
    exp_spline,
):

    v0i_electron_volt = -ref_calc_params.v0i * HARTREE

    # apply v0r shift
    theo_spline = interpax.CubicSpline(
        origin_grid + v0r_shift,
        non_interpolated_intensity,
        check=False,
        extrapolate=False,
    )

    return rfactor_func(
        theo_spline,
        v0i_electron_volt,
        interpolation_step,
        target_grid,
        exp_spline,
    )

@jax.jit
def _recombine_delta_amps(energy_batched, sorting_order, prefactors):
    # combine to one array
    delta_amps = jnp.concatenate(energy_batched, axis=0)
    # re-sort to the original order
    delta_amps = delta_amps[sorting_order]
    # apply prefactors and return
    return delta_amps * prefactors

@partial(jax.jit, static_argnames=['deriv_deg', 'bc_type'])
def _interpolate_intensity(
    intensity,
    origin_grid,
    target_grid,
    deriv_deg,
    bc_type,
):
        spline = interpax.CubicSpline(
        origin_grid,
        intensity,
        bc_type=bc_type,
        extrapolate=False,
        check=False,  # TODO: do check once in the object creation
        )
        for _ in range(deriv_deg):
            spline = spline.derivative()
        return spline(target_grid)
