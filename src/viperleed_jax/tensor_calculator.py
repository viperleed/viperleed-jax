"""Module tensor_calculator."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@Paulhai7)')
__created__ = '2024-05-03'

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from interpax import CubicSpline
from viperleed.calc import LOGGER as logger
from viperleed.calc.files.iorfactor import beamlist_to_array

from viperleed_jax import rfactor, utils
from viperleed_jax.batching import Batching
from viperleed_jax.constants import BOHR, HARTREE
from viperleed_jax.dense_quantum_numbers import (
    vmapped_l_array_to_compressed_quantum_index,
)
from viperleed_jax.interpolation import interpolate_ragged_array
from viperleed_jax.lib import math
from viperleed_jax.lib.derived_quantities.normalized_occupations import (
    NormalizedOccupations,
)
from viperleed_jax.lib.derived_quantities.onset_height_change import (
    OnsetHeightChange,
)
from viperleed_jax.lib.derived_quantities.propagtor import Propagators
from viperleed_jax.lib.derived_quantities.t_matrix import TMatrix
from viperleed_jax.lib.tensor_leed.t_matrix import vib_dependent_tmatrix
from viperleed_jax.lib_intensity import intensity_prefactors, sum_intensity
from viperleed_jax.rfactor import R_FACTOR_SYNONYMS


class TensorLEEDCalculator:
    """Main class for calculating tensor LEED intensities and R-factors.

    Parameters
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
    ):
        self.ref_calc_params = ref_calc_params
        self.ref_calc_result = ref_calc_result
        self.phaseshifts = phaseshifts
        self.recalculate_ref_t_matrices = (
            rparams.VLJ_CONFIG['recalc_ref_t_matrices'])

        self.interpolation_deg = interpolation_deg
        self.bc_type = bc_type
        self.use_symmetry = rparams.VLJ_CONFIG['use_symmetry']

        self.occ_norm_method = rparams.VLJ_CONFIG['occ_norm']

        # get experimental intensities and hk
        if not rparams.expbeams:
            msg = (
                'No (pseudo)experimental beams loaded. This is required '
                'for the structure optimization.'
            )
            logger.error(msg)

        exp_energies, _, _, exp_intensities = beamlist_to_array(
            rparams.expbeams
        )
        exp_hk = [b.hk for b in rparams.expbeams]

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

        # set l_max
        self.max_l_max = ref_calc_params.max_lmax

        # TODO: refactor into a dataclass
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


        # determine and set batch sizes
        self._set_batch_sizes(rparams)

        # default R-factor is Pendry
        self.rfactor_func = rfactor.pendry_R

        if self.interpolation_deg != 3:
            raise NotImplementedError

        # calculate batching
        self.batching = Batching(self.energies, ref_calc_params.lmax)

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

    def _set_batch_sizes(self, rparams):
        """Set batch sizes for energies and atoms based on rparams."""
        # TODO: implement a memory-aware automatic batching
        batch_energies, batch_atoms = (
            rparams.VLJ_BATCH['energies'],
            rparams.VLJ_BATCH['atoms'],
        )
        if batch_atoms == -1:
            batch_atoms = self.n_atoms
        self.batch_atoms = batch_atoms

        if batch_energies == -1:
            batch_energies = self.energies.shape[0]
        self.batch_energies = batch_energies

        logger.debug(
            f'Using batch sizes: {self.batch_energies} energies, '
            f'{self.batch_atoms} atoms.'
        )

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

    def check_parameter_space_set(self):
        """Check whether the parameter space has been set.

        Raises
        ------
            ValueError: If the parameter space is not set.
        """
        if self._parameter_space is None:
            raise ValueError('Parameter space not set.')

    @property
    def parameter_space(self):
        """Return the parameter space."""
        self.check_parameter_space_set()
        return self._parameter_space

    def split_free_params(self, free_params):
        self.check_parameter_space_set()
        return self._split_free_params(free_params)

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
        self._parameter_space = parameter_space

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

        # set up the derived quantities
        self._setup_derived_quantities()

        # set split free parameters function
        self._split_free_params = jax.jit(
            self.parameter_space.split_free_params()
        )
        # v0r transformer - could be made into a derived quantity
        self._v0r_transformer = jax.jit(self.parameter_space.v0r_transformer())

        logger.info('Calculator initialized with parameter space.')
        logger.debug(
            'This parameter space requires dynamic calculation of '
            f'{self.calc_t_matrices.n_dynamic_t_matrices} t-matrix(-ces) and '
            f'{self.calc_propagators.n_dynamic_values} propagator(s).'
        )

    def _setup_derived_quantities(self):
        """Set up derived quantities for the calculator."""
        self.check_parameter_space_set()

        # onset height of the inner potential
        self.calc_onset_height_change = OnsetHeightChange(self.parameter_space)

        # normalized occupations (i.e. chemical weights)
        self.calc_normalized_occupations = NormalizedOccupations(
            self.parameter_space,
            self.atom_ids.tolist(),
            op_type= self.occ_norm_method,
        )

        # atomic t-matrices (will calculate static t-matrices during init)
        self.calc_t_matrices = TMatrix(
            self.parameter_space,
            self.energies,
            self.phaseshifts,
            self.batch_energies,
            self.max_l_max,
        )

        # propagators (will calculate static propagators during init)
        self.calc_propagators = Propagators(
            self.parameter_space,
            self.kappa,
            self.energies,
            self.batch_energies,
            self.batch_atoms,
            self.max_l_max,
            use_symmetry=self.use_symmetry,
        )

    def _calculate_reference_t_matrices(self, ref_vib_amps, site_elements):
        ref_t_matrices = []
        for site_el, vib_amp in zip(site_elements, ref_vib_amps):
            batched_t_matrix = jax.vmap(
                vib_dependent_tmatrix, in_axes=(None, 0, 0, None)
            )
            ref_t_matrices.append(
                batched_t_matrix(
                    self.max_l_max,
                    self.phaseshifts[site_el][:, : self.max_l_max + 1],
                    self.energies,
                    vib_amp,
                )
            )
        ref_t_matrices = jnp.array(ref_t_matrices)
        return jnp.einsum('ael->eal', ref_t_matrices)

    def _calc_delta_amp_prefactors(self):
        energies = self.energies
        v_imag = self.ref_calc_params.v0i

        # energy dependent quantities
        out_k_par2 = self.ref_calc_params.kx_in
        out_k_par3 = self.ref_calc_params.ky_in

        k_inside = jnp.sqrt(2 * energies - 2j * v_imag + 1j * math.EPS)

        # Propagator evaluated relative to the muffin tin zero i.e.
        # it uses energy = incident electron energy + inner potential
        out_k_par = out_k_par2**2 + out_k_par3**2
        out_k_perp_inside = jnp.sqrt(
            ((2 * energies - 2j * v_imag)[:, jnp.newaxis] - out_k_par)
            + 1j * math.EPS
        )

        # Prefactors from Equation (41) from Rous, Pendry 1989
        return jnp.einsum(
            'e,eb,->eb',
            1 / k_inside,
            1 / out_k_perp_inside,
            1 / (2 * (self.unit_cell_area)),
        )

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

    def expand_params(self, free_params):
        _free_params = np.asarray(free_params)
        v0r_params, geo_params, vib_params, occ_params = self.split_free_params(
            _free_params
        )
        v0r_shift = self.parameter_space.v0r_transformer()(v0r_params)
        vib_amps = self.parameter_space.vib_tree(vib_params)
        displacements = self.parameter_space.geo_tree(geo_params)
        occupations = self.calc_normalized_occupations(occ_params)
        return v0r_shift, displacements, vib_amps, occupations

    def delta_amplitude(self, free_params):
        """Calculate the delta amplitude for a given set of free parameters."""
        self.check_parameter_space_set()
        _free_params = jnp.asarray(free_params)
        # split free parameters
        (_, geo_params, vib_params, occ_params) = self._split_free_params(
            _free_params
        )

        # chemical weights
        chem_weights = self.calc_normalized_occupations(occ_params)

        # Loop over batches
        # -----------------

        # Use python for loop here, as batches can have different array sizes

        batched_delta_amps = []
        for batch in self.batching.batches:
            l_max = batch.l_max
            energy_ids = jnp.asarray(batch.energy_indices)

            # propagators - already rotated
            propagators = self.calc_propagators(
                geo_params,
                energy_ids,
            )

            # crop propagators
            propagators = propagators[
                :, :, : (l_max + 1) ** 2, : (l_max + 1) ** 2
            ]

            # # dynamic t-matrices
            t_matrices = self.calc_t_matrices(
                vib_params,
                l_max,
                energy_ids,
            )

            # crop t-matrices
            ref_t_matrices = self.ref_t_matrices[energy_ids, :, : l_max + 1]
            t_matrices = t_matrices[:, :, : l_max + 1]

            # map t-matrices to compressed quantum index
            mapped_t_matrix_vib = vmapped_l_array_to_compressed_quantum_index(
                t_matrices, l_max
            )
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
            self.delta_amp_prefactors,
        )

    def intensity(self, free_params):
        _free_params = jnp.asarray(free_params)
        delta_amplitude = self.delta_amplitude(_free_params)
        _, geo_params, _, _ = self._split_free_params(_free_params)
        prefactors = intensity_prefactors(
            self.calc_onset_height_change(geo_params),
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
        _free_params = jnp.asarray(free_params)
        non_interpolated_intensity = self.intensity(_free_params)

        v0r_param, *_ = self._split_free_params(_free_params)
        v0r_shift = self._v0r_transformer(v0r_param)

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

    def benchmark(
        self, free_params=None, n_repeats=10, csv_file_path=None, use_grad=True
    ):
        """Run benchmarks and add log results."""
        logger.info('Runnning timing benchmarks for tensor-LEED calculation...')
        bench_results = utils.benchmark_calculator(
            self, free_params, n_repeats, csv_file_path, use_grad
        )
        logger.info(utils.format_benchmark_results(bench_results) + '\n')

    # TODO: needs tests
    def apply_to_slab(self, slab, rpars, free_params):
        # atom basis from parameter space
        atom_basis = self.parameter_space.atom_basis
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

        # expand the reduced parameter vector
        v0r, displacements, vibrations, occupations = self.expand_params(
            free_params)

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
                s for s in atom_basis.scatterers if s.num == at.num
            ]
            scatterer_indices = [
                atom_basis.scatterers.index(s) for s in at_scatterers
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
            for element in site.vibamp:
                siteel_scatterers = [
                    s
                    for s in atom_basis.scatterers
                    if s.site == site.label and s.element == element
                ]
                if len(siteel_scatterers) == 0:
                    continue

                scatterer_indices = [
                    atom_basis.scatterers.index(s)
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

        # # Optionally write to file
        # if write_to_file:
        #     # write POSCAR
        #     poscar.write(slab, 'POSCAR_TL_optimized', comments='all')
        #     # write VIBROCC
        #     writeVIBROCC(slab, rpars, 'VIBROCC_TL_optimized')


def calculate_delta_t_matrix(
    propagator, t_matrix_vib, t_matrix_ref, chem_weight
):
    # delta_t_matrix is the change of the atomic t-matrix with new
    # vibrational amplitudes and after applying the displacement
    # Equation (33) in Rous, Pendry 1989
    delta_t_matrix = jnp.dot(propagator.T * (1j * t_matrix_vib), propagator.T)
    delta_t_matrix = delta_t_matrix - jnp.diag(1j * t_matrix_ref)
    return delta_t_matrix * chem_weight


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
    theo_spline = CubicSpline(
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
    spline = CubicSpline(
        origin_grid,
        intensity,
        bc_type=bc_type,
        extrapolate=False,
        check=False,  # TODO: do check once in the object creation
    )
    for _ in range(deriv_deg):
        spline = spline.derivative()
    return spline(target_grid)
