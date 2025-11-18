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
from viperleed.calc.files.iorfactor import (
    beamlist_to_array,
    prepare_rfactor_energy_ranges,
)
from viperleed.calc.lib import leedbase

from viperleed_jax import rfactor, utils
from viperleed_jax.batching import Batching
from viperleed_jax.constants import BOHR, HARTREE
from viperleed_jax.dense_quantum_numbers import (
    vmapped_l_array_to_compressed_quantum_index,
)
from viperleed_jax.interpolation import interpolate_ragged_array
from viperleed_jax.lib import math
from viperleed_jax.lib.calculator import map_indices
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
        interpolation_deg=None,
        recalculate_ref_t_matrices=None,
        bc_type='not-a-knot',
    ):
        self.ref_calc_params = ref_calc_params
        self.ref_calc_result = ref_calc_result
        self.phaseshifts = phaseshifts

        if recalculate_ref_t_matrices is None:
            self.recalculate_ref_t_matrices = rparams.VLJ_CONFIG[
                'recalc_ref_t_matrices'
            ]
        else:
            logger.debug(
                'Overriding recalculate_ref_t_matrices to '
                f'{recalculate_ref_t_matrices}'
            )
            self.recalculate_ref_t_matrices = recalculate_ref_t_matrices

        self.interpolation_deg = rparams.INTPOL_DEG
        if interpolation_deg is not None:
            logger.debug(
                f'Overriding interpolation degree to {interpolation_deg}'
            )
            self.interpolation_deg = interpolation_deg
        self.bc_type = bc_type
        self.use_symmetry = rparams.VLJ_CONFIG['use_symmetry']

        # get beam correspondence
        # note that beams with no experimental correspondence are assigned -1
        # this is handled in average_beam_array()
        # Note also that the first call of getBeamCorrespondence removes
        # experimental beams that have no theoretical counterpart.
        # It is therefore important that this is called before loading the
        # experimental beams below.
        beam_correspondence = leedbase.getBeamCorrespondence(slab, rparams)
        self.beam_correspondence = tuple(beam_correspondence)
        logger.debug(f'Beam correspondence: {self.beam_correspondence}')

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

        # get V0r shift steps
        _rparams_copy = copy.deepcopy(rparams)
        _, _, iv_shift_range, _ = prepare_rfactor_energy_ranges(_rparams_copy)
        self._v0r_shift_steps = np.arange(
            iv_shift_range.start,
            iv_shift_range.stop + iv_shift_range.step,
            iv_shift_range.step,
        )

        # set up atom numbers
        self._atom_nums = np.array(
            [at.num for at in slab.atlist if not at.is_bulk]
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

        self.origin_grid = ref_calc_params.incident_energy_ev

        self.delta_amp_prefactors = self._calc_delta_amp_prefactors()

        self.exp_spline = None

        self._requested_batch_energies = rparams.VLJ_BATCH['energies']
        self._requested_batch_atoms = rparams.VLJ_BATCH['atoms']

        # default R-factor is Pendry
        self.rfactor_func = rfactor.pendry_R

        if self.interpolation_deg != 3:
            msg = (
                'Only cubic interpolation (degree 3) is currently supported. '
                'This is a limitation of the underlying interpolation '
                'library.'
            )
            raise NotImplementedError(msg)

        # calculate batching
        self.batching = Batching(self.energies, ref_calc_params.lmax)

        self.set_experiment_intensity(exp_intensities, exp_energies)

        self.kappa = jnp.array(self.ref_calc_params.kappa)

        # evaluate the wave vectors
        self.wave_vectors = self._eval_wave_vectors()

    def _set_batch_sizes(self):
        """Set batch sizes for energies and atoms based on rparams."""
        # TODO: implement a memory-aware automatic batching
        batch_energies, batch_atoms = (
            self._requested_batch_energies,
            self._requested_batch_atoms,
        )
        if batch_atoms == -1:
            batch_atoms = len(self.parameter_space.atom_basis.scatterers)
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
        self.check_parameter_space_set()
        return self._atom_ids

    @property
    def scatterer_to_atom_map(self):
        """Map scatterer IDs to atom IDs."""
        # TODO: move to parameter space or AtomBasis
        self.check_parameter_space_set()
        return map_indices(self.atom_ids, self._atom_nums)

    @property
    def n_atoms(self):
        """Return the number of atoms in the parameter space.

        This is (usually) different from the number of scatterers!
        """
        return np.unique(np.array(self.atom_ids)).size

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

        # set the atom IDs from the parameter space
        self._atom_ids = self.parameter_space.atom_basis.atom_ids

        # determine and set batch sizes
        # (needs to be done here, since we need atom info from parameter space)
        self._set_batch_sizes()

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

        logger.info(
            '\nParameter space set\n-----------------------\n'
            f'{self.parameter_space.info}'
        )

    def _setup_derived_quantities(self):
        """Set up derived quantities for the calculator."""
        self.check_parameter_space_set()

        # onset height of the inner potential
        self.calc_onset_height_change = OnsetHeightChange(self.parameter_space)

        # normalized occupations (i.e. chemical weights)
        self.calc_normalized_occupations = NormalizedOccupations(
            self.parameter_space,
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
        """Calculate the delta amplitude for a given set of free parameters.

        Note on chemical weights
        ------------------------
        The effect of chemical perturbations (i.e. substitutions) is calculated
        by simply multiplying the t-matrix with the (always positive) occupation
        weights in the calculate_delta_t_matrix() function.
        This seems weird at first glance, as one may expect that species species
        that have decreased occupations in respect to the reference
        calculation should have their contributions subtracted.
        However, this is not the case in the mixed t-matrix formalism, and is
        shown e.g. in the TensErLEED paper by Blum and Heinz eqs. (28-31).
        """
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
                self.scatterer_to_atom_map,
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

    def R(self, free_params, per_beam=False, **kwargs):
        """Evaluate R-factor."""
        if self.comp_intensity is None:
            raise ValueError('Comparison intensity not set.')
        _free_params = jnp.asarray(free_params)

        v0r_param, *_ = self._split_free_params(_free_params)
        v0r_shift = self._v0r_transformer(v0r_param)

        # calculate the non-interpolated intensity
        non_interpolated_intensity = self.intensity(_free_params)

        # average the intensities according to the beam correspondence
        non_interpolated_intensity = average_beam_array(
            non_interpolated_intensity, self.beam_correspondence
        )

        return calc_r_factor(
            non_interpolated_intensity,
            v0r_shift,
            self.ref_calc_params,
            self.rfactor_func,
            self.origin_grid,
            self.interpolation_step,
            self.target_grid,
            self.exp_spline,
            per_beam=per_beam,
            **kwargs,
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
            free_params
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

            averaged_displacement = (
                at_displacements * at_occupations[:, np.newaxis]
            )
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
                    atom_basis.scatterers.index(s) for s in siteel_scatterers
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


def evaluate_perturbed_t_matrix(propagator, vib_t_matrix):
    """Evaluate the perturbed t-matrix for a given propagator and vibrational t-matrix."""
    # Equation (33) in Rous, Pendry 1989
    perturbed_t_matrix = jnp.dot(
        propagator.T * (1j * vib_t_matrix), propagator.T
    )
    return perturbed_t_matrix


def average_perturbed_t_matrices(
    perturbed_t_matrices, scatterer_to_atom_map, chem_weights, n_atoms
):
    """Average the perturbed t-matrices over the atom basis."""
    _scatterer_to_atom_map = np.asarray(scatterer_to_atom_map)
    n_atom_basis = _scatterer_to_atom_map.size
    if n_atom_basis != chem_weights.size:
        raise ValueError('atom_ids and chem_weights must have the same size.')
    elif perturbed_t_matrices.shape[0] != n_atom_basis:
        raise ValueError(
            'perturbed_t_matrices must have the same number of rows as atom_ids.'
        )
    # multiply each perturbed t-matrix with the corresponding chemical weight
    weighted_t_matrices = (
        perturbed_t_matrices * chem_weights[:, jnp.newaxis, jnp.newaxis]
    )
    # sum over the atom basis
    averaged_t_matrix = jax.ops.segment_sum(
        data=weighted_t_matrices,
        segment_ids=_scatterer_to_atom_map,
        num_segments=n_atoms,
    )
    return averaged_t_matrix


def evaluate_delta_t_matrix(averaged_t_matrix, t_matrix_ref):
    """Evaluate the delta t-matrix."""
    # Equation (33) in Rous, Pendry 1989
    delta_t_matrix = averaged_t_matrix - jnp.diag(1j * t_matrix_ref)
    return delta_t_matrix


@partial(jax.jit, static_argnames=['batch_atoms', 'scatterer_to_atom_map'])
def batch_delta_amps(
    energy_ids,
    propagators,
    t_matrix_vib,
    t_matrix_ref,
    amps_in,
    amps_out,
    chem_weights,
    scatterer_to_atom_map,
    batch_atoms,
):
    # the number of scatterers != number of atoms (shared occupations)
    n_scatterers = t_matrix_vib.shape[1]
    n_atoms = t_matrix_ref.shape[1]

    # @jax.checkpoint
    def calc_energy(e_id):
        def compute_perturbed_t_matrices(a):
            # get the propagator for the current atom
            propagator = propagators[e_id, a].conj()
            # get the vibrational t-matrix for the current atom
            t_matrix_vib_a = t_matrix_vib[e_id, a]
            # calculate the perturbed t-matrix
            return evaluate_perturbed_t_matrix(propagator, t_matrix_vib_a)

        perturbed_t_matrices = jax.lax.map(
            compute_perturbed_t_matrices,
            jnp.arange(n_scatterers),
            batch_size=batch_atoms,
        )

        # average the perturbed t-matrices over the atom basis
        averaged_t_matrix = average_perturbed_t_matrices(
            perturbed_t_matrices,
            scatterer_to_atom_map,
            chem_weights,
            n_atoms,
        )

        def compute_delta_t_matrix(a):
            # get the perturbed t-matrix for the current atom
            perturbed_t_matrix = averaged_t_matrix[a]
            # get the reference t-matrix for the current atom
            t_matrix_ref_a = t_matrix_ref[e_id, a]
            # calculate the delta t-matrix
            return evaluate_delta_t_matrix(perturbed_t_matrix, t_matrix_ref_a)

        delta_t_matrix = jax.lax.map(
            compute_delta_t_matrix,
            jnp.arange(n_atoms),
            batch_size=batch_atoms,
        )

        def compute_atom_contrib(a):
            # get the propagator for the current atom
            return jnp.einsum(
                'bl,lk,k->b',
                amps_out[e_id, a],
                delta_t_matrix[a],
                amps_in[e_id, a],
                optimize='optimal',
            )

        beam_contribs = jax.lax.map(
            compute_atom_contrib,
            jnp.arange(n_atoms),
            batch_size=batch_atoms,
        )
        return jnp.sum(beam_contribs, axis=0)

    # batch over energies?
    return jax.lax.map(calc_energy, energy_ids)


@partial(
    jax.jit,
    static_argnames=(
        'interpolation_step',
        'rfactor_func',
        'per_beam',
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
    per_beam=False,
    **kwargs,
):
    v0i_electron_volt = -ref_calc_params.v0i * HARTREE

    # apply v0r shift
    theo_spline = CubicSpline(
        # V0r (potential) offset gets applied here to the origin grid
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
        per_beam=per_beam,
        **kwargs,
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


@partial(jax.jit, static_argnames=['beam_correspondence'])
def average_beam_array(beam_array, beam_correspondence):
    """Average the beam array over the beam correspondence.

    Parameters
    ----------
    beam_array : array_like
        The beam array to average, shape (n_energies, n_beams).
    beam_correspondence : tuple
        A tuple containing the beam correspondence, which maps the
        experimental beams to the theoretical beams. It should be a 1D array
        of integers with shape (n_beams,).

    Returns
    -------
    array_like
        The averaged beam array, shape (n_energies, n_averaged_beams).

    Raises
    ------
    ValueError
        If the number of beams in the beam array does not match the length of
        the beam correspondence.
    """
    # convert beam correspondence to numpy array
    beam_corr = np.array(beam_correspondence, dtype=np.int32)

    # check if beam_array and beam_corr have the same number of beams
    if beam_array.shape[1] != beam_corr.shape[0]:
        raise ValueError(
            'Beam array and beam correspondence must have the same number of beams.'
        )

    # beam_corr may contain -1 for beams that have no correspondence; these
    # need to be filtered out
    valid_beam_mask = beam_corr != -1
    beam_corr = beam_corr[valid_beam_mask]
    filtered_beam_array = beam_array[:, valid_beam_mask]
    # determine the number of averaged beams after filtering
    n_averaged_beams = np.unique(beam_corr).size

    # get weights for averaging
    ones = jnp.ones_like(beam_corr, dtype=float)
    summed = jax.ops.segment_sum(ones, beam_corr, num_segments=n_averaged_beams)
    averaged_beam_weights = jnp.reciprocal(summed)

    # sum beams according to the beam correspondence
    mix_beams_vmap = jax.vmap(jax.ops.segment_sum, in_axes=(0, None, None))
    averaged = mix_beams_vmap(filtered_beam_array, beam_corr, n_averaged_beams)

    # apply the averaged weights
    return averaged * averaged_beam_weights[jnp.newaxis, :]
