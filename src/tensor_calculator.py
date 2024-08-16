
import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

from viperleed.calc import symmetry
from viperleed.calc import LOGGER as logger

from functools import partial
from src import delta
from src import rfactor
from src.constants import BOHR, HARTREE
from src.interpolation import *
from src.lib_intensity import intensity_prefactor, sum_intensity
from src.parameter_handler import TensorParameterTransformer

import interpax

_R_FACTOR_SYNONYMS = {
    rfactor.pendry_R: ('pendry', 'r_p', 'rp', 'pendry r-factor'),
    rfactor.R_1: ('r1', 'r_1', 'r1 factor'),
    rfactor.R_2: ('r2', 'r_2', 'r2 factor'),
    rfactor.R_ms: ('ms', 'msr', 'rms', 'r_ms', 'r_ms factor'),
    rfactor.R_zj: ('zj', 'zj factor', 'zannazi', 'zannazi jona', 'zannazi-jona'),
}


@register_pytree_node_class
class TensorLEEDCalculator:
    """
    Main class for calculating tensor LEED intensities and R-factors.
    
    The experimental intensities are not required during initialization, but
    they are needed to calculate R-factors.

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
    beam_indices : list
        The indices of the beams used in the LEED calculations.
    interpolation_step : float, optional
        The step size for interpolation, by default 0.5.
    interpolation_deg : int, optional
        The degree of interpolation, by default 3.
    bc_type : str, optional
        The boundary condition type for interpolation, by default 'not-a-knot'.
    batch_lmax : bool, optional
        Whether to use batched calculation for lmax, by default False.
    """

    def __init__(self, ref_data, phaseshifts, slab, rparams,
                 beam_indices,
                 interpolation_step=0.5,
                 interpolation_deg=3,
                 bc_type='not-a-knot',
                 batch_lmax=False):
        self.ref_data = ref_data
        self.phaseshifts = phaseshifts
        self.batch_lmax = batch_lmax
        self.interpolation_deg = interpolation_deg
        self.bc_type=bc_type
        self.beam_indices = jnp.array(beam_indices)
        # reading from IVBEAMS does not guarantee correct order!
        #self.beam_indices = jnp.array([beam.hk for beam in rparams.ivbeams])
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

        non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
        # TODO check this
        self.is_surface_atom = jnp.array([at.layer.num == 0 for at in non_bulk_atoms])

        self.ref_vibrational_amps = jnp.array(
            [at.site.vibamp[at.el] for at in non_bulk_atoms])
        self.origin_grid = ref_data.incident_energy_ev

        self.exp_spline = None
        self.parameter_transformer = self._get_parameter_transformer(slab, rparams)

        # default R-factor is Pendry
        self.rfactor_func = rfactor.pendry_R

        if self.interpolation_deg != 3:
            raise NotImplementedError

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
    def parameter_space(self):
        if self._parameter_space is None:
            raise ValueError("Parameter space not set.")
        return self._parameter_space

    @property
    def n_free_parameters(self):
        return self.parameter_space.n_free_parameters

    def set_rfactor(self, rfactor_name):
        _rfactor_name = rfactor_name.lower().strip()
        for func, synonyms in _R_FACTOR_SYNONYMS.items():
            if _rfactor_name in synonyms:
                self.rfactor_func = func
                # TODO: log rfactor change
                return
        raise ValueError(f"Unknown R-factor name: {rfactor_name}")

    def set_experiment_intensity(self, comp_intensity, comp_energies):
        self.comp_intensity = comp_intensity
        self.comp_energies = comp_energies
        self.exp_spline = interpolate_ragged_array(
            self.comp_energies,
            self.comp_intensity,
            bc_type=self.bc_type,
        )

    def set_parameter_space(self, delta_slab):
        if self._parameter_space is not None:
            logger.debug("Overwriting parameter space.")
        # take delta_slab and set the parameter space
        self._parameter_space = delta_slab.freeze()
        logger.info("Parameter space set:\n"
                    f"{delta_slab.parameter_space.info}")
        logger.info(
            "This parameter space requires dynamic calculation of "
            f"{self._parameter_space.n_dynamic_t_matrices} t-matrices and "
            f"{self._parameter_space.n_dynamic_propagators} propagators."
        )
        # pre-calculate the static t-matrices
        # TODO

        # pre-calculate the static propagators
        # TODO


    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def delta_amplitude(self, vib_amps, displacements):
        """TODO: docstring"""
        return self._delta_amplitude(vib_amps, displacements)

    def _delta_amplitude(self, vib_amps, displacements):
        """Internal non-jitted version of delta_amplitude."""
        return delta.delta_amplitude(vib_amps, displacements,
                              ref_data=self.ref_data,
                              unit_cell_area=self.unit_cell_area,
                              phaseshifts=self.phaseshifts,
                              batch_lmax=self.batch_lmax
                              )

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def intensity(self, vib_amps, displacements):
        return self._intensity(vib_amps, displacements)

    def delta_amplitude_from_reduced(self, reduced_params):
        _, vib_amps, displacements = self.parameter_transformer.unflatten_parameters(reduced_params)
        return self._delta_amplitude(vib_amps, displacements)

    def intensity_from_reduced(self, reduced_params):
        _, vib_amps, displacements = self.parameter_transformer.unflatten_parameters(reduced_params)
        return self._intensity(vib_amps, displacements)

    def _intensity(self, vib_amps, displacements):
        delta_amps = self._delta_amplitude(vib_amps, displacements)
        refraction_prefactor = self._intensity_prefactor(displacements)
        return sum_intensity(refraction_prefactor, self.ref_data.ref_amps,
                             delta_amps)

    def _intensity_prefactor(self, displacements):
        return intensity_prefactor(
            displacements,
            self.ref_data, self.beam_indices, self.theta, self.phi,
            self.reciprocal_unit_cell, self.is_surface_atom)

    @property
    def unperturbed_intensity(self):
        """Return intensity from reference data without pertubation."""
        zero_deltas = jnp.zeros(shape=(self.ref_data.n_energies,
                                       self.ref_data.n_beams))
        refraction_prefactor = self._intensity_prefactor(
            jnp.array([[0.0, 0.0, 0.0],]*self.n_atoms))
        return sum_intensity(refraction_prefactor, self.ref_data.ref_amps,
                             zero_deltas)

    @partial(jax.jit, static_argnames=('self', 'deriv_deg'))
    def interpolated(self, vib_amps, displacements, deriv_deg=0):
        return self._interpolated(vib_amps, displacements, deriv_deg)

    def _interpolated(self, vib_amps, displacements, deriv_deg=0):
        non_interpolated_intensity = self._intensity(vib_amps, displacements)
        spline =  interpax.CubicSpline(self.origin_grid,
                                    non_interpolated_intensity,
                                    bc_type=self.bc_type,
                                    extrapolate=False,
                                    check=False,                                # TODO: do check once in the object creation
                                    )
        for i in range(deriv_deg):
            spline = spline.derivative()
        return spline(self.target_grid)

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def R(self, vib_amps, displacements, v0r_shift=0.):
        return self._R(vib_amps, displacements, v0r_shift)

    def _R(self, vib_amps, displacements, v0r_shift=0.):
        if self.comp_intensity is None:
            raise ValueError("Comparison intensity not set.")
        v0i_electron_volt = -self.ref_data.v0i*HARTREE
        non_interpolated_intensity = self._intensity(vib_amps, displacements)
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

    @jax.jit
    def R_val_and_grad(self, vib_amps, displacements, v0_real_steps):
        # TODO: urgent: currently only gives gradients for geo displacements
        return jax.value_and_grad(self._R, argnums=(1))(vib_amps, displacements, v0_real_steps)

    @jax.jit
    def R_from_reduced(self, reduced_params):
        return self._R_from_reduced(reduced_params)

    def _R_from_reduced(self, reduced_params):
        v0r_step, vib_amps, displacements = self.parameter_transformer.unflatten_parameters(reduced_params)
        return self._R(vib_amps, displacements, v0r_step)

    @jax.jit
    def R_grad_from_reduced(self, reduced_params):
        return self._R_grad_from_reduced(reduced_params)

    def _R_grad_from_reduced(self, reduced_params):
        return jax.grad(self._R_from_reduced)(reduced_params)

    @jax.jit
    def R_val_and_grad_from_reduced(self, reduced_params):
        return jax.value_and_grad(self._R_from_reduced)(reduced_params)

    def _benchmark():
        pass

    # JAX PyTree methods

    def tree_flatten(self):
        dynamic_elements = {
            'rfactor_name': _R_FACTOR_SYNONYMS[self.rfactor_func][0]
        }
        simple_elements = {
            'ref_data': self.ref_data,
            'phaseshifts': self.phaseshifts,
            'batch_lmax': self.batch_lmax,
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
            'parameter_transformer': self.parameter_transformer,
            'exp_spline': self.exp_spline,
            'comp_intensity': self.comp_intensity,
            'comp_energies': self.comp_energies,
            'origin_grid': self.origin_grid,
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


def make_1d_ragged_cubic_spline(x, y, axis=0, bc_type="not-a-knot", extrapolate=False):
    """Construct a piecewise cubic spline interpolator with ragged edges.

    The interpolator uses a cubic spline to interpolate data.
    """
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError("x and y must be 1-dimensional arrays.")
    y_mask = jnp.isnan(y)
    x_subarray, y_subarray = x[~y_mask], y[~y_mask]
    start_index = jnp.where(~y_mask)[0][0]
    subarray_spline = interpax.CubicSpline(x_subarray, y_subarray, axis, bc_type, extrapolate, check=False)

    return subarray_spline, start_index

def interpolate_ragged_array(x, y, axis=0, bc_type="not-a-knot", extrapolate=False):
    all_coeffs = jnp.full((4, y.shape[0], y.shape[1]), fill_value=jnp.nan)
    for dim in range(y.shape[1]):
        spline, start_id = make_1d_ragged_cubic_spline(x, y[:, dim], axis=0, bc_type=bc_type, extrapolate=None)
        all_coeffs = all_coeffs.at[:, start_id:start_id+spline.c.shape[1], dim].set(spline.c)
    spline = interpax.PPoly.construct_fast(all_coeffs, x)
    return spline
