
import jax
import jax.numpy as jnp

from viperleed.calc import symmetry

from functools import partial
from src import delta
from src import rfactor
from src.constants import BOHR, HARTREE
from src.interpolation import *
from src.lib_intensity import intensity_prefactor, sum_intensity
from src.parameter_handler import TensorParameterTransformer

class TensorLEEDCalculator:

    def __init__(self, ref_data, phaseshifts, slab, rparams,
                 beam_indices,
                 interpolation_step=0.5,
                 interpolation_deg=3,
                 batch_lmax=False):
        self.ref_data = ref_data
        self.phaseshifts = phaseshifts
        self.batch_lmax = batch_lmax
        self.interpolation_deg = interpolation_deg
        self.beam_indices = jnp.array(beam_indices)
        # reading from IVBEAMS does not guarantee correct order!
        #self.beam_indices = jnp.array([beam.hk for beam in rparams.ivbeams])
        self.comp_intensity = None
        self.comp_energies = None
        self.interpolation_step = interpolation_step

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
        self.interpolator = StaticNotAKnotSplineInterpolator(
            ref_data.incident_energy_ev,
            self.target_grid,
            self.interpolation_deg # TODO: take from rparams.INTPOL_DEG
        )
        self.parameter_transformer = self._get_parameter_transformer(slab, rparams)


    @property
    def unit_cell_area(self):
        return jnp.linalg.norm(jnp.cross(self.unit_cell[:,0], self.unit_cell[:,1]))

    @property
    def reciprocal_unit_cell(self):
        return 2*jnp.pi*jnp.linalg.inv(self.unit_cell)

    @property
    def n_atoms(self):
        return len(self.ref_vibrational_amps)

    def _get_parameter_transformer(self, slab, rparams):
        # find and enforce symmetry on slab
        rparams.SYMMETRY_FIND_ORI = True
        slab.full_update(rparams)

        # find symmetry
        plane_group = symmetry.findSymmetry(slab, rparams)
        # enforce symmetry
        symmetry.enforceSymmetry(slab, rparams, plane_group)

        # make parameter transformer
        return TensorParameterTransformer(slab, self.interpolation_step)

    def set_experiment_intensity(self, comp_intensity, comp_energies):
        self.comp_intensity = comp_intensity
        self.comp_energies = comp_energies
        # set interpolator
        self.exp_interpolator = StaticNotAKnotSplineInterpolator(
            comp_energies,
            self.target_grid,
            self.interpolation_deg
        )

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

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def interpolated(self, vib_amps, displacements, deriv_deg=0):
        return self._interpolated(vib_amps, displacements, deriv_deg)

    def _interpolated(self, vib_amps, displacements, deriv_deg=0):
        non_interpolated_intensity = self._intensity(vib_amps, displacements)
        rhs = not_a_knot_rhs(non_interpolated_intensity)
        bspline_coeffs = get_bspline_coeffs(self.interpolator, rhs)
        return evaluate_spline(bspline_coeffs, self.interpolator, deriv_deg)

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def R_pendry(self, vib_amps, displacements, v0_real_steps=0):
        return self._R_pendry(vib_amps, displacements, v0_real_steps)

    def _R_pendry(self, vib_amps, displacements, v0_real_steps=0):
        if self.comp_intensity is None:
            raise ValueError("Comparison intensity not set.")
        v0i_electron_volt = -self.ref_data.v0i*HARTREE
        non_interpolated_intensity = self._intensity(vib_amps, displacements)
        return rfactor.pendry_R(
            non_interpolated_intensity,
            self.exp_interpolator,
            self.interpolator,
            v0_real_steps,
            v0i_electron_volt,
            self.interpolation_step,
            self.comp_intensity
        )

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def R_pendry_val_and_grad(self, vib_amps, displacements, v0_real_steps):
        # TODO: urgent: currently only gives gradients for geo displacements
        return jax.value_and_grad(self._R_pendry, argnums=(1))(vib_amps, displacements, v0_real_steps)

    def _R_pendry_from_reduced(self, reduced_params):
        v0r_step, vib_amps, displacements = self.parameter_transformer.unflatten_parameters(reduced_params)
        return self._R_pendry(vib_amps, displacements, v0r_step)

    def _R_pendry_grad_from_reduced(self, reduced_params):
        return jax.grad(self._R_pendry_from_reduced)(reduced_params)

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def R_pendry_from_reduced(self, reduced_params):
        return self._R_pendry_from_reduced(reduced_params)

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def R_pendry_grad_from_reduced(self, reduced_params):
        return self._R_pendry_grad_from_reduced(reduced_params)

    @partial(jax.jit, static_argnames=('self')) # TODO: not good, redo as pytree
    def R_pendry_val_and_grad_from_reduced(self, reduced_params):
        return jax.value_and_grad(self._R_pendry_from_reduced)(reduced_params)

    @property
    def zero_displacement():
        pass

    def _benchmark():
        pass

    # JAX PyTree methods

    def tree_flatten(self):
        pass

    def tree_unflatten():
        pass
