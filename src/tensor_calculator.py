
import jax
import jax.numpy as jnp


from functools import partial
from src.rfactor import *
from src.lib_intensity import *
from src.interpolation import *
from constants import BOHR, HARTREE
from src.lib_delta import delta_amplitude

class TensorLEEDCalculator:

    def __init__(self, ref_data, phaseshifts, slab, rparams,
                 interpolation_step=0.5,
                 interpolation_deg=3):
        self.ref_data = ref_data
        self.phaseshifts = phaseshifts
        self.beam_indices = jnp.array([beam.hk for beam in rparams.ivbeams])
        self.comp_intensity = None

        self.target_grid = jnp.arange(rparams.THEO_ENERGIES.start,
                                      rparams.THEO_ENERGIES.stop,
                                      interpolation_step)

        unit_cell_area = jnp.linalg.norm(jnp.cross(slab.ab_cell[:,0], slab.ab_cell[:,1]))
        # In Bohr radii
        self.unit_cell_area = unit_cell_area / BOHR**2

        # theta and phi
        self.theta = rparams.THETA
        self.phi = rparams.PHI

        non_bulk_atoms = [at for at in slab.atlist if not at.is_bulk]
        # TODO check this
        self.is_surface_atom = np.array([at.layer.num == 0 for at in non_bulk_atoms])

        self.interpolator = StaticNotAKnotSplineInterpolator(
            ref_data.incident_energy_ev,
            self.target_grid,
            interpolation_deg # TODO: take from rparams.INTPOL_DEG
        )

    def set_experiment_intesity(self, comp_intensity):
        self.comp_intensity = comp_intensity

    @property
    def delta_amp_func(self):
        return partial(delta_amplitude,
                       ref_data=self.ref_data,
                       phaseshifts=self.phaseshifts,
                       unit_cell_area=self.unit_cell_area)

    @property
    def intensity_func(self):
        return partial(_calc_intensity, ref_data=self.ref_data, phaseshifts=self.phaseshifts,
                       unit_cell_area=self.unit_cell_area, theta=self.theta, phi=self.phi,
                       beam_indices=self.beam_indices, is_surface_atom=self.is_surface_atom)


    # not absoulutely necessary, but could be useful
    @property
    def interpolation_func(self):
        return partial(_interpolated_intensity_and_deriv, interpolator=self.interpolator)


    @property
    def pendry_R_vs_reference_func(self):
        if self.comp_intensity is None:
            raise ValueError("Comparison intensity not set.")
        v0i_electron_volt = -self.ref_data.v0i*HARTREE
        e_step = 0.5 #TODO: dynamic!!
        v0_real = 3.0
        return partial(pendry_R,
                       interpolator_1=self.interpolator,
                       interpolator_2=self.interpolator,
                       v0_real=v0_real,
                       v0_imag=v0i_electron_volt,
                       energy_step=e_step,
                       intensity_1=self.comp_intensity)

    @property
    def zero_displacement():
        pass

    def _benchmark():
        pass

def _calc_intensity(vib_amps, displacements, ref_data, phaseshifts, unit_cell_area, theta, phi, beam_indices, is_surface_atom):
    delta_amps = delta_amplitude(vib_amps, displacements, ref_data, unit_cell_area, phaseshifts)
    refraction_prefactor = intensity_prefactor(displacements, ref_data, beam_indices, theta, phi, unit_cell_area, is_surface_atom)
    return sum_intensity(refraction_prefactor, ref_data.ref_amps, delta_amps)

def _interpolated_intensity_and_deriv(raw_intensity, interpolator):
    rhs = not_a_knot_rhs(raw_intensity)
    bspline_coeffs = get_bspline_coeffs(interpolator, rhs)
    interpolated_intensity = evaluate_spline(bspline_coeffs, interpolator, 0)
    interpolated_deriv = evaluate_spline(bspline_coeffs, interpolator, 1)
    return interpolated_intensity, interpolated_deriv
