"""Module atom_units.

We use Angstrom and eV as inputs, but use atomic units internally."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-09-05'

import jax
import jax.numpy as jnp
import numpy as np

from viperleed_jax.constants import BOHR, HARTREE


@jax.jit
def to_internal_displacement_vector(displacement_vector_ang):
    """Convert from Angstrom to atomic units and left handed coordinate system.

    Internal representation of displacement vector is in Bohr and left handed
    coordinate system.

    Parameters
    ----------
    displacement_vector : array_like
        Displacement vector in Angstrom.

    Returns
    -------
    jnp.array
        Displacement vector in Bohr and left handed coordinate system.
    """
    # make sure it's the correct shape vector
    vector = jnp.array(displacement_vector_ang).reshape((-1, 3))
    # convert from Angstrom to Bohr
    vector = vector / BOHR
    # and change into left handed coordinate system by flipping y
    vector = vector * jnp.array([1, 1, -1])
    return vector

@jax.jit
def to_internal_vib_amps(vib_amps_ang):
    """Convert from Angstrom to atomic units.

    Parameters
    ----------
    vib_amps : array_like
        Vibrational amplitudes in Angstrom.

    Returns
    -------
    jnp.array
        Vibrational amplitudes in Bohr.
    """
    return vib_amps_ang / BOHR


def to_atomic_unit_energy(energy_eV):
    """Convert energy from eV to atomic units.

    Parameters
    ----------
    energy : float, array_like
        Energy in eV.

    Returns
    -------
    jnp.array
        Energy in atomic units.
    """
    return energy_eV / HARTREE


def kappa(energy, v_imag):
    """Return wave vector kappa (atomic units).

    In atomic units, the wave vector is given by:
    .. math::
    \kappa = \sqrt{2(E + V_{0i}})}

    Parameters
    ----------
    energy : float, array_like
        Electron energy in Hartree.
    v_imag : float, array_like
        Imaginary part of the inner potential in Hartree.

    Returns
    -------
    np.array
        Wave vector kappa.
    """
    return np.sqrt(2 * energy + 2j * v_imag)
