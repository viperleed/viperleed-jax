"""Module atom_units.

We use Angstrom and eV as inputs, but use atomic units internally."""
import jax.numpy as jnp

from viperleed_jax.constants import BOHR, HARTREE

def to_internal_displacement_vector(displacement_vector):
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
    vector = displacement_vector.reshape((3,))
    # convert from Angstrom to Bohr
    vector = vector / BOHR
    # and change into left handed coordinate system by flipping y
    vector = vector * jnp.array([1, 1, -1])
    return vector



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
    jnp.array
        Wave vector kappa.
    """
    return jnp.sqrt(2 * energy + 2j * v_imag)
