import jax.numpy as jnp

from viperleed_jax.constants import BOHR, HARTREE

def kappa(energy, v_imag):
    """Return wave vector kappa.

    \kappa = \sqrt{2(E + V_{0i}})}

    Parameters
    ----------
    energy : float, array_like
        Electron energy.
    v_imag : float, array_like
        Imaginary part of the inner potential.

    Returns
    -------
    jnp.array
        Wave vector kappa.
    """
    return jnp.sqrt(2 * energy + 2j * v_imag)
