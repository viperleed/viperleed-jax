"""Module displacements/regex."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@PaulHai7)')
__created__ = '2024-02-22'

from functools import partial

import jax
import jax.numpy as jnp

from viperleed_jax.constants import BOHR, DISP_Z_DIR_ID


@jax.jit
def sum_intensity(prefactors, reference_amplitudes, delta_amplitudes):
    return prefactors * abs(reference_amplitudes + delta_amplitudes) ** 2

def _wave_vectors(ref_data, theta, phi, trar, beam_indices):
    e_kin = ref_data.energies
    v_real = ref_data.v0r
    v_imag = ref_data.v0i
    n_energies = e_kin.shape[0]
    n_beams = beam_indices.shape[0]
    # incident wave vector
    in_k_vacuum = jnp.sqrt(jnp.maximum(0, 2 * (e_kin - v_real)))
    in_k_par = in_k_vacuum * jnp.sin(theta)  # parallel component
    in_k_par_2 = in_k_par * jnp.cos(phi)  # shape =( n_energy )
    in_k_par_3 = in_k_par * jnp.sin(phi)  # shape =( n_energy )
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
    out_wave_vec = jnp.dot(beam_indices, trar)  # shape =(n_beams, 2)
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


@partial(jax.jit, static_argnames=['n_beams'])
def intensity_prefactors(
    displacements,
    atoms_ref_z_pos,
    n_beams,
    theta,
    wave_vectors):
    # onset height change was called CXDisp in the original code
    onset_height_change = potential_onset_height_change(
        atoms_ref_z_pos, displacements
    )

    # from lib_intensity
    (in_k_vacuum, in_k_perp_vacuum, out_k_perp, out_k_perp_vacuum) = wave_vectors

    a = out_k_perp_vacuum
    c = in_k_vacuum * jnp.cos(theta)

    # TODO: re-check if it should be a.real or abs(a)
    return (
        abs(
            jnp.exp(
                -1j
                * onset_height_change
                / BOHR
                * (
                    jnp.outer(
                        in_k_perp_vacuum, jnp.ones(shape=(n_beams,))
                    )
                    + out_k_perp
                )
            )
        )
        ** 2
        * a.real
        / jnp.outer(c, jnp.ones(shape=(n_beams,))).real
    )


def potential_onset_height_change(atoms_ref_z_pos, displacements):
    """Calculate the change in the highest atom z position.

    This is needed because the onset height of the inner potential is
    defined as the z position of the highest atom in the slab.
    Therefore, changes to this height may change refraction of the incoming
    electron wave.

    Parameters
    ----------
    atoms_ref_z_pos : array_like
        Reference z positions of the atoms.
    displacements : array_like
        Displacements of the atoms in the slab.

    Returns
    -------
    jax.Array(float)
        Change in the highest atom z position.
    """
    z_changes = jnp.asarray(displacements)[:, DISP_Z_DIR_ID]
    new_z_pos = atoms_ref_z_pos + z_changes
    return jnp.max(new_z_pos) - jnp.max(atoms_ref_z_pos)
