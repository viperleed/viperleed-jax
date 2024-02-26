import jax
import jax.numpy as jnp

from src.lib_tscatf import HARTREE


def intensity_prefactor(CDISP, e_kin, v_real, v_imag, beam_indices, theta, phi, trar, is_surface_atom):
    # prefactors (refraction) from amplitudes to intensities
    n_geo = CDISP.shape[0]
    n_energies = e_kin.shape[0]
    n_beams = beam_indices.shape[0]

    in_k, bk_z, out_k_z, out_k_perp = _wave_vectors(e_kin, v_real, v_imag, theta, phi, trar, beam_indices)

    a = jnp.sqrt(out_k_perp)
    c = in_k * jnp.cos(theta)

    prefactor = jnp.full((n_geo, n_energies, n_beams), dtype=jnp.float64, fill_value=jnp.nan)
    for i in range(n_geo):
        CXDisp = _potential_onset_height_change(CDISP[i,:,:], is_surface_atom)
        # TODO: @Paul: should we use out_k_z here really? Also this raises a warning in numpy about complex casting
        prefactor[i,:,:] = abs(jnp.exp(-1j * CXDisp * (jnp.outer(bk_z, jnp.ones(shape=(n_beams,))) + out_k_z
                                                    ))) ** 2 * a / jnp.outer(c, jnp.ones(shape=(n_beams,))).real
    return prefactor


def _wave_vectors(e_kin, v_real, v_imag, theta, phi, trar, beam_indices):
    n_energies = e_kin.shape[0]
    n_beams = beam_indices.shape[0]
    # incident wave vector
    in_k = jnp.sqrt(jnp.maximum(0, 2 * (e_kin - v_real)))
    in_k_par = in_k * jnp.sin(theta)  # parallel component
    bk_2 = in_k_par * jnp.cos(phi)  # shape =( n_energy )
    bk_3 = in_k_par * jnp.sin(phi)  # shape =( n_energy )
    bk_z = jnp.empty_like(e_kin, dtype="complex64")
    bk_z = 2 * e_kin - bk_2 ** 2 - bk_3 ** 2 - 2 * 1j * v_imag
    bk_z = jnp.sqrt(bk_z)

    # outgoing wave vector components
    bk_components = jnp.stack((bk_2, bk_3))  # shape =(n_en, 2)
    bk_components = jnp.outer(bk_components, jnp.ones(shape=(n_beams,))).reshape(
    (n_energies, 2, n_beams))  # shape =(n_en ,2 ,n_beams)
    out_wave_vec = jnp.dot(beam_indices, trar)  # shape =(n_beams, 2)
    out_wave_vec = jnp.outer(jnp.ones_like(e_kin), out_wave_vec).reshape((n_energies, 2, n_beams))  # shape =(n_en , n_beams)
    out_components = bk_components + out_wave_vec

    # out k vector
    out_k = (2 * jnp.outer(e_kin, jnp.ones(shape=(n_beams,)))  # 2*E
         + bk_components[:, 0, :] ** 2  # + h **2
         + bk_components[:, 1, :] ** 2  # + k **2
         ).astype(dtype="complex64")
    out_k_z = jnp.empty_like(out_k, dtype="complex64")  # shape =(n_en , n_beams )
    out_k_z = jnp.sqrt(out_k - 2 * 1.0j * jnp.outer(v_imag, jnp.ones(shape=(n_beams,))))
    out_k_perp = out_k - 2 * jnp.outer(v_real, jnp.ones(shape=(n_beams,)))

    # TODO: @Paul: below quantities are unused, is that intentional?
    out_k_par = 2 * 1.0j * jnp.outer(v_imag, jnp.ones(shape=(n_beams,)))
    out_bk_2 = out_k_par * jnp.cos(phi)
    out_bk_3 = out_k_par * jnp.sin(phi)

    return in_k, bk_z, out_k_z, out_k_perp


def _potential_onset_height_change(displacement_step, is_surface_atom):
    """Calculates the displacement of the topmost surface atom in the z direction."""
    # TODO: this is actually not really correct!
    # We should not consider suface atoms, but only the topmost one probably!!
    surface_z = displacement_step[is_surface_atom, 0] # z disp for surface atoms
    return jnp.max(surface_z)
