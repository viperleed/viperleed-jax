import jax
import jax.numpy as jnp

from src.lib_delta import HARTREE


def sum_intensity(prefactors, reference_amplitudes, delta_amplitudes):
    return prefactors * abs(reference_amplitudes + delta_amplitudes) ** 2


def intensity_prefactor(displacement, ref_data,
                        beam_indices, theta, phi, trar, is_surface_atom):
    # prefactors (refraction) from amplitudes to intensities
    n_beams = beam_indices.shape[0]
    (in_k_vacuum, in_k_perp_vacuum,
     out_k_perp, out_k_perp_vacuum) = _wave_vectors(ref_data, theta, phi,
                                                    trar, beam_indices)

    a = out_k_perp_vacuum
    c = in_k_vacuum * jnp.cos(theta)

    CXDisp = _potential_onset_height_change(displacement, is_surface_atom)
    prefactor = abs(jnp.exp(-1j * CXDisp/BOHR * (jnp.outer(in_k_perp_vacuum, jnp.ones(shape=(n_beams,))) + out_k_perp
                                                ))) ** 2 * a.real / jnp.outer(c, jnp.ones(shape=(n_beams,))).real
    return prefactor


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
    in_k_perp_vacuum = 2 * e_kin - in_k_par_2 ** 2 - in_k_par_3 ** 2 - 2 * 1j * v_imag
    in_k_perp_vacuum = jnp.sqrt(in_k_perp_vacuum)

    # outgoing wave vector components
    in_k_par_components = jnp.stack((in_k_par_2, in_k_par_3))  # shape =(n_en, 2)
    in_k_par_components = jnp.outer(in_k_par_components, jnp.ones(shape=(n_beams,))).reshape(
    (n_energies, 2, n_beams))  # shape =(n_en ,2 ,n_beams)
    out_wave_vec = jnp.dot(beam_indices, trar.T)  # shape =(n_beams, 2)
    out_wave_vec = jnp.outer(jnp.ones_like(e_kin), out_wave_vec.transpose()).reshape((n_energies, 2, n_beams))  # shape =(n_en , n_beams)
    out_k_par_components = in_k_par_components + out_wave_vec

    # out k vector
    out_k_perp_vacuum = (2*jnp.outer(e_kin-v_real,jnp.ones(shape=(n_beams,)))
                - out_k_par_components[:, 0, :] ** 2
                - out_k_par_components[:, 1, :] ** 2).astype(dtype="complex64")
    out_k_perp = jnp.sqrt(out_k_perp_vacuum + 2*jnp.outer(v_real-1j*v_imag, jnp.ones(shape=(n_beams,))))
    out_k_perp_vacuum = jnp.sqrt(out_k_perp_vacuum)

    return in_k_vacuum, in_k_perp_vacuum, out_k_perp, out_k_perp_vacuum


def _potential_onset_height_change(displacement_step, is_surface_atom):
    """Calculates the displacement of the topmost surface atom in the z direction."""
    # TODO: this is actually not really correct!
    # We should not consider suface atoms, but only the topmost one probably!!
    surface_z = displacement_step[is_surface_atom, 0] # z disp for surface atoms
    return jnp.max(surface_z)
