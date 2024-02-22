import numpy as np
import scipy
import jax
import jax.numpy as jnp

from lib_tscatf import HARTREE


def intensity_prefactor(CDISP, e_kin, v_real, v_imag, beam_indices, theta, phi, trar):
    # prefactors (refraction) from amplitudes to intensities
    n_geo = CDISP.shape[0]
    n_energies = e_kin.shape[0]
    n_beams = beam_indices.shape[0]

    in_k, bk_z, out_k_z, out_k_perp = _wave_vectors(e_kin, v_real, v_imag, theta, phi, trar, beam_indices)

    a = np.sqrt(out_k_perp)
    c = in_k * np.cos(theta)

    prefactor = np.full((n_geo, n_energies, n_beams), dtype=np.float64, fill_value=np.nan)
    for i in range(n_geo):
        CXDisp = CDISP[i,0,0]
        prefactor[i,:,:] = abs(np.exp(-1j * CXDisp * (np.outer(bk_z, np.ones(shape=(n_beams,))) + out_k_z
                                                    ))) ** 2 * a / np.outer(c, np.ones(shape=(n_beams,))).real
    return prefactor


def _wave_vectors(e_kin, v_real, v_imag, theta, phi, trar, beam_indices):
    n_energies = e_kin.shape[0]
    n_beams = beam_indices.shape[0]
    # incident wave vector
    in_k = np.sqrt(np.maximum(0, 2 * (e_kin - v_real)))
    in_k_par = in_k * np.sin(theta)  # parallel component
    bk_2 = in_k_par * np.cos(phi)  # shape =( n_energy )
    bk_3 = in_k_par * np.sin(phi)  # shape =( n_energy )
    bk_z = np.empty_like(e_kin, dtype="complex64")
    bk_z = 2 * e_kin - bk_2 ** 2 - bk_3 ** 2 - 2 * 1j * v_imag
    bk_z = np.sqrt(bk_z)

    # outgoing wave vector components
    bk_components = np.stack((bk_2, bk_3))  # shape =(n_en, 2)
    bk_components = np.outer(bk_components, np.ones(shape=(n_beams,))).reshape(
    (n_energies, 2, n_beams))  # shape =(n_en ,2 ,n_beams)
    out_wave_vec = np.dot(beam_indices, trar)  # shape =(n_beams, 2)
    out_wave_vec = np.outer(np.ones_like(e_kin), out_wave_vec).reshape((n_energies, 2, n_beams))  # shape =(n_en , n_beams)
    out_components = bk_components + out_wave_vec

    # out k vector
    out_k = (2 * np.outer(e_kin, np.ones(shape=(n_beams,)))  # 2*E
         + bk_components[:, 0, :] ** 2  # + h **2
         + bk_components[:, 1, :] ** 2  # + k **2
         ).astype(dtype="complex64")
    out_k_z = np.empty_like(out_k, dtype="complex64")  # shape =(n_en , n_beams )
    out_k_z = np.sqrt(out_k - 2 * 1.0j * np.outer(v_imag, np.ones(shape=(n_beams,))))
    out_k_perp = out_k - 2 * np.outer(v_real, np.ones(shape=(n_beams,)))

    # TODO: @Paul: below quantities are unused, is that intentional?
    out_k_par = 2 * 1.0j * np.outer(v_imag, np.ones(shape=(n_beams,)))
    out_bk_2 = out_k_par * np.cos(phi)
    out_bk_3 = out_k_par * np.sin(phi)

    return in_k, bk_z, out_k_z, out_k_perp
