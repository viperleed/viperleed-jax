"""Module propagators."""

__authors__ = ('Alexander M. Imre (@amimre)', 'Paul Haidegger (@Paulhai7)')
__created__ = '2024-09-03'

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from viperleed_jax.dense_quantum_numbers import DENSE_QUANTUM_NUMBERS
from viperleed_jax.gaunt_coefficients import CSUM_COEFFS
from viperleed_jax.lib.math import (
    EPS,
    bessel,
    safe_norm,
    spherical_harmonics_components,
)


# @partial(jax.profiler.annotate_function, name="calc_propagator")
def calc_propagator(LMAX, c, c_sph_harm_components, kappa):
    c_norm = safe_norm(c)

    BJ = bessel(kappa * c_norm, 2 * LMAX)
    YLM = c_sph_harm_components

    dense_m_2d = DENSE_QUANTUM_NUMBERS[LMAX][:, :, 2]
    dense_mp_2d = DENSE_QUANTUM_NUMBERS[LMAX][:, :, 3]

    # AI: I don't fully understand this, technically it should be MPP = -M - MP
    dense_mpp = dense_mp_2d - dense_m_2d

    # pre-computed coeffs, capped to LMAX
    capped_coeffs = CSUM_COEFFS[
        : 2 * LMAX + 1, : (LMAX + 1) ** 2, : (LMAX + 1) ** 2
    ]

    idx_lookup = jnp.stack(
        [lpp * lpp + lpp - dense_mpp for lpp in range(2 * LMAX + 1)]
    )  # shape: (2*LMAX+1, n, n)

    def propagator_lpp_element(lpp, running_sum):
        bessel_values = BJ[lpp]
        ylm_values = YLM[idx_lookup[lpp]]
        # Equation (34) from Rous, Pendry 1989
        return (
            running_sum + bessel_values * ylm_values * capped_coeffs[lpp, :, :]
        )  # * (abs(dense_mpp) <= lpp)

    # we could skip some computations because some lpp are guaranteed to give
    # zero contributions, but this would need a way around the non-static array
    # sizes

    # This is the propagator from the origin to C
    propagator = jax.lax.fori_loop(
        0,
        LMAX * 2 + 1,
        propagator_lpp_element,
        jnp.zeros(
            shape=((LMAX + 1) ** 2, (LMAX + 1) ** 2), dtype=jnp.complex128
        ),
    )
    propagator *= 4 * jnp.pi
    return jnp.where(
        c_norm >= EPS * 100, propagator, jnp.identity((LMAX + 1) ** 2)
    )


# Using equation (34) from Rous, Pendry 1989 it is easy to show that the
# propagator for a vanishing displacement is the identity matrix.
# (The Bessel functions for zero argument are zero for all non-zero orders, thus
# l''=0 is the only non-zero term in the sum. If l'' is 0, m''=0 and l=l' are
# necessary conditions.)


def symmetry_operations(l_max, plane_symmetry_operation):
    """_summary_

    Parameters
    ----------
    l_max : int
        Maximum angular momentum quantum number. Compiled as static argument.
    plane_symmetry_operation : 2x2 array
        The in-plane symmetry operation matrix.

    Returns
    -------
    jax.numpy.ndarray, shape=((l_max+1)**2, (l_max+1)**2)
        Tensor that can be applied element-wise to the propagator to apply the
        symmetry operation.
    bool
        Whether the symmetry operation is a mirror operation. If True, the
        propagator should be transposed.
    """
    dense_m_2d = DENSE_QUANTUM_NUMBERS[l_max][:, :, 2]
    dense_mp_2d = DENSE_QUANTUM_NUMBERS[l_max][:, :, 3]
    # AI: I don't fully understand this, technically it should be MPP = -M - MP
    dense_mpp = dense_mp_2d - dense_m_2d

    plane_symmetry_det = np.linalg.det(plane_symmetry_operation)
    if abs(plane_symmetry_det) - 1 > 1e-8:
        raise ValueError(
            'The determinant of the plane symmetry operation '
            'matrix must be 1 or -1.'
        )
    contains_mirror = plane_symmetry_det < 0
    mirror_x = np.array([[-1.0, 0.0], [0.0, 1.0]])
    if contains_mirror:
        sym_op = plane_symmetry_operation @ mirror_x
    else:
        sym_op = plane_symmetry_operation

    plane_rotation_angle = get_plane_symmetry_operation_rotation_angle(sym_op)

    symmetry_tensor = jnp.exp(plane_rotation_angle * 1j * (dense_mpp)).T
    if contains_mirror:
        symmetry_tensor = (-1.0) ** (-dense_mpp) * jnp.exp(
            plane_rotation_angle * 1j * (dense_mpp)
        ).T

    return symmetry_tensor, contains_mirror


def get_plane_symmetry_operation_rotation_angle(plane_symmetry_operation):
    """Return the rotation angle for a plane symmetry operation.

    The rotation angle is calculated for the plane symmetry operation by
    embedding it in 3D space. Note that for non-rotation matrices, a rotation
    will still be calculated, but it will be meaningless.

    Parameters
    ----------
    plane_symmetry_operation : ndarray (2,2)
        In plane symmetry operation matrix.

    Returns
    -------
    float
        Rotation angle in radians.
    """
    return (
        np.log(
            plane_symmetry_operation[0, 0] + 1j * plane_symmetry_operation[1, 0]
        )
        / 1j
    ).real


def _calculate_dynamic_propagator(
    l_max, batch_atoms, displacements, components, kappa
):
    """
    Compute dynamic propagators for a single energy index.

    Returns an array of shape (num_displacements, ...).
    """
    return jax.lax.map(
        lambda atom_idx: calc_propagator(
            l_max,
            displacements[atom_idx],
            components[atom_idx],
            kappa,
        ),
        jnp.arange(len(displacements)),
        batch_size=batch_atoms,
    )


@partial(
    jax.jit,
    static_argnames=['l_max', 'batch_atoms', 'batch_energies', 'use_symmetry'],
)
def calculate_propagators(
    propagtor_context,
    displacements,
    energy_indices,
    batch_energies,
    batch_atoms,
    l_max,
    use_symmetry=True,
):
    # We want the final result indexed as (energies, atom_basis, lm, l'm')
    energy_indices = jnp.array(energy_indices)

    # Precompute the spherical harmonics components for each displacement.
    displacement_components = jnp.array(
        [spherical_harmonics_components(l_max, disp) for disp in displacements]
    )

    def process_energy(e_idx):
        # --- Dynamic propagators ---
        if len(displacements) > 0:
            # Now call the per-energy dynamic propagator.
            dyn = _calculate_dynamic_propagator(
                l_max,
                batch_atoms,
                displacements,
                displacement_components,
                propagtor_context.kappa[e_idx],
            )
        else:
            dyn = jnp.zeros_like(propagtor_context.static_propagators[0])

        if use_symmetry:
            # --- Static propagators ---
            if len(propagtor_context.static_propagators) == 0:
                stat = jnp.zeros_like(dyn)
            else:
                # Assuming self._static_propagators is indexed as
                # (atom_basis, num_energies, lm, m)
                stat = propagtor_context.static_propagators[:, e_idx, :, :]

            # --- Map to atom basis using propagator_id ---
            mapped_dyn = dyn[propagtor_context.propagator_id]
            mapped_stat = stat[propagtor_context.propagator_id]

            # --- Combine dynamic and static parts ---
            # Condition is broadcast along the last two axes.
            cond = propagtor_context.is_dynamic_propagator[:, None, None]
            combined = jnp.where(cond, mapped_dyn, mapped_stat)
            # combined now has shape (atom_basis, lm, m)

            # --- Apply selective transposition ---
            trans_int = propagtor_context.propagator_transpose_int[
                :, None, None
            ]
            return (1 - trans_int) * combined + trans_int * jnp.transpose(
                combined, (0, 2, 1)
            )
            # combined remains (atom_basis, lm, m)

        return dyn

    # Process each energy individually.
    # Each process_energy returns (atom_basis, lm, m); mapping over energies
    # yields shape: (num_energies, atom_basis, lm, m)
    per_energy = jax.lax.map(
        process_energy, energy_indices, batch_size=batch_energies
    )
    # Transpose to (atom_basis, num_energies, lm, m) to match what the symmetry
    # einsum expects.
    # per_energy = jnp.transpose(per_energy, (1, 0, 2, 3))

    if use_symmetry:
        # --- Apply rotations (symmetry operations) and rearrange ---
        propagators = jnp.einsum(
            'ealm,alm->ealm',
            per_energy,
            propagtor_context.symmetry_operations,
            optimize='optimal',
        )
    else:
        # --- Apply rotations (symmetry operations) and rearrange ---
        propagators = jnp.einsum(
            'ealm->ealm',
            per_energy,
            optimize='optimal',
        )

    # Final shape is (energies, atom_basis, lm, m)
    return propagators
