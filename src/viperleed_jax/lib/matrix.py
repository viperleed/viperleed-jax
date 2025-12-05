"""Module matrix from viperleed_jax.lib."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-06-10'
__copyright__ = 'Copyright (c) 2023-2025 ViPErLEED developers'
__license__ = 'GPLv3+'


import numpy as np


def closest_to_identity(matrices):
    """
    Select the matrix from a set that is closest to the identity matrix.

    Parameters
    ----------
    matrices (list of numpy.ndarray): List of matrices with the same shape.

    Returns
    -------
    int: Index of the matrix that is closest to the identity matrix.

    Raises
    ------
    ValueError: If the matrices have different shapes.
    """
    if not matrices:
        raise ValueError('The list of matrices is empty.')

    # Check if all matrices have the same shape
    reference_shape = matrices[0].shape
    for matrix in matrices:
        if matrix.shape != reference_shape:
            raise ValueError('All matrices must have the same shape.')

    # Create an identity matrix with the same shape as the input matrices
    identity_matrix = np.eye(reference_shape[0], reference_shape[1])

    # Calculate the Frobenius norm of the difference between each matrix and the identity matrix
    distances = [
        np.linalg.norm(matrix - identity_matrix, 'fro') for matrix in matrices
    ]

    # Find the index of the matrix with the smallest distance
    return np.argmin(distances)


def off_diagonal_frobenius(matrix):
    """
    Compute the frobenius norm of off-diagional elements for a real matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D real matrix (can be non-square).

    Returns
    -------
    float
        The Frobenius norm of the off-diagonal part of the matrix.
    """
    # Ensure input is a numpy array
    matrix = np.array(matrix)

    # Create the diagonal projection of the matrix
    diagonal_projection = np.zeros_like(matrix)
    np.fill_diagonal(diagonal_projection, np.diag(matrix))

    # Compute the Frobenius norm of the difference
    difference = matrix - diagonal_projection
    return np.linalg.norm(difference, 'fro')
