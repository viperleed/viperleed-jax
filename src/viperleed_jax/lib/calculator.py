"""Library for tensor_calculator."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2025-04-28'


def map_indices(source, reference):
    """
    Map each value in `source` to its index in `reference`.

    Parameters
    ----------
    source : np.ndarray of shape (n,)
        Array of integers, possibly with duplicates.
    reference : np.ndarray of shape (m,)
        Array of unique integers. All values in `source` must exist in `reference`.

    Returns
    -------
    np.ndarray of shape (n,)
        Array of indices such that `reference[result[i]] == source[i]`.

    Raises
    ------
    KeyError if any value in `source` is not found in `reference`.
    """
    # Create a dictionary mapping values in `reference` to their indices
    ref_map = {val: i for i, val in enumerate(reference)}

    # Map each element in `source` to its corresponding index in `reference`
    result = tuple([ref_map[val] for val in source])

    return result
