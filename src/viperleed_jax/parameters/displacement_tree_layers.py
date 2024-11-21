"""Module displacement_tree_layers."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-21'

from enum import Enum


DisplacementTreeLayers = Enum(
    'HLTreeLayers',
    [
        'Base',
        'Symmetry',
        'Backend_Constraints',
        'User_Constraints',
        'Implicit_Constraints',
        'Root',
    ],
)
