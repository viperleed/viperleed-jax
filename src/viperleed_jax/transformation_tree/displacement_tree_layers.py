"""Module displacement_tree_layers."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-21'
__copyright__ = 'Copyright (c) 2023-2025 ViPErLEED developers'
__license__ = 'GPLv3+'

from enum import Enum

DisplacementTreeLayers = Enum(
    'DisplacementTreeLayers',
    [
        'Base',
        'Symmetry',
        'Backend_Constraints',
        'Offsets',
        'User_Constraints',
        'Implicit_Constraints',
        'Root',
    ],
)
