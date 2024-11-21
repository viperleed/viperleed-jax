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
