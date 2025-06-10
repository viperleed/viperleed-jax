import numpy as np
import pytest
from pytest_cases import parametrize_with_cases, fixture

from viperleed_jax.atom_basis import AtomBasis
from viperleed_jax.transformation_tree.displacement_tree_layers import (
    DisplacementTreeLayers,
)
from viperleed_jax.transformation_tree.geo_parameters import GeoTree
from viperleed_jax.transformation_tree.linear_transformer import (
    AffineTransformer,
)
from viperleed_jax.transformation_tree.reduced_space import (
    apply_affine_to_subspace,
)

from ..structures import CaseStatesAfterInit


from viperleed_jax.files.displacements.lines import (
    OffsetsLine,
    OccDeltaLine,
    ConstraintLine,
)
