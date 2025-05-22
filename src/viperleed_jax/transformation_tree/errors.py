"""Module transformation_tree/errors."""

__authors__ = ('Alexander M. Imre (@amimre)',)
__created__ = '2024-11-25'


class TransformationTreeError(Exception):
    """Base class for errors in the transformation tree and nodes."""

class NodeCreationError(TransformationTreeError):
    """Error raised when a node cannot be created."""

class InvalidNodeError(NodeCreationError):
    """Error raised when the inputs for a node are invalid."""
