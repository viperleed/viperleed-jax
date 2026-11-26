"""Tests for the graphical export of transformation trees."""

__authors__ = ('Florian Kraushofer (@fkraushofer)',)
__created__ = '2026-11-26'

import anytree

from viperleed_jax.transformation_tree import tree


def test_defaults_unchanged(monkeypatch):
    """Check that export options do not leak into the defaults."""
    defaults = dict(tree.DEFAULT_GRAPHICAL_EXPORT_OPTIONS)
    used_options = []

    class _FakeExporter:
        def __init__(self, _, options):
            used_options.append(options)

        def to_picture(self, filename):
            pass

    monkeypatch.setattr(tree, 'UniqueDotExporter', _FakeExporter)
    root = anytree.Node('root')
    tree.save_tree_diagram_pdf(root, 'a.pdf', {'rankdir': 'TB', 'dpi': 50})
    assert tree.DEFAULT_GRAPHICAL_EXPORT_OPTIONS == defaults
    tree.save_tree_diagram_pdf(root, 'b.pdf', None)
    assert tree.DEFAULT_GRAPHICAL_EXPORT_OPTIONS == defaults
    assert sorted(used_options[0]) == ['dpi=50', 'rankdir=TB']
    assert used_options[1] == [f'{k}={v}' for k, v in defaults.items()]
