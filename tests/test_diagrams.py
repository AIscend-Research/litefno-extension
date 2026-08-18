"""The mode-shell diagram must keep describing what the layer actually does.

``scripts/make_diagrams.py`` reimplements the shell selection in numpy so the
figure can be rebuilt without torch. That duplication is the point of failure:
someone changes the radius rule in the model and the picture keeps showing the
old one, which is worse than having no picture. These tests pin the two.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_make_diagrams():
    """Import scripts/make_diagrams.py, which is not part of the package."""
    path = ROOT / "scripts" / "make_diagrams.py"
    spec = importlib.util.spec_from_file_location("make_diagrams", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_diagrams"] = module
    spec.loader.exec_module(module)
    return module


md = _load_make_diagrams()


@pytest.mark.parametrize("fundamental,n_harmonics", [(3.5, 2), (3.0, 3), (2.0, 4), (5.0, 1)])
def test_shell_radii_matches_model(fundamental, n_harmonics):
    torch = pytest.importorskip("torch")  # noqa: F841
    from litefno.models.harmonic import harmonic_shells

    assert md.shell_radii(fundamental, n_harmonics, 12) == harmonic_shells(
        fundamental, n_harmonics, 12
    )


@pytest.mark.parametrize("modes1,modes2", [(12, 12), (8, 12), (13, 7)])
@pytest.mark.parametrize("shells", [[3.5, 7.0], [2.0], [1.0, 2.0, 3.0]])
def test_shell_mask_matches_model(modes1, modes2, shells):
    """Same modes selected, allowing for the deliberate row reordering.

    The diagram lays k_y out centred so the picture looks like a spectrum; the
    layer stores the negative half folded to the end. Rolling one into the
    other is the only difference permitted.
    """
    torch = pytest.importorskip("torch")
    from litefno.models.harmonic import harmonic_mask

    model_mask = harmonic_mask(modes1, modes2, shells).numpy()
    diagram_mask, ky, _ = md.shell_mask(modes1, modes2, shells)

    # Centred order -> folded order: the positive rows come first.
    neg = modes1 - (modes1 // 2 + 1)
    refolded = np.roll(diagram_mask, -neg, axis=0)

    assert refolded.shape == model_mask.shape
    np.testing.assert_array_equal(refolded, model_mask)
    # And the count quoted in the figure legend is the count the layer biases.
    assert int(diagram_mask.sum()) == int(model_mask.sum())


def test_shell_mask_selects_a_ring_not_an_axis():
    """A shell must catch modes off the axes, or the ring claim is false."""
    mask, ky, kx = md.shell_mask(12, 12, [5.0])
    off_axis = [
        (int(y), int(x))
        for i, y in enumerate(ky)
        for j, x in enumerate(kx)
        if mask[i, j] and y != 0 and x != 0
    ]
    assert (3, 4) in off_axis or (-3, 4) in off_axis
    assert len(off_axis) >= 4


def test_generated_svg_is_wellformed_and_self_contained(tmp_path):
    """No external refs, no scripts, and parseable as XML.

    The figure gets embedded in an HTML page under a strict CSP, so an
    external reference would silently render as nothing.
    """
    import xml.etree.ElementTree as ET

    out = tmp_path / "mode_shells.svg"
    md.render_mode_shells(out)
    text = out.read_text(encoding="utf-8")

    ET.parse(out)  # raises on malformed XML, including "--" inside a comment
    assert "<script" not in text
    assert "http://" not in text.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https://" not in text
    assert "<image" not in text


def test_committed_svg_is_current():
    """The checked-in figure matches what the script emits today."""
    import tempfile

    committed = ROOT / "figures" / "diagrams" / "mode_shells.svg"
    if not committed.exists():
        pytest.skip("figure not generated yet; run scripts/make_diagrams.py")

    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "mode_shells.svg"
        md.render_mode_shells(fresh)
        assert fresh.read_text(encoding="utf-8") == committed.read_text(
            encoding="utf-8"
        ), (
            "figures/diagrams/mode_shells.svg is stale -- rerun scripts/make_diagrams.py"
        )
