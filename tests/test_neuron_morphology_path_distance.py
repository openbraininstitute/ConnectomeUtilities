# SPDX-License-Identifier: Apache-2.0
"""Exact-value tests for MorphologyPathDistanceCalculator public API."""
import numpy as np
import pandas as pd
import pytest
import morphio
from morphio import PointLevel, SectionType
from morphio.mut import Morphology as MutMorph

from conntility.subcellular import MorphologyPathDistanceCalculator


def _build_fixture_morphology():
    """
    Hand-built tree with known lengths (all section lengths = 10):

        soma
        ├── sec0 (basal): (0,0,0)->(5,0,0)->(10,0,0)   two segments of length 5
        │   ├── sec1: (10,0,0)->(20,0,0)
        │   │   ├── sec2: (20,0,0)->(30,0,0)
        │   │   └── sec3: (20,0,0)->(20,10,0)
        │   └── sec4: (10,0,0)->(10,10,0)
        └── sec5 (axon root): (0,0,0)->(0,0,10)
    """
    m = MutMorph()
    m.soma.points = [[0.0, 0.0, 0.0]]
    m.soma.diameters = [1.0]
    s0 = m.append_root_section(
        PointLevel(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
        ),
        SectionType.basal_dendrite,
    )
    s1 = s0.append_section(
        PointLevel([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], [1.0, 1.0]),
        SectionType.basal_dendrite,
    )
    s1.append_section(
        PointLevel([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], [1.0, 1.0]),
        SectionType.basal_dendrite,
    )
    s1.append_section(
        PointLevel([[20.0, 0.0, 0.0], [20.0, 10.0, 0.0]], [1.0, 1.0]),
        SectionType.basal_dendrite,
    )
    s0.append_section(
        PointLevel([[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]], [1.0, 1.0]),
        SectionType.basal_dendrite,
    )
    m.append_root_section(
        PointLevel([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]], [1.0, 1.0]),
        SectionType.axon,
    )
    return morphio.Morphology(m)


def _locs(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "afferent_section_id",
            "afferent_segment_id",
            "afferent_segment_offset",
        ],
    )


@pytest.fixture(scope="module")
def morphology():
    return _build_fixture_morphology()


@pytest.fixture(scope="module")
def calc(morphology):
    return MorphologyPathDistanceCalculator(morphology)


def test_morphology_connectivity(morphology):
    """Section 0 is a root; tree edges match the fixture."""
    assert 0 in morphology.connectivity[-1]
    assert dict(morphology.connectivity) == {-1: [0, 5], 0: [1, 4], 1: [2, 3]}
    assert [s.id for s in morphology.root_sections] == [0, 5]
    assert len(morphology.sections) == 6


def test_init_requires_first_section_rooted_at_soma():
    class _FakeMorph:
        connectivity = {-1: [1], 1: [0]}  # section 0 not under soma

    with pytest.raises(AssertionError, match="first section"):
        MorphologyPathDistanceCalculator(_FakeMorph())


def test_segment_offset_table(morphology, calc):
    expected = np.array(
        [
            [0.0, 5.0, 10.0],
            [0.0, 10.0, np.nan],
            [0.0, 10.0, np.nan],
            [0.0, 10.0, np.nan],
            [0.0, 10.0, np.nan],
            [0.0, 10.0, np.nan],
        ]
    )
    table = MorphologyPathDistanceCalculator.segment_offset_table(morphology)
    np.testing.assert_allclose(table, expected, equal_nan=True)
    np.testing.assert_allclose(calc.offset, expected, equal_nan=True)


def test_within_section_offsets(calc):
    locs = _locs(
        [
            [1, 0, 0.0],
            [1, 1, 0.0],
            [1, 1, 5.0],
            [2, 0, 0.0],
            [3, 0, 5.0],
            [4, 0, 0.0],
            [5, 0, 0.0],
            [6, 0, 0.0],
            [0, 0, 99.0],  # soma -> always 0
        ]
    )
    np.testing.assert_allclose(
        calc.within_section_offsets(locs),
        [0.0, 5.0, 10.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_within_section_offsets_custom_column_names(calc):
    locs = pd.DataFrame(
        {
            "sid": [1, 3],
            "seg": [1, 0],
            "off": [2.5, 4.0],
        }
    )
    np.testing.assert_allclose(
        calc.within_section_offsets(
            locs, str_section_id="sid", str_segment_id="seg", str_offset="off"
        ),
        [7.5, 4.0],
    )


def test_path_distances_pairwise_exact(calc):
    """Covers same-section, ancestor, descendant, sibling, multi-root, and soma."""
    locs = _locs(
        [
            [1, 0, 0.0],   # root of sec0
            [1, 1, 0.0],   # mid sec0
            [1, 1, 5.0],   # end sec0
            [2, 0, 0.0],   # root sec1
            [3, 0, 5.0],   # mid sec2
            [4, 0, 0.0],   # root sec3 (sibling of sec2)
            [5, 0, 0.0],   # root sec4
            [6, 0, 0.0],   # root sec5 (other root)
            [0, 0, 99.0],  # soma (~ root of sec0)
        ]
    )
    expected = np.array(
        [
            [0.0, 5.0, 10.0, 10.0, 25.0, 20.0, 10.0, 0.0, 0.0],
            [5.0, 0.0, 5.0, 5.0, 20.0, 15.0, 5.0, 5.0, 5.0],
            [10.0, 5.0, 0.0, 0.0, 15.0, 10.0, 0.0, 10.0, 10.0],
            [10.0, 5.0, 0.0, 0.0, 15.0, 10.0, 0.0, 10.0, 10.0],
            [25.0, 20.0, 15.0, 15.0, 0.0, 5.0, 15.0, 25.0, 25.0],
            [20.0, 15.0, 10.0, 10.0, 5.0, 0.0, 10.0, 20.0, 20.0],
            [10.0, 5.0, 0.0, 0.0, 15.0, 10.0, 0.0, 10.0, 10.0],
            [0.0, 5.0, 10.0, 10.0, 25.0, 20.0, 10.0, 0.0, 0.0],
            [0.0, 5.0, 10.0, 10.0, 25.0, 20.0, 10.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(calc.path_distances(locs), expected)


def test_path_distances_rectangular_and_same_section_only(calc):
    locs = _locs(
        [
            [1, 0, 0.0],
            [1, 1, 0.0],
            [1, 1, 5.0],
            [2, 0, 0.0],
            [3, 0, 5.0],
        ]
    )
    np.testing.assert_allclose(
        calc.path_distances(locs.iloc[:2], locs.iloc[3:5]),
        [[10.0, 25.0], [5.0, 20.0]],
    )
    same = calc.path_distances(locs.iloc[:3], same_section_only=True)
    np.testing.assert_allclose(
        same,
        [[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]],
    )


def test_path_distances_tree_relations_geometry(calc):
    """Spot-check geometric path lengths for each relation class."""
    # same section: |5 - 0| on sec0
    a = _locs([[1, 1, 0.0]])
    b = _locs([[1, 0, 0.0]])
    np.testing.assert_allclose(calc.path_distances(a, b), [[5.0]])

    # ancestor: point mid sec2 (o=5) to mid sec0 (o=5)
    # depth_from_soma: 20+5=25 and 0+5=5 -> dist 20
    a = _locs([[3, 0, 5.0]])
    b = _locs([[1, 1, 0.0]])
    np.testing.assert_allclose(calc.path_distances(a, b), [[20.0]])

    # siblings under sec1: roots coincide, dist = o_a + o_b
    a = _locs([[3, 0, 3.0]])
    b = _locs([[4, 0, 4.0]])
    np.testing.assert_allclose(calc.path_distances(a, b), [[7.0]])

    # cousins (sec2 vs sec4): branch at end of sec0, dist = 10 + o_a + o_b
    a = _locs([[3, 0, 2.0]])
    b = _locs([[5, 0, 3.0]])
    np.testing.assert_allclose(calc.path_distances(a, b), [[15.0]])

    # different roots: branch at soma, dist = o_a + o_b
    a = _locs([[1, 1, 5.0]])  # end of sec0, within-section offset 10
    b = _locs([[6, 0, 4.0]])
    np.testing.assert_allclose(calc.path_distances(a, b), [[14.0]])


def test_nearest_neighbor_distances_unnormalized(calc):
    locs = _locs(
        [
            [3, 0, 1.0],
            [3, 0, 9.0],
            [4, 0, 2.0],
            [5, 0, 3.0],
            [2, 0, 4.0],
        ]
    )
    locs.index = pd.Index(["g1", "g1", "g1", "g2", "g2"], name="group")
    result = calc.nearest_neighbor_distances(locs, normalize=False)
    assert list(result.index) == ["g1", "g2"]
    np.testing.assert_allclose(result["g1"], [3.0, 8.0, 3.0])
    np.testing.assert_allclose(result["g2"], [7.0, 7.0])


def test_nearest_neighbor_distances_normalized(calc):
    locs = _locs(
        [
            [3, 0, 1.0],
            [3, 0, 9.0],
            [4, 0, 2.0],
            [5, 0, 3.0],
            [2, 0, 4.0],
        ]
    )
    locs.index = pd.Index(["g1", "g1", "g1", "g2", "g2"], name="group")
    locs["pre_id"] = [10, 10, 11, 20, 21]

    np.random.seed(0)
    z = calc.nearest_neighbor_distances(locs, normalize=True, normalize_n=5)
    np.testing.assert_allclose(
        z.to_numpy(),
        [-0.8027055346737423, -0.8677626681257792],
    )

    np.random.seed(0)
    z_preserve = calc.nearest_neighbor_distances(
        locs, normalize=True, normalize_preserve="pre_id", normalize_n=5
    )
    np.testing.assert_allclose(
        z_preserve.to_numpy(),
        [-1.0053382799917074, -0.38719109432113064],
    )
