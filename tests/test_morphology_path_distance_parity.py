# SPDX-License-Identifier: Apache-2.0
"""Public-API results locked to values produced by MorphologyPathDistanceCalculator on main."""
import morphio
import numpy as np
import pandas as pd
import pytest
from morphio import PointLevel, SectionType
from morphio.mut import Morphology as MutMorph

from conntility.subcellular import MorphologyPathDistanceCalculator
from mpdc_test_utils import (
    as_offset_array,
    assert_public_offset_api,
    build_branched_morphology,
    build_chain_morphology,
    build_tiny_nonround_morphology,
    locs_df,
)


@pytest.fixture(scope="module")
def branched():
    return build_branched_morphology()


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_nonround_morphology()


@pytest.fixture(scope="module")
def calc(branched):
    return MorphologyPathDistanceCalculator(branched)


def test_offset_table_matches_main_values(branched, calc):
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
    np.testing.assert_allclose(
        MorphologyPathDistanceCalculator.segment_offset_table(branched),
        expected,
        equal_nan=True,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(as_offset_array(calc.offset), expected, equal_nan=True)
    np.testing.assert_allclose(
        as_offset_array(calc.offset),
        MorphologyPathDistanceCalculator.segment_offset_table(branched),
        equal_nan=True,
    )
    assert_public_offset_api(calc)


def test_within_section_offsets_match_main_values(calc):
    locs = locs_df(
        [
            [1, 0, 0.0],
            [1, 1, 2.5],
            [3, 0, 5.0],
            [6, 0, 1.0],
            [0, 0, 99.0],
        ]
    )
    np.testing.assert_allclose(
        calc.within_section_offsets(locs),
        [0.0, 7.5, 5.0, 1.0, 0.0],
        rtol=0,
        atol=0,
    )


def test_path_distances_match_main_values_on_branched_tree(calc):
    locs = locs_df(
        [
            [1, 0, 0.0],
            [1, 1, 0.0],
            [1, 1, 5.0],
            [2, 0, 0.0],
            [3, 0, 5.0],
            [4, 0, 0.0],
            [5, 0, 0.0],
            [6, 0, 0.0],
            [0, 0, 99.0],
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
    np.testing.assert_allclose(calc.path_distances(locs), expected, atol=1e-12)
    np.testing.assert_allclose(
        calc.path_distances(locs.iloc[:3], locs.iloc[3:6]),
        [[10.0, 25.0, 20.0], [5.0, 20.0, 15.0], [0.0, 15.0, 10.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calc.path_distances(locs.iloc[:4], same_section_only=True),
        [
            [0.0, 5.0, 10.0, np.nan],
            [5.0, 0.0, 5.0, np.nan],
            [10.0, 5.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan, 0.0],
        ],
        equal_nan=True,
        atol=1e-12,
    )


def test_path_distances_match_main_values_on_nonround_geometry(tiny):
    calc = MorphologyPathDistanceCalculator(tiny)
    tips = locs_df([[2, 0, 7.3], [3, 0, 9.1]])
    np.testing.assert_allclose(
        calc.path_distances(tips),
        [[0.0, 16.4], [16.4, 0.0]],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        calc.path_distances(locs_df([[1, 0, 0.0]]), locs_df([[2, 0, 7.3]])),
        [[20.2]],
        atol=1e-10,
    )


def test_path_distances_match_main_values_on_long_chain():
    morph, seg = build_chain_morphology(n_sections=80, seg_len=97.3)
    calc = MorphologyPathDistanceCalculator(morph)
    start = locs_df([[1, 0, 0.0]])
    end = locs_df([[80, 0, seg]])
    np.testing.assert_allclose(
        calc.path_distances(start, end),
        [[80 * 97.3]],
        atol=1e-9,
    )


def test_nearest_neighbor_unnormalized_matches_main_values(calc):
    locs = locs_df(
        [
            [3, 0, 1.0],
            [3, 0, 9.0],
            [4, 0, 2.0],
            [5, 0, 3.0],
            [2, 0, 4.0],
        ]
    )
    locs.index = pd.Index(["g1", "g1", "g1", "g2", "g2"], name="group")
    got = calc.nearest_neighbor_distances(locs, normalize=False)
    assert list(got.index) == ["g1", "g2"]
    np.testing.assert_allclose(got["g1"], [3.0, 8.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(got["g2"], [7.0, 7.0], atol=1e-12)


def test_custom_column_names_match_main_values(calc):
    locs = pd.DataFrame({"sid": [1, 3], "seg": [1, 0], "off": [2.5, 4.0]})
    kwargs = dict(str_section_id="sid", str_segment_id="seg", str_offset="off")
    np.testing.assert_allclose(
        calc.within_section_offsets(locs, **kwargs),
        [7.5, 4.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calc.path_distances(locs, **kwargs),
        [[0.0, 16.5], [16.5, 0.0]],
        atol=1e-12,
    )


def test_offset_advanced_indexing_matches_dense_table(branched):
    """obi-one indexes offset with parallel section/segment arrays."""
    calc = MorphologyPathDistanceCalculator(branched)
    dense = as_offset_array(calc.offset)
    sec = np.array([0, 1, 2, 5])
    seg = np.array([0, 0, 0, 0])
    np.testing.assert_allclose(calc.offset[sec, seg], dense[sec, seg])


def test_offset_view_matches_dense_including_nan_padding(branched):
    calc = MorphologyPathDistanceCalculator(branched)
    dense_static = MorphologyPathDistanceCalculator.segment_offset_table(branched)
    dense_view = as_offset_array(calc.offset)
    np.testing.assert_allclose(dense_view, dense_static, equal_nan=True)
    assert np.isnan(calc.offset[1, dense_view.shape[1] - 1]) or dense_view.shape[1] == 2


def test_offset_flat_storage_saves_memory_on_uneven_sections():
    """One long section + many short ones: dense padding dominates flat cumlen."""
    mut = MutMorph()
    long_pts = [[float(i), 0.0, 0.0] for i in range(200)]
    long_dia = [1.0] * 200
    root = mut.append_root_section(PointLevel(long_pts, long_dia), SectionType.axon)
    tip = long_pts[-1]
    for k in range(50):
        root.append_section(
            PointLevel([tip, [tip[0], float(k + 1), 0.0]], [1.0, 1.0]),
            SectionType.axon,
        )
    morph = morphio.Morphology(mut)
    calc = MorphologyPathDistanceCalculator(morph)
    dense = as_offset_array(calc.offset)
    assert calc.offset.nbytes < dense.nbytes
    np.testing.assert_allclose(
        dense, MorphologyPathDistanceCalculator.segment_offset_table(morph), equal_nan=True
    )
