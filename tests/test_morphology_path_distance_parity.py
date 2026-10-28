# SPDX-License-Identifier: Apache-2.0
"""Parity of the public API against MorphologyPathDistanceCalculator on main."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conntility.subcellular import MorphologyPathDistanceCalculator
from mpdc_test_utils import (
    as_offset_array,
    assert_public_offset_api,
    build_branched_morphology,
    build_chain_morphology,
    build_tiny_nonround_morphology,
    load_main_morphology_path_distance_calculator,
    locs_df,
)


@pytest.fixture(scope="module")
def MainCalc():
    return load_main_morphology_path_distance_calculator()


@pytest.fixture(scope="module")
def branched():
    return build_branched_morphology()


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_nonround_morphology()


def test_offset_table_matches_main(MainCalc, branched):
    cur = MorphologyPathDistanceCalculator.segment_offset_table(branched)
    ref = MainCalc.segment_offset_table(branched)
    np.testing.assert_allclose(cur, ref, equal_nan=True, rtol=0, atol=0)


def test_instance_offset_matches_main_and_static(MainCalc, branched):
    cur = MorphologyPathDistanceCalculator(branched)
    ref = MainCalc(branched)
    np.testing.assert_allclose(as_offset_array(cur.offset), ref.offset, equal_nan=True)
    np.testing.assert_allclose(
        as_offset_array(cur.offset),
        MorphologyPathDistanceCalculator.segment_offset_table(branched),
        equal_nan=True,
    )
    assert_public_offset_api(cur)


def test_within_section_offsets_match_main(MainCalc, branched):
    cur = MorphologyPathDistanceCalculator(branched)
    ref = MainCalc(branched)
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
        cur.within_section_offsets(locs), ref.within_section_offsets(locs)
    )


def test_path_distances_match_main_on_branched_tree(MainCalc, branched):
    cur = MorphologyPathDistanceCalculator(branched)
    ref = MainCalc(branched)
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
    np.testing.assert_allclose(cur.path_distances(locs), ref.path_distances(locs), atol=1e-12)
    np.testing.assert_allclose(
        cur.path_distances(locs.iloc[:3], locs.iloc[3:6]),
        ref.path_distances(locs.iloc[:3], locs.iloc[3:6]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cur.path_distances(locs.iloc[:4], same_section_only=True),
        ref.path_distances(locs.iloc[:4], same_section_only=True),
        equal_nan=True,
        atol=1e-12,
    )


def test_path_distances_match_main_on_nonround_geometry(MainCalc, tiny):
    cur = MorphologyPathDistanceCalculator(tiny)
    ref = MainCalc(tiny)
    tips = locs_df([[2, 0, 7.3], [3, 0, 9.1]])
    np.testing.assert_allclose(cur.path_distances(tips), ref.path_distances(tips), atol=1e-10)
    root_to_child = locs_df([[1, 0, 0.0]])
    child_tip = locs_df([[2, 0, 7.3]])
    np.testing.assert_allclose(
        cur.path_distances(root_to_child, child_tip),
        ref.path_distances(root_to_child, child_tip),
        atol=1e-10,
    )


def test_path_distances_match_main_on_long_chain(MainCalc):
    morph, seg = build_chain_morphology(n_sections=80, seg_len=97.3)
    cur = MorphologyPathDistanceCalculator(morph)
    ref = MainCalc(morph)
    start = locs_df([[1, 0, 0.0]])
    end = locs_df([[80, 0, seg]])
    np.testing.assert_allclose(
        cur.path_distances(start, end),
        ref.path_distances(start, end),
        atol=1e-9,
    )


def test_nearest_neighbor_unnormalized_matches_main(MainCalc, branched):
    cur = MorphologyPathDistanceCalculator(branched)
    ref = MainCalc(branched)
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
    got = cur.nearest_neighbor_distances(locs, normalize=False)
    expected = ref.nearest_neighbor_distances(locs, normalize=False)
    assert list(got.index) == list(expected.index)
    for key in got.index:
        np.testing.assert_allclose(got[key], expected[key], atol=1e-12)


def test_custom_column_names_match_main(MainCalc, branched):
    cur = MorphologyPathDistanceCalculator(branched)
    ref = MainCalc(branched)
    locs = pd.DataFrame({"sid": [1, 3], "seg": [1, 0], "off": [2.5, 4.0]})
    kwargs = dict(str_section_id="sid", str_segment_id="seg", str_offset="off")
    np.testing.assert_allclose(
        cur.within_section_offsets(locs, **kwargs),
        ref.within_section_offsets(locs, **kwargs),
    )
    np.testing.assert_allclose(
        cur.path_distances(locs, **kwargs),
        ref.path_distances(locs, **kwargs),
        atol=1e-12,
    )


def test_offset_advanced_indexing_matches_dense_table(branched):
    """obi-one indexes offset with parallel section/segment arrays."""
    calc = MorphologyPathDistanceCalculator(branched)
    dense = as_offset_array(calc.offset)
    sec = np.array([0, 1, 2, 5])
    seg = np.array([0, 0, 0, 0])
    np.testing.assert_allclose(calc.offset[sec, seg], dense[sec, seg])
