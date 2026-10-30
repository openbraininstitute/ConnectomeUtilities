# SPDX-License-Identifier: Apache-2.0
"""OptimizedMorphologyPathDistanceCalculator matches MorphologyPathDistanceCalculator results."""
import numpy as np
import pandas as pd
import pytest

from conntility.subcellular import (
    MorphologyPathDistanceCalculator,
    OptimizedMorphologyPathDistanceCalculator,
)
from mpdc_test_utils import (
    build_branched_morphology,
    build_chain_morphology,
    build_tiny_nonround_morphology,
    locs_df,
)


@pytest.fixture(scope="module")
def branched():
    return build_branched_morphology()


@pytest.fixture(scope="module")
def ref(branched):
    return MorphologyPathDistanceCalculator(branched)


@pytest.fixture(scope="module")
def opt(branched):
    return OptimizedMorphologyPathDistanceCalculator(branched)


def test_init_requires_first_section_rooted_at_soma():
    class _FakeMorph:
        connectivity = {-1: [1], 1: [0]}
        n_sections = 2

    with pytest.raises(AssertionError, match="first section"):
        OptimizedMorphologyPathDistanceCalculator(_FakeMorph())


def test_within_section_offsets_match_original(ref, opt):
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
        opt.within_section_offsets(locs),
        ref.within_section_offsets(locs),
        rtol=0,
        atol=0,
    )


def test_path_distances_match_original_pairwise_and_rectangular(ref, opt):
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
    np.testing.assert_allclose(
        opt.path_distances(locs), ref.path_distances(locs), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        opt.path_distances(locs.iloc[:3], locs.iloc[3:6]),
        ref.path_distances(locs.iloc[:3], locs.iloc[3:6]),
        rtol=0,
        atol=1e-6,
    )


def test_path_distances_same_section_only_match_original(ref, opt):
    locs = locs_df(
        [
            [1, 0, 0.0],
            [1, 1, 0.0],
            [1, 1, 5.0],
            [2, 0, 0.0],
        ]
    )
    got = opt.path_distances(locs, same_section_only=True)
    expected = ref.path_distances(locs, same_section_only=True)
    np.testing.assert_allclose(got, expected, equal_nan=True, rtol=0, atol=1e-6)


def test_custom_column_names_match_original(ref, opt):
    locs = pd.DataFrame({"sid": [1, 3], "seg": [1, 0], "off": [2.5, 4.0]})
    kwargs = dict(str_section_id="sid", str_segment_id="seg", str_offset="off")
    np.testing.assert_allclose(
        opt.within_section_offsets(locs, **kwargs),
        ref.within_section_offsets(locs, **kwargs),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        opt.path_distances(locs, **kwargs),
        ref.path_distances(locs, **kwargs),
        rtol=0,
        atol=1e-6,
    )


def test_nonround_and_long_chain_match_original():
    tiny = build_tiny_nonround_morphology()
    tips = locs_df([[2, 0, 7.3], [3, 0, 9.1]])
    np.testing.assert_allclose(
        OptimizedMorphologyPathDistanceCalculator(tiny).path_distances(tips),
        MorphologyPathDistanceCalculator(tiny).path_distances(tips),
        rtol=0,
        atol=1e-5,
    )

    morph, seg = build_chain_morphology(n_sections=80, seg_len=97.3)
    start = locs_df([[1, 0, 0.0]])
    end = locs_df([[80, 0, seg]])
    np.testing.assert_allclose(
        OptimizedMorphologyPathDistanceCalculator(morph).path_distances(start, end),
        MorphologyPathDistanceCalculator(morph).path_distances(start, end),
        rtol=0,
        atol=1e-4,
    )
