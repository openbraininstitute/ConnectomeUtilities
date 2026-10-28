# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and reference loader for MorphologyPathDistanceCalculator tests."""
import importlib.util
import subprocess
from pathlib import Path

import morphio
import numpy as np
import pandas as pd
from morphio import PointLevel, SectionType
from morphio.mut import Morphology as MutMorph

from conntility.subcellular import MorphologyPathDistanceCalculator

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAIN_MODULE_CACHE = None


def locs_df(rows, columns=None):
    if columns is None:
        columns = [
            "afferent_section_id",
            "afferent_segment_id",
            "afferent_segment_offset",
        ]
    return pd.DataFrame(rows, columns=columns)


def build_branched_morphology():
    """
    Known-geometry tree (section lengths = 10):

        soma
        ├── sec0 (basal): (0,0,0)->(5,0,0)->(10,0,0)
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


def build_tiny_nonround_morphology():
    """Root + two children with non-round lengths (12.9 / 7.3 / 9.1)."""
    mut = MutMorph()
    root = mut.append_root_section(
        PointLevel([[0, 0, 0], [12.9, 0, 0]], [1, 1]), SectionType.axon
    )
    root.append_section(PointLevel([[12.9, 0, 0], [20.2, 0, 0]], [1, 1]), SectionType.axon)
    root.append_section(PointLevel([[12.9, 0, 0], [12.9, 9.1, 0]], [1, 1]), SectionType.axon)
    return morphio.Morphology(mut)


def build_chain_morphology(n_sections=50, seg_len=97.3):
    mut = MutMorph()
    parent = mut.append_root_section(
        PointLevel([[0, 0, 0], [seg_len, 0, 0]], [1, 1]), SectionType.axon
    )
    for i in range(1, n_sections):
        x0 = i * seg_len
        parent = parent.append_section(
            PointLevel([[x0, 0, 0], [x0 + seg_len, 0, 0]], [1, 1]), SectionType.axon
        )
    return morphio.Morphology(mut), seg_len


def load_main_morphology_path_distance_calculator():
    """Load MorphologyPathDistanceCalculator as it existed on origin/main (pairwise D/R)."""
    global _MAIN_MODULE_CACHE
    if _MAIN_MODULE_CACHE is not None:
        return _MAIN_MODULE_CACHE.MorphologyPathDistanceCalculator

    src = subprocess.check_output(
        ["git", "show", "main:conntility/subcellular/neuron_morphology_path_distance.py"],
        cwd=_REPO_ROOT,
        text=True,
    )
    path = Path("/tmp/mpdc_main_original_for_tests.py")
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("mpdc_main_original_for_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MAIN_MODULE_CACHE = mod
    return mod.MorphologyPathDistanceCalculator


def as_offset_array(offset):
    """ densify public offset table (ndarray or array-like wrapper). """
    return np.asarray(offset, dtype=float)


def assert_public_offset_api(calc):
    """Contract used by obi-one: 2d indexable offset[sec, seg] (scalar and array)."""
    offset = calc.offset
    arr = as_offset_array(offset)
    assert arr.ndim == 2
    assert len(offset) == arr.shape[0]
    # scalar
    v0 = offset[0, 0]
    assert np.isscalar(v0) or np.ndim(v0) == 0
    assert float(v0) == 0.0
    # advanced indexing like obi-one / map_synapse_locations
    sec = np.array([0, 0, min(1, arr.shape[0] - 1)])
    seg = np.array([0, 0, 0])
    got = offset[sec, seg]
    assert np.asarray(got).shape == (3,)
    np.testing.assert_allclose(got, arr[sec, seg], equal_nan=True)
