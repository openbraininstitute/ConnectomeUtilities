# SPDX-License-Identifier: Apache-2.0
import numpy
import pandas
import pytest

morphio = pytest.importorskip("morphio")
from morphio import PointLevel, SectionType  # noqa: E402
from morphio.mut import Morphology  # noqa: E402

from conntility.subcellular import MorphologyPathDistanceCalculator  # noqa: E402


def _tiny_morphology():
    """A root section with two children branching from its end.

        root:   (0,0,0) -> (12.9,0,0)            length 12.9
        child1: (12.9,0,0) -> (20.2,0,0)         length 7.3
        child2: (12.9,0,0) -> (12.9,9.1,0)       length 9.1

    Section ids are 1-based in path_distances (0 is the soma), so the root is
    section 1 and the children are sections 2 and 3. Non-round lengths avoid
    trivially exact float matches.
    """
    mut = Morphology()
    root = mut.append_root_section(PointLevel([[0, 0, 0], [12.9, 0, 0]], [1, 1]), SectionType.axon)
    root.append_section(PointLevel([[12.9, 0, 0], [20.2, 0, 0]], [1, 1]), SectionType.axon)
    root.append_section(PointLevel([[12.9, 0, 0], [12.9, 9.1, 0]], [1, 1]), SectionType.axon)
    return morphio.Morphology(mut)


def test_path_distances_match_tree_geometry():
    pdc = MorphologyPathDistanceCalculator(_tiny_morphology())

    # Between the two child tips: up child 1 to the branch point (7.3), down child 2 (9.1).
    tips = pandas.DataFrame(
        {
            "afferent_section_id": [2, 3],
            "afferent_segment_id": [0, 0],
            "afferent_segment_offset": [7.3, 9.1],
        }
    )
    d = pdc.path_distances(tips)
    assert d.shape == (2, 2)
    numpy.testing.assert_allclose(numpy.diag(d), [0.0, 0.0], atol=1e-4)
    numpy.testing.assert_allclose(d[0, 1], 16.4, atol=1e-3)
    numpy.testing.assert_allclose(d[1, 0], 16.4, atol=1e-3)

    # From the root start to child 1's tip: along the root (12.9) then along the child (7.3).
    root_start = pandas.DataFrame(
        {"afferent_section_id": [1], "afferent_segment_id": [0], "afferent_segment_offset": [0.0]}
    )
    child_tip = pandas.DataFrame(
        {"afferent_section_id": [2], "afferent_segment_id": [0], "afferent_segment_offset": [7.3]}
    )
    numpy.testing.assert_allclose(pdc.path_distances(root_start, child_tip)[0, 0], 20.2, atol=1e-3)


def test_distances_keep_micron_scale_accuracy_over_a_long_path():
    # Confirm that accumulating along a long chain of sections still lands within a small
    # fraction of a micron of the analytic answer.
    mut = Morphology()
    seg = 97.3  # non-round section length
    n_sections = 200
    parent = mut.append_root_section(
        PointLevel([[0, 0, 0], [seg, 0, 0]], [1, 1]), SectionType.axon
    )
    for i in range(1, n_sections):
        x0 = i * seg
        parent = parent.append_section(
            PointLevel([[x0, 0, 0], [x0 + seg, 0, 0]], [1, 1]), SectionType.axon
        )
    m = morphio.Morphology(mut)
    pdc = MorphologyPathDistanceCalculator(m)

    # Root start to the tip of the last section = the whole chain.
    start = pandas.DataFrame(
        {"afferent_section_id": [1], "afferent_segment_id": [0], "afferent_segment_offset": [0.0]}
    )
    end = pandas.DataFrame(
        {
            "afferent_section_id": [n_sections],
            "afferent_segment_id": [0],
            "afferent_segment_offset": [seg],
        }
    )
    expected = n_sections * seg  # 19460.0 um
    numpy.testing.assert_allclose(pdc.path_distances(start, end)[0, 0], expected, atol=0.05)
