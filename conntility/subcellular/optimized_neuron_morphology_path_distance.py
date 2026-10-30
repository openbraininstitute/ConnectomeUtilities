# SPDX-License-Identifier: Apache-2.0
"""Memory-optimized morphology path-distance calculator (LCA / O(n) tables)."""
from collections import deque

import numpy

_FLOAT_DTYPE = numpy.float32


class OptimizedMorphologyPathDistanceCalculator(object):
    """
    Low-memory alternative to MorphologyPathDistanceCalculator.

    Same ``within_section_offsets`` / ``path_distances`` results, using O(n) tree
    tables and on-demand LCA instead of dense O(n²) section tables.
    """

    def __init__(self, morphology):
        """
        Args:
          morphology (morphio.Morphology): The morphology you want to calculate path distances for.
        """
        connectivity = morphology.connectivity
        assert 0 in connectivity[-1], (
            "This class requires the first section to be connected to the soma!"
        )
        n_sections = morphology.n_sections
        self._n_sections = n_sections
        self._section_offsets = numpy.asarray(morphology.section_offsets, dtype=numpy.int64)
        self._cumlen = _cumulative_point_path_lengths(morphology.points, self._section_offsets)
        section_length = self._cumlen[self._section_offsets[1:] - 1]
        (
            self._path_to_root,
            self._dfs_time,
            self._ancestor_jump,
        ) = _section_tree_tables(connectivity, n_sections, section_length)

    def within_section_offsets(
        self,
        locs,
        str_section_id="afferent_section_id",
        str_segment_id="afferent_segment_id",
        str_offset="afferent_segment_offset",
    ):
        """Same contract as MorphologyPathDistanceCalculator.within_section_offsets."""
        section_ids = locs[str_section_id].to_numpy()
        segment_ids = locs[str_segment_id].to_numpy().astype(numpy.int64, copy=False)
        # Dense offset[-1] is the last section (soma id 0 → index -1); match that wrap.
        sec = (section_ids.astype(numpy.int64, copy=False) - 1) % self._n_sections
        beg = self._section_offsets[sec]
        n_pts = self._section_offsets[sec + 1] - beg
        within_section = numpy.full(sec.shape, numpy.nan, dtype=_FLOAT_DTYPE)
        in_range = (segment_ids >= 0) & (segment_ids < n_pts)
        within_section[in_range] = self._cumlen[beg[in_range] + segment_ids[in_range]]
        within_section = within_section + numpy.asarray(
            locs[str_offset].to_numpy(), dtype=_FLOAT_DTYPE
        )
        within_section[section_ids == 0] = 0.0
        return within_section

    def path_distances(
        self,
        locs_from,
        locs_to=None,
        str_section_id="afferent_section_id",
        str_segment_id="afferent_segment_id",
        str_offset="afferent_segment_offset",
        same_section_only=False,
    ):
        """Same contract as MorphologyPathDistanceCalculator.path_distances."""
        if locs_to is None:
            locs_to = locs_from

        offset_from = self.within_section_offsets(
            locs_from,
            str_section_id=str_section_id,
            str_segment_id=str_segment_id,
            str_offset=str_offset,
        )
        offset_to = self.within_section_offsets(
            locs_to,
            str_section_id=str_section_id,
            str_segment_id=str_segment_id,
            str_offset=str_offset,
        )
        if same_section_only:
            section_ids_from = locs_from[str_section_id].to_numpy()
            section_ids_to = locs_to[str_section_id].to_numpy()
            on_same_section = section_ids_from[:, None] == section_ids_to[None, :]
            distances = numpy.full(on_same_section.shape, numpy.nan, dtype=float)
            row, col = numpy.nonzero(on_same_section)
            distances[row, col] = numpy.abs(offset_from[row] - offset_to[col])
            return distances

        section_from = locs_from[str_section_id].to_numpy(dtype=numpy.int32, copy=True) - 1
        section_to = locs_to[str_section_id].to_numpy(dtype=numpy.int32, copy=True) - 1
        section_from[section_from < 0] = 0
        section_to[section_to < 0] = 0

        return _path_distances_vectorized(
            section_from,
            section_to,
            offset_from,
            offset_to,
            self._dfs_time,
            self._path_to_root,
            self._cumlen,
            self._section_offsets,
            self._ancestor_jump,
        )


def _cumulative_point_path_lengths(points, section_offsets):
    """Per-point distance from the owning section's first point."""
    cumlen = numpy.empty(len(points), dtype=_FLOAT_DTYPE)
    for section_id in range(len(section_offsets) - 1):
        beg = int(section_offsets[section_id])
        end = int(section_offsets[section_id + 1])
        cumlen[beg] = 0.0
        if end - beg > 1:
            segment_lengths = numpy.sqrt(
                numpy.sum(numpy.diff(points[beg:end], axis=0) ** 2, axis=1)
            )
            cumlen[beg + 1:end] = numpy.cumsum(segment_lengths)
    return cumlen


def _section_tree_tables(connectivity, n_sections, section_length):
    parent_section = numpy.full(n_sections, -1, dtype=numpy.int32)
    path_to_root = numpy.zeros(n_sections, dtype=_FLOAT_DTYPE)
    dfs_time = numpy.full((n_sections, 2), -1, dtype=numpy.int32)

    for parent_id, child_ids in connectivity.items():
        if parent_id < 0:
            continue
        for child_id in child_ids:
            parent_section[child_id] = parent_id

    dfs_clock = 0
    stack = deque((root_id, 0) for root_id in reversed(connectivity.get(-1, ())))
    while stack:
        section_id, next_child_index = stack.pop()
        child_ids = connectivity.get(section_id, ())
        if next_child_index == 0:
            dfs_time[section_id, 0] = dfs_clock
            dfs_clock += 1
            parent_id = parent_section[section_id]
            if parent_id >= 0:
                path_to_root[section_id] = (
                    path_to_root[parent_id] + section_length[parent_id]
                )
        if next_child_index < len(child_ids):
            stack.append((section_id, next_child_index + 1))
            stack.append((child_ids[next_child_index], 0))
        else:
            dfs_time[section_id, 1] = dfs_clock
            dfs_clock += 1

    n_jumps = max(1, n_sections.bit_length())
    ancestor_jump = numpy.full((n_jumps, n_sections), -1, dtype=numpy.int32)
    ancestor_jump[0] = parent_section
    for jump_level in range(1, n_jumps):
        previous_jump = ancestor_jump[jump_level - 1]
        has_ancestor = previous_jump >= 0
        ancestor_jump[jump_level, has_ancestor] = previous_jump[
            previous_jump[has_ancestor]
        ]

    return path_to_root, dfs_time, ancestor_jump


def _path_to_section_end(path_to_root, cumlen, section_offsets, section_ids):
    last_point = section_offsets[section_ids + 1] - 1
    return path_to_root[section_ids] + cumlen[last_point]


def _section_pair_relations_and_root_distances(
    section_from,
    section_to,
    dfs_time,
    path_to_root,
    cumlen,
    section_offsets,
    ancestor_jump,
):
    n_from = len(section_from)
    n_to = len(section_to)

    dfs_enter = dfs_time[:, 0]
    dfs_exit = dfs_time[:, 1]
    enter_from = dfs_enter[section_from][:, None]
    exit_from = dfs_exit[section_from][:, None]
    enter_to = dfs_enter[section_to][None, :]
    exit_to = dfs_exit[section_to][None, :]

    same = section_from[:, None] == section_to[None, :]
    to_anc = (enter_to <= enter_from) & (exit_from <= exit_to) & ~same
    from_anc = (enter_from <= enter_to) & (exit_to <= exit_from) & ~same
    coll = ~(same | to_anc | from_anc)

    climb = numpy.empty((n_from, n_to), dtype=section_from.dtype)
    climb[:] = section_from[:, None]
    jumped = numpy.empty_like(climb)
    safe = numpy.empty_like(climb)

    for jump_level in range(ancestor_jump.shape[0] - 1, -1, -1):
        numpy.maximum(climb, 0, out=safe)
        jumped[:] = ancestor_jump[jump_level, safe]
        jumped[climb < 0] = -1
        numpy.maximum(jumped, 0, out=safe)
        jumped_is_ancestor_of_to = (
            (jumped >= 0)
            & (dfs_enter[safe] <= enter_to)
            & (exit_to <= dfs_exit[safe])
        )
        should_jump = (
            coll & (climb >= 0) & (jumped >= 0) & ~jumped_is_ancestor_of_to
        )
        climb[should_jump] = jumped[should_jump]

    jumped[:] = -1
    valid_climb = climb >= 0
    jumped[valid_climb] = ancestor_jump[0, climb[valid_climb]]
    lca = jumped

    path_from = path_to_root[section_from][:, None]
    path_to = path_to_root[section_to][None, :]
    root_distance = numpy.zeros((n_from, n_to), dtype=_FLOAT_DTYPE)
    root_distance[to_anc] = (path_from - path_to)[to_anc]
    root_distance[from_anc] = (path_to - path_from)[from_anc]

    rows, cols = numpy.nonzero(coll)
    lca_ids = lca[rows, cols]
    branch = numpy.zeros(len(lca_ids), dtype=_FLOAT_DTYPE)
    has_section_lca = lca_ids >= 0
    if numpy.any(has_section_lca):
        branch[has_section_lca] = _path_to_section_end(
            path_to_root, cumlen, section_offsets, lca_ids[has_section_lca]
        )
    root_distance[rows, cols] = (
        path_to_root[section_from[rows]]
        + path_to_root[section_to[cols]]
        - 2.0 * branch
    )
    return same, to_anc, from_anc, coll, root_distance


def _apply_offsets_to_section_distances(
    same, to_anc, from_anc, coll, root_distance, offset_from, offset_to
):
    n_from = len(offset_from)
    n_to = len(offset_to)
    distances = numpy.empty((n_from, n_to), dtype=_FLOAT_DTYPE)

    rows, cols = numpy.nonzero(same)
    distances[rows, cols] = numpy.abs(offset_from[rows] - offset_to[cols])

    rows, cols = numpy.nonzero(to_anc)
    distances[rows, cols] = (
        root_distance[rows, cols] + offset_from[rows] - offset_to[cols]
    )

    rows, cols = numpy.nonzero(from_anc)
    distances[rows, cols] = (
        root_distance[rows, cols] - offset_from[rows] + offset_to[cols]
    )

    rows, cols = numpy.nonzero(coll)
    distances[rows, cols] = (
        root_distance[rows, cols] + offset_from[rows] + offset_to[cols]
    )
    return distances


def _path_distances_vectorized(
    section_from,
    section_to,
    offset_from,
    offset_to,
    dfs_time,
    path_to_root,
    cumlen,
    section_offsets,
    ancestor_jump,
):
    same, to_anc, from_anc, coll, root_distance = (
        _section_pair_relations_and_root_distances(
            section_from,
            section_to,
            dfs_time,
            path_to_root,
            cumlen,
            section_offsets,
            ancestor_jump,
        )
    )
    return _apply_offsets_to_section_distances(
        same, to_anc, from_anc, coll, root_distance, offset_from, offset_to
    )
