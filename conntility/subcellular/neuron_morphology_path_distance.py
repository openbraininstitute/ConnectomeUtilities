# SPDX-License-Identifier: Apache-2.0
from collections import deque

import morphio
import numpy


class SectionOffsetTable(object):
    """
    Compact within-section path offsets backed by MorphIO flat point storage.

    Public indexing matches the historical dense table: ``offset[section_id, segment_id]``
    (0-based section ids, no soma row). Out-of-range segment ids yield NaN, like the
    padded dense representation. ``numpy.asarray(offset)`` materializes the dense table.
    """

    def __init__(self, path_length_from_section_root, section_offsets):
        self._path_length_from_section_root = path_length_from_section_root
        self._section_offsets = section_offsets
        self._n_sections = len(self._section_offsets) - 1
        points_per_section = numpy.diff(self._section_offsets)
        self._max_points = int(points_per_section.max()) if self._n_sections else 0
        self._points_per_section = points_per_section

    @property
    def shape(self):
        return (self._n_sections, self._max_points)

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        return self._path_length_from_section_root.dtype

    @property
    def nbytes(self):
        return (
            self._path_length_from_section_root.nbytes + self._section_offsets.nbytes
        )

    def __len__(self):
        return self._n_sections

    def section_lengths(self):
        """Euclidean length of each section (path length to the last point)."""
        if self._n_sections == 0:
            return numpy.zeros(0, dtype=float)
        last_point_index = self._section_offsets[1:] - 1
        return self._path_length_from_section_root[last_point_index]

    def to_dense(self):
        table = numpy.full(self.shape, numpy.nan, dtype=float)
        for section_id in range(self._n_sections):
            beg = int(self._section_offsets[section_id])
            end = int(self._section_offsets[section_id + 1])
            table[section_id, : end - beg] = (
                self._path_length_from_section_root[beg:end]
            )
        return table

    def __array__(self, dtype=None):
        dense = self.to_dense()
        if dtype is not None:
            return numpy.asarray(dense, dtype=dtype)
        return dense

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("SectionOffsetTable requires two indices [section, segment]")
        section_id, segment_id = key
        section_id = numpy.asarray(section_id)
        segment_id = numpy.asarray(segment_id)
        # Match ndarray negative indexing on axis 0 (soma locs use section_id-1 == -1).
        section_id = numpy.where(section_id < 0, section_id + self._n_sections, section_id)
        is_scalar = section_id.ndim == 0 and segment_id.ndim == 0

        if is_scalar:
            n_points = int(self._points_per_section[int(section_id)])
            segment_id = int(segment_id)
            if segment_id < 0 or segment_id >= n_points:
                return numpy.nan
            point_index = int(self._section_offsets[int(section_id)]) + segment_id
            return float(self._path_length_from_section_root[point_index])

        section_id, segment_id = numpy.broadcast_arrays(section_id, segment_id)
        section_id = section_id.astype(numpy.int64, copy=False)
        segment_id = segment_id.astype(numpy.int64, copy=False)
        values = numpy.full(section_id.shape, numpy.nan, dtype=float)
        point_index = self._section_offsets[section_id] + segment_id
        in_range = (segment_id >= 0) & (segment_id < self._points_per_section[section_id])
        values[in_range] = self._path_length_from_section_root[point_index[in_range]]
        return values


class MorphologyPathDistanceCalculator(object):
    """
    A helper to quickly calculate path distances along the morphology of a neuron.

    Relies on pre-calculating section-tree metadata (parent links, root path lengths,
    and within-section segment offsets) for a single morphology. This is only really
    worth the cost when you want to look up distances between hundreds of points on a
    single morphology. For smaller use cases (or cases where the distances are to be
    calculated on many different morphologies) other code may be faster.

    NOTE: This class makes the strong assumption that the first section in the morphology
    is a direct descendant of the soma.
    """

    def __init__(self, morphology: morphio.Morphology):
        """
        Args:
          morphology (morphio.Morphology): The morphology you want to calculate path distances for.
        """
        self.m = morphology
        # MorphIO returns a fresh dict/lists on each connectivity access — snapshot once.
        connectivity = morphology.connectivity
        assert 0 in connectivity[-1], (
            "This class requires the first section to be connected to the soma!"
        )
        self._n_sections = morphology.n_sections
        # Compact flat path lengths; public .offset keeps dense-table indexing semantics.
        self.offset = _section_offset_table(morphology)
        (
            self._parent_section,
            self._path_length_to_section_root,
            self._section_length,
            self._dfs_enter,
            self._dfs_exit,
            self._ancestor_jump,
        ) = _section_tree_tables(
            connectivity, self._n_sections, self.offset.section_lengths()
        )

    @staticmethod
    def segment_offset_table(m):
        """
        Calculate for each segment in each section the distance from the root of the section to the
        root of the segment. Since different sections have different numbers of segments, all lists
        of offsets are padded with NaN and stacked to a 2d array.

        Built from MorphIO's flat ``points`` / ``section_offsets`` arrays (no Section wrappers).

        Args:
          m (morphio.Morphology): The morphogy in question

        Returns:
          offset_table (numpy.array): Table of segment offsets. To be indexed as
          offset_table[sec_id][seg_id].
          Note: sec_id is base-0, i.e. index 0 is NOT the soma. The soma is not listed
          at all (assumed to be just a point).
        """
        return _section_offset_table(m).to_dense()

    def within_section_offsets(self, locs,
                               str_section_id="afferent_section_id",
                               str_segment_id="afferent_segment_id",
                               str_offset="afferent_segment_offset"):
        """
        Calculate the offset of points within a section.

        Args:
          locs (pandas.DataFrame): dataframe where columns specify the section id, segment id and within-segment offset
          of points on the neuron morphology.
          str_section_id (default="afferent_section_id"): The name of the column holding the section id.
          str_segment_id (default="afferent_segment_id"): The name of the column holding the segment id.
          str_offset (defaults="afferent_segment_offset"): The name of the column holding the within-segment offset.

        Note: The default column names are set to what bluepysnap returns when asked for afferent synapses.
        Also, the section id is expected to be base-1 indexed. That is, index 0 is the soma,
        index 1 is the first "proper" section.
        As path distance along the soma is a tricky concept, the function will return 0.0
        for all entries on the soma!

        Returns:
          offsets (numpy.array): The within-section offsets in the same order as the input.
        """
        section_ids = locs[str_section_id].values
        within_section = self.offset[section_ids - 1, locs[str_segment_id]]
        assert not numpy.any(numpy.isnan(within_section))
        within_section = within_section + locs[str_offset].values
        within_section[section_ids == 0] = 0.0
        return within_section

    def path_distances(self, locs_from, locs_to=None,
                       str_section_id="afferent_section_id",
                       str_segment_id="afferent_segment_id",
                       str_offset="afferent_segment_offset",
                       same_section_only=False):
        """
        Calculate path distances. Either pairwise, or between two sets of points.
        Args:
          locs_from (pandas.DataFrame): Locations of points on the morphology represented
          by this object to calculate the distances from. A dataframe where columns specify
          the section id, segment id and within-segment offset of points on the neuron morphology.
          locs_to (pandas.DataFrame, default=None): Locations to calculate the distances to. If none, locs_from is used.
          str_section_id (default="afferent_section_id"): The name of the column holding the section id.
          str_segment_id (default="afferent_segment_id"): The name of the column holding the segment id.
          str_offset (defaults="afferent_segment_offset"): The name of the column holding the within-segment offset.
          same_section_only (bool, default=False): If True, set values for all pairs NOT on the same section to NaN.

          Note: The default column names are set to what bluepysnap returns when asked for afferent synapses.
          Also, the section id is expected to be base-1 indexed. That is, index 0 is the soma,
          index 1 is the first "proper" section.
          This is also compatible with how bluepysnap returns the information.
          Since the concept of path distance along the soma is tricky, this method assumes
          that the soma is a point. Consequently, for all locations on the soma, the values
          will be as if the location was at the root of a root section.

        Returns:
          dist (numpy.array): Array of pairwise path distances.
        """
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
            # +1 for soma id 0; use cached section count (not len(m.sections)).
            on_same_section = numpy.eye(self._n_sections + 1, dtype=bool)
            on_same_section = on_same_section[
                numpy.ix_(locs_from[str_section_id], locs_to[str_section_id])
            ]
            distances = numpy.full(on_same_section.shape, numpy.nan, dtype=float)
            row, col = numpy.nonzero(on_same_section)
            distances[row, col] = numpy.abs(offset_from[row] - offset_to[col])
            return distances

        section_from = locs_from[str_section_id].to_numpy(dtype=numpy.int32, copy=True) - 1
        section_to = locs_to[str_section_id].to_numpy(dtype=numpy.int32, copy=True) - 1
        # Since we test above that the first section is rooted at the soma, we can use that
        # section in place of the soma. This makes sense, because we assume the soma to be a
        # point (and consequently, within_section_offset) returns 0.0 for all locations on the soma.
        section_from[section_from < 0] = 0
        section_to[section_to < 0] = 0

        return _path_distances_vectorized(
            section_from,
            section_to,
            offset_from,
            offset_to,
            self._dfs_enter,
            self._dfs_exit,
            self._path_length_to_section_root,
            self._section_length,
            self._parent_section,
            self._ancestor_jump,
        )

    @staticmethod
    def __nnd__(distance_matrix, location_indices):
        if len(location_indices) <= 1:
            return numpy.nan
        return numpy.nanmin(
            distance_matrix[numpy.ix_(location_indices, location_indices)], axis=0
        )

    @staticmethod
    def __rnd_sampler__(locs, prop_col, n_rnd):
        import pandas
        if prop_col is None:
            locations_by_key = locs["index"].reset_index(drop=True)
        else:
            locations_by_key = locs.set_index(prop_col)["index"]
        sample_keys = pandas.Series(numpy.unique(locations_by_key.index))

        def sampler(sample_size):
            for _ in range(n_rnd):
                yield locations_by_key[sample_keys.sample(sample_size)].values
        return sampler

    def nearest_neighbor_distances(self, locs, normalize=False,
                                   normalize_preserve=None,
                                   normalize_n=10,
                                   str_section_id="afferent_section_id",
                                   str_segment_id="afferent_segment_id",
                                   str_offset="afferent_segment_offset",
                                   same_section_only=False):
        """
        Analyzes the nearest neighbor distances between sets of dendritic locations and optionally
        normalizes them against random controls.

        Args:
          locs (pandas.DataFrame): Locations of points on the morphology represented by this object.
          For details, see MorphologyPathDistanceCalculator.path_distances. Additionally, must have
          an index that defines a grouping (see below).
          str_section_id (str, optional): See MorphologyPathDistanceCalculator.path_distances.
          str_segment_id (str, optional): See MorphologyPathDistanceCalculator.path_distances.
          str_offset (str, optional): See MorphologyPathDistanceCalculator.path_distances.
          same_section_only (bool, default: False): See MorphologyPathDistanceCalculator.path_distances.
          normalize (bool, default: False): Whether to normalize the results by converting to a
          z-score with respect to a random control
          normalize_preserve (optional): Name of a column in locs that defines a grouping.
          When normalizing and this is specified, then locations belonging to the same group will be kept
          together in the random controls. Ignored if normalize=False
          normalize_n (int, default: 10): How many random controls to generate for normalization.
          Ignored if normalize=False.

        Explaining this analysis in the context of locations of synapses on dendritic trees and their
        (potential) clustering.
        The DataFrame locs defines the locations of all afferent synapses onto the morphology in terms of
        section id, segment id and offset. It has an index (can be MultiIndex) that groups them into any
        number of groups by their unique values. This can be the name of the source region, identifier of
        a structural or functional cluster, or simply a bool specifying whether the synapse is of interest
        or not. The analysis then investigates whether the synapses belonging to the same group are
        clustered, i.e. closer together than expected, or not, and to what degree.

        res = M.nearest_neighbor_distances(locs, normalize=False) simply returns a DataFrame with the
        nearest neighbor distances of synapses in each group. It has one row per unique value of the input
        index, and values are lists with the nearest neighbor distances of all synapses associated with
        that index.

        res = M.nearest_neighbor_distances(locs, normalize=True) converts the distances into z-scores
        by comparing to a random control. The control is generated by randomly sampling the same number of
        locations from locs. That is, it compares the distances in each group to the distances in random
        groups of the same sizes.

        res = M.nearest_neighor_distances(locs, normalize=True, normalize_preserve="some_property")
        slightly adjusts how the random control is generates. The value of normalize_preserve must be the
        name of a column in locs. Then, in the generation of the controls, locations that have the same
        value in that column will always be sampled together. For example, the column could hold an
        identifier of the presynaptic neuron. Then you can use tis feature to test the clustering of
        inputs from a neuronal assembly while taking into account that inputs from the same presynaptic
        neuron are already likely to be slightly clustered.

        res = M.nearest_neighbor_distance(..., normalize_n=25) controls how many random controls are
        generated for normalization.

        Examples
        """
        pairwise_distances = self.path_distances(
            locs,
            str_section_id=str_section_id,
            str_segment_id=str_segment_id,
            str_offset=str_offset,
            same_section_only=same_section_only,
        )
        numpy.fill_diagonal(pairwise_distances, numpy.nan)
        index_names = list(locs.index.names)
        assert "index" not in index_names
        locs = locs.reset_index().reset_index()
        observed = locs.groupby(index_names)["index"].apply(
            lambda indices: self.__nnd__(pairwise_distances, indices)
        )
        if not normalize:
            return observed

        sampler = self.__rnd_sampler__(locs, normalize_preserve, normalize_n)
        if normalize_preserve is None:
            normalize_preserve = "index"
        sample_sizes = locs.groupby(index_names)[normalize_preserve].apply(
            lambda values: len(values.drop_duplicates())
        )
        controls = sample_sizes.apply(
            lambda size: numpy.hstack(
                [self.__nnd__(pairwise_distances, indices) for indices in sampler(size)]
            )
        )
        return observed.combine(
            controls,
            lambda observed_values, control_values: (
                (numpy.nanmean(observed_values) - numpy.nanmean(control_values))
                / numpy.nanstd(control_values)
            ),
        )


def _cumulative_point_path_lengths(morphology):
    """Per-point distance from the owning section root, aligned with MorphIO points."""
    points = morphology.points
    section_offsets = morphology.section_offsets
    path_length_from_section_root = numpy.empty(len(points), dtype=float)
    for section_id in range(len(section_offsets) - 1):
        beg = section_offsets[section_id]
        end = section_offsets[section_id + 1]
        path_length_from_section_root[beg] = 0.0
        segment_lengths = numpy.sqrt(
            numpy.sum(
                numpy.diff(points[beg:end], axis=0) ** 2,
                axis=1,
            )
        )
        path_length_from_section_root[beg + 1:end] = numpy.cumsum(segment_lengths)
    return path_length_from_section_root, section_offsets


def _section_offset_table(morphology):
    path_length_from_section_root, section_offsets = _cumulative_point_path_lengths(
        morphology
    )
    return SectionOffsetTable(path_length_from_section_root, section_offsets)


def _section_tree_tables(connectivity, n_sections, section_length):
    """
    Build O(n) section-tree tables used for on-demand path distances.

    ``connectivity`` must already be snapshotted from MorphIO (each access allocates).

    Returns:
        parent_section: Parent section id for each section, -1 for roots.
        path_length_to_section_root: Path length from the soma to each section's first point.
        section_length: Euclidean length of each section (same array as passed in).
        dfs_enter, dfs_exit: DFS enter/exit times for O(1) ancestor tests.
        ancestor_jump: Binary-lifting table; ancestor_jump[k, i] is the 2**k-th ancestor of i.
    """
    parent_section = numpy.full(n_sections, -1, dtype=numpy.int32)
    path_length_to_section_root = numpy.zeros(n_sections, dtype=float)
    dfs_enter = numpy.full(n_sections, -1, dtype=numpy.int32)
    dfs_exit = numpy.full(n_sections, -1, dtype=numpy.int32)

    # connectivity is parent_id -> child section ids; -1 is the soma.
    # Prefer this over m.sections / root_sections (those allocate Section wrappers).
    for parent_id, child_ids in connectivity.items():
        if parent_id < 0:
            continue
        for child_id in child_ids:
            parent_section[child_id] = parent_id

    # Roots stay at path length 0 (soma treated as a point). Children attach at the
    # end of their parent, so path_to_root[child] = path_to_root[parent] + length[parent].
    # The same walk records DFS enter/exit times for ancestor queries.
    dfs_clock = 0
    stack = deque(
        (root_id, 0) for root_id in reversed(connectivity.get(-1, ()))
    )
    while stack:
        section_id, next_child_index = stack.pop()
        child_ids = connectivity.get(section_id, ())
        if next_child_index == 0:
            dfs_enter[section_id] = dfs_clock
            dfs_clock += 1
            parent_id = parent_section[section_id]
            if parent_id >= 0:
                path_length_to_section_root[section_id] = (
                    path_length_to_section_root[parent_id] + section_length[parent_id]
                )
        if next_child_index < len(child_ids):
            stack.append((section_id, next_child_index + 1))
            stack.append((child_ids[next_child_index], 0))
        else:
            dfs_exit[section_id] = dfs_clock
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

    return (
        parent_section,
        path_length_to_section_root,
        section_length,
        dfs_enter,
        dfs_exit,
        ancestor_jump,
    )


def _path_distances_vectorized(
    section_from,
    section_to,
    offset_from,
    offset_to,
    dfs_enter,
    dfs_exit,
    path_length_to_section_root,
    section_length,
    parent_section,
    ancestor_jump,
):
    """Vectorized pairwise path distances using O(1) ancestor tests and binary lifting."""
    n_from = len(section_from)
    n_to = len(section_to)
    section_from = numpy.broadcast_to(
        section_from[:, None], (n_from, n_to)
    ).copy()
    section_to = numpy.broadcast_to(
        section_to[None, :], (n_from, n_to)
    ).copy()

    on_same_section = section_from == section_to
    # to is ancestor of from / from is ancestor of to (strict)
    to_is_ancestor = (
        (dfs_enter[section_to] <= dfs_enter[section_from])
        & (dfs_exit[section_from] <= dfs_exit[section_to])
        & ~on_same_section
    )
    from_is_ancestor = (
        (dfs_enter[section_from] <= dfs_enter[section_to])
        & (dfs_exit[section_to] <= dfs_exit[section_from])
        & ~on_same_section
    )
    are_collaterals = ~(on_same_section | to_is_ancestor | from_is_ancestor)

    path_from_root = path_length_to_section_root[section_from]
    path_to_root_at_to = path_length_to_section_root[section_to]
    section_root_distance = numpy.zeros((n_from, n_to), dtype=float)
    section_root_distance[to_is_ancestor] = (
        path_from_root - path_to_root_at_to
    )[to_is_ancestor]
    section_root_distance[from_is_ancestor] = (
        path_to_root_at_to - path_from_root
    )[from_is_ancestor]

    # Collaterals / cousins / other roots: LCA via binary lifting from `from`, then
    # branch point at the end of the LCA section (or soma depth 0).
    climb_from = section_from.copy()
    for jump_level in range(ancestor_jump.shape[0] - 1, -1, -1):
        jumped = numpy.where(
            climb_from >= 0,
            ancestor_jump[jump_level, numpy.maximum(climb_from, 0)],
            -1,
        )
        jumped_is_ancestor_of_to = (
            (jumped >= 0)
            & (dfs_enter[numpy.maximum(jumped, 0)] <= dfs_enter[section_to])
            & (dfs_exit[section_to] <= dfs_exit[numpy.maximum(jumped, 0)])
        )
        should_jump = (
            are_collaterals
            & (climb_from >= 0)
            & (jumped >= 0)
            & ~jumped_is_ancestor_of_to
        )
        climb_from[should_jump] = jumped[should_jump]
    lowest_common_ancestor = numpy.where(
        climb_from >= 0, parent_section[climb_from], -1
    )
    branch_path_length = numpy.zeros((n_from, n_to), dtype=float)
    has_section_lca = are_collaterals & (lowest_common_ancestor >= 0)
    branch_path_length[has_section_lca] = (
        path_length_to_section_root[lowest_common_ancestor]
        + section_length[lowest_common_ancestor]
    )[has_section_lca]
    section_root_distance[are_collaterals] = (
        path_from_root + path_to_root_at_to - 2.0 * branch_path_length
    )[are_collaterals]

    offset_from = offset_from[:, None]
    offset_to = offset_to[None, :]
    distances = numpy.empty((n_from, n_to), dtype=float)
    distances[on_same_section] = numpy.abs(offset_from - offset_to)[on_same_section]
    distances[to_is_ancestor] = (
        section_root_distance + offset_from - offset_to
    )[to_is_ancestor]
    distances[from_is_ancestor] = (
        section_root_distance - offset_from + offset_to
    )[from_is_ancestor]
    distances[are_collaterals] = (
        section_root_distance + offset_from + offset_to
    )[are_collaterals]
    return distances
